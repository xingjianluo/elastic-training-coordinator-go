package checkpoint

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ErrorKind 区分检查点失败原因，调用方可据此分别处理。
type ErrorKind int

const (
	KindOK ErrorKind = iota
	KindPartialWrite
	KindBadMagic
	KindVersionMismatch
	KindCorrupt
	KindMissing
	KindNotFound
	KindNoValid
)

var (
	// ErrPartialWrite 写入被中断：只有 .tmp 临时文件，或文件头/载荷被截断。
	ErrPartialWrite = errors.New("checkpoint: partial write (interrupted temp file or truncated payload)")
	// ErrBadMagic 文件存在完整头但魔数不对，根本不是本组件的检查点。
	ErrBadMagic = errors.New("checkpoint: bad magic, not a checkpoint file")
	// ErrVersionMismatch 魔数正确但格式版本不被支持。
	ErrVersionMismatch = errors.New("checkpoint: version mismatch")
	// ErrCorrupt 文件完整但校验和不符或 JSON 无法解析。
	ErrCorrupt = errors.New("checkpoint: content corrupt (checksum mismatch or unreadable body)")
	// ErrMissing 目录下没有任何检查点。
	ErrMissing = errors.New("checkpoint: no checkpoint found")
	// ErrCheckpointNotFound 指定回滚点存在性检查失败：没有该 step 的正式检查点。
	ErrCheckpointNotFound = errors.New("checkpoint: no checkpoint at requested step")
	// ErrNoValidCheckpoint 目录里虽有文件，但没有一个能通过完整校验。
	ErrNoValidCheckpoint = errors.New("checkpoint: no valid checkpoint available in directory")
)

// FormatVersion 是检查点格式版本。
const FormatVersion uint32 = 1

const (
	magicText  = "ETCKPT01"
	headerLen  = 8 + 4 + 4 // magic + version + payloadLen
	trailerLen = sha256.Size
)

// Member 检查点里保存的成员信息。
type Member struct {
	ID   string `json:"id"`
	Rank int    `json:"rank"`
}

// State 是一次检查点的完整内容。
type State struct {
	Version   uint32    `json:"version"`
	Era       uint64    `json:"era"`
	Step      int       `json:"step"`
	Params    []float64 `json:"params"`
	Velocity  []float64 `json:"velocity"`
	Members   []Member  `json:"members"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Store 负责检查点的原子写、列出与读取。
type Store struct {
	dir string
}

func NewStore(dir string) *Store { return &Store{dir: dir} }

// finalName 按 (step,era) 生成带序号的最终文件名。
func finalName(st *State) string {
	return fmt.Sprintf("ckpt-s%07d-e%07d.etc", st.Step, st.Era)
}

func tempName(st *State) string { return finalName(st) + ".tmp" }

// FinalPath 返回某次状态对应的最终检查点路径（不要求文件已存在）。
func (s *Store) FinalPath(st *State) string {
	return filepath.Join(s.dir, finalName(st))
}

// Save 原子写入：先写 .tmp 并 fsync，再 rename 到最终文件并 fsync 目录。
func (s *Store) Save(st *State) error {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return fmt.Errorf("checkpoint: mkdir: %w", err)
	}
	st.Version = FormatVersion
	body, err := json.Marshal(st)
	if err != nil {
		return fmt.Errorf("checkpoint: marshal: %w", err)
	}
	sum := sha256.Sum256(body)

	final := filepath.Join(s.dir, finalName(st))
	tmp := filepath.Join(s.dir, tempName(st))

	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("checkpoint: open temp: %w", err)
	}
	hdr := make([]byte, headerLen)
	copy(hdr[0:8], magicText)
	binary.BigEndian.PutUint32(hdr[8:12], st.Version)
	binary.BigEndian.PutUint32(hdr[12:16], uint32(len(body)))
	if _, err := f.Write(hdr); err != nil {
		f.Close()
		return fmt.Errorf("checkpoint: write header: %w", err)
	}
	if _, err := f.Write(body); err != nil {
		f.Close()
		return fmt.Errorf("checkpoint: write body: %w", err)
	}
	if _, err := f.Write(sum[:]); err != nil {
		f.Close()
		return fmt.Errorf("checkpoint: write checksum: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("checkpoint: fsync: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("checkpoint: close temp: %w", err)
	}
	if err := os.Rename(tmp, final); err != nil {
		return fmt.Errorf("checkpoint: rename: %w", err)
	}
	// fsync 目录，保证 rename 落盘。
	if d, err := os.Open(s.dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	return nil
}

// SavePartial 仅供测试/故障演练：把不完整内容写到 .tmp 并故意不 rename。
func (s *Store) SavePartial(st *State, bodyBytes int) (string, error) {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return "", err
	}
	st.Version = FormatVersion
	body, _ := json.Marshal(st)
	if bodyBytes >= 0 && bodyBytes < len(body) {
		body = body[:bodyBytes] // 故意截断
	}
	tmp := filepath.Join(s.dir, tempName(st))
	f, err := os.Create(tmp)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hdr := make([]byte, headerLen)
	copy(hdr[0:8], magicText)
	binary.BigEndian.PutUint32(hdr[8:12], st.Version)
	binary.BigEndian.PutUint32(hdr[12:16], uint32(len(body)))
	_, _ = f.Write(hdr)
	_, _ = f.Write(body) // 故意不写校验和、不 fsync、不 rename
	return tmp, nil
}

// writeBadFile 仅供测试：直接写任意字节到一个最终文件名。
func (s *Store) writeBadFile(name string, data []byte) (string, error) {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return "", err
	}
	p := filepath.Join(s.dir, name)
	return p, os.WriteFile(p, data, 0o600)
}

// TempFiles 返回遗留的半截临时文件。
func (s *Store) TempFiles() ([]string, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".tmp") {
			out = append(out, filepath.Join(s.dir, e.Name()))
		}
	}
	sort.Strings(out)
	return out, nil
}

// Latest 返回最新（step,era 最大）的正式检查点路径。
// 没有正式文件时：有遗留 .tmp -> ErrPartialWrite，否则 -> ErrMissing。
func (s *Store) Latest() (string, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", ErrMissing
		}
		return "", err
	}
	type nameKey struct {
		path string
		step int
		era  uint64
	}
	var finals []nameKey
	haveTemp := false
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".tmp") {
			haveTemp = true
			continue
		}
		var stp int
		var era uint64
		if _, err := fmt.Sscanf(name, "ckpt-s%07d-e%07d.etc", &stp, &era); err == nil {
			finals = append(finals, nameKey{filepath.Join(s.dir, name), stp, era})
		}
	}
	if len(finals) == 0 {
		if haveTemp {
			return "", ErrPartialWrite
		}
		return "", ErrMissing
	}
	sort.Slice(finals, func(i, j int) bool {
		if finals[i].step != finals[j].step {
			return finals[i].step < finals[j].step
		}
		return finals[i].era < finals[j].era
	})
	return finals[len(finals)-1].path, nil
}

// Load 读取最新检查点并完整校验。
func (s *Store) Load() (*State, error) {
	p, err := s.Latest()
	if err != nil {
		return nil, err
	}
	return s.LoadPath(p)
}

// LoadPath 读取并校验指定路径，对半截/坏魔数/版本不符/损坏返回不同错误。
func (s *Store) LoadPath(path string) (*State, error) {
	// 任何停留在 .tmp 阶段的文件都视为写入被中断。
	if strings.HasSuffix(path, ".tmp") {
		return nil, fmt.Errorf("%w: %s", ErrPartialWrite, filepath.Base(path))
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrMissing
		}
		return nil, err
	}
	defer f.Close()
	raw, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("%w: cannot read: %v", ErrCorrupt, err)
	}

	// 1. 连魔数都没写全 -> 半截写入。
	if len(raw) < len(magicText) {
		return nil, fmt.Errorf("%w: %s (only %d bytes)", ErrPartialWrite, filepath.Base(path), len(raw))
	}
	// 2. 魔数不对 -> 不是本组件检查点。
	if string(raw[:len(magicText)]) != magicText {
		return nil, fmt.Errorf("%w: %s", ErrBadMagic, filepath.Base(path))
	}
	// 3. 头部不完整 -> 半截写入。
	if len(raw) < headerLen {
		return nil, fmt.Errorf("%w: %s (header truncated)", ErrPartialWrite, filepath.Base(path))
	}
	version := binary.BigEndian.Uint32(raw[8:12])
	bodyLen := binary.BigEndian.Uint32(raw[12:16])
	// 4. 版本不符（优先于截断判定：头完整时版本信息已可信）。
	if version != FormatVersion {
		return nil, fmt.Errorf("%w: file version=%d, supported=%d (%s)",
			ErrVersionMismatch, version, FormatVersion, filepath.Base(path))
	}
	// 5. 声明的载荷+校验和没写全 -> 半截写入。
	if uint32(len(raw)) < uint32(headerLen)+bodyLen+trailerLen {
		return nil, fmt.Errorf("%w: %s (payload/checksum truncated, want %d bytes got %d)",
			ErrPartialWrite, filepath.Base(path), headerLen+int(bodyLen)+trailerLen, len(raw))
	}
	body := raw[headerLen : headerLen+bodyLen]
	sum := raw[headerLen+bodyLen : headerLen+bodyLen+trailerLen]
	// 6. 校验和不符 -> 损坏。
	got := sha256.Sum256(body)
	if string(sum) != string(got[:]) {
		return nil, fmt.Errorf("%w: %s (sha256 mismatch)", ErrCorrupt, filepath.Base(path))
	}
	var st State
	if err := json.Unmarshal(body, &st); err != nil {
		return nil, fmt.Errorf("%w: %s (%v)", ErrCorrupt, filepath.Base(path), err)
	}
	if len(st.Params) != len(st.Velocity) {
		return nil, fmt.Errorf("%w: %s (params/velocity length disagree)", ErrCorrupt, filepath.Base(path))
	}
	if st.Step < 0 {
		return nil, fmt.Errorf("%w: %s (negative step)", ErrCorrupt, filepath.Base(path))
	}
	return &st, nil
}

// KindOf 返回检查点错误的可区分原因。
func KindOf(err error) ErrorKind {
	switch {
	case err == nil:
		return KindOK
	case errors.Is(err, ErrPartialWrite):
		return KindPartialWrite
	case errors.Is(err, ErrBadMagic):
		return KindBadMagic
	case errors.Is(err, ErrVersionMismatch):
		return KindVersionMismatch
	case errors.Is(err, ErrCorrupt):
		return KindCorrupt
	case errors.Is(err, ErrMissing):
		return KindMissing
	case errors.Is(err, ErrCheckpointNotFound):
		return KindNotFound
	case errors.Is(err, ErrNoValidCheckpoint):
		return KindNoValid
	default:
		return KindCorrupt
	}
}
