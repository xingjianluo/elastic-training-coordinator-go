package checkpoint

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Candidate 是目录中一个正式检查点文件的候选条目（按 (step,era) 排序）。
type Candidate struct {
	Path string
	Step int
	Era  uint64
}

// Skipped 记录恢复时被跳过的一个检查点及原因。
type Skipped struct {
	Path string
	Step int
	Era  uint64
	Err  error
}

func (s Skipped) String() string {
	return fmt.Sprintf("%s(s=%d,e=%d): %v", filepath.Base(s.Path), s.Step, s.Era, s.Err)
}

// parseFinalName 仅按文件名解析 (step,era)；不匹配返回 false。
func parseFinalName(name string) (step int, era uint64, ok bool) {
	if strings.HasSuffix(name, ".tmp") {
		return 0, 0, false
	}
	if _, err := fmt.Sscanf(name, "ckpt-s%07d-e%07d.etc", &step, &era); err != nil {
		return 0, 0, false
	}
	return step, era, true
}

// lessKey 按 (step,era) 字典序比较。
func lessKey(s1 int, e1 uint64, s2 int, e2 uint64) bool {
	if s1 != s2 {
		return s1 < s2
	}
	return e1 < e2
}

// ListCandidates 按 (step,era) 升序列出目录中的正式检查点文件
// （只读目录项，不读内容、不校验）。
func (s *Store) ListCandidates() ([]Candidate, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []Candidate
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		step, era, ok := parseFinalName(e.Name())
		if !ok {
			continue
		}
		out = append(out, Candidate{Path: filepath.Join(s.dir, e.Name()), Step: step, Era: era})
	}
	sort.Slice(out, func(i, j int) bool {
		return lessKey(out[i].Step, out[i].Era, out[j].Step, out[j].Era)
	})
	return out, nil
}

// onlyTempErr 复刻 Latest() 在“没有任何正式文件”时的判定：
// 有遗留 .tmp -> ErrPartialWrite，否则 -> ErrMissing。
func (s *Store) onlyTempErr() error {
	tmps, err := s.TempFiles()
	if err != nil {
		return err
	}
	if len(tmps) > 0 {
		return fmt.Errorf("%w: leftover temp files %v", ErrPartialWrite, tmps)
	}
	return ErrMissing
}

// LoadLatestValid 从最新到最旧逐个校验，返回第一个能完整通过校验的检查点；
// skipped 如实记录沿途被跳过（半截/损坏/版本不符等）的条目及可区分原因。
// 目录没有正式文件：有遗留 .tmp -> ErrPartialWrite，否则 -> ErrMissing；
// 有正式文件但无一有效 -> ErrNoValidCheckpoint（携带每个文件的跳过原因）。
func (s *Store) LoadLatestValid() (st *State, path string, skipped []Skipped, err error) {
	cands, err := s.ListCandidates()
	if err != nil {
		return nil, "", nil, err
	}
	if len(cands) == 0 {
		return nil, "", nil, s.onlyTempErr()
	}
	for i := len(cands) - 1; i >= 0; i-- {
		cd := cands[i]
		state, lerr := s.LoadPath(cd.Path)
		if lerr != nil {
			skipped = append(skipped, Skipped{Path: cd.Path, Step: cd.Step, Era: cd.Era, Err: lerr})
			continue
		}
		return state, cd.Path, skipped, nil
	}
	return nil, "", skipped, fmt.Errorf("%w: every checkpoint failed validation; %s",
		ErrNoValidCheckpoint, joinSkipped(skipped))
}

// LoadStep 读取指定 step 下 era 最大的有效检查点（显式回滚用）。
// 该 step 没有任何正式文件 -> ErrCheckpointNotFound；
// 该 step 有文件但全部无效 -> ErrNoValidCheckpoint。
func (s *Store) LoadStep(step int) (*State, string, error) {
	cands, err := s.ListCandidates()
	if err != nil {
		return nil, "", err
	}
	var same []Candidate
	var skipReasons []string
	for _, cd := range cands {
		if cd.Step != step {
			continue
		}
		same = append(same, cd)
	}
	if len(same) == 0 {
		return nil, "", fmt.Errorf("%w: step=%d", ErrCheckpointNotFound, step)
	}
	// same 已按 (step,era) 升序，从 era 最大者往前尝试。
	for i := len(same) - 1; i >= 0; i-- {
		cd := same[i]
		state, lerr := s.LoadPath(cd.Path)
		if lerr != nil {
			skipReasons = append(skipReasons, fmt.Sprintf("%s: %v", filepath.Base(cd.Path), lerr))
			continue
		}
		return state, cd.Path, nil
	}
	return nil, "", fmt.Errorf("%w: no valid checkpoint at step=%d (%s)",
		ErrNoValidCheckpoint, step, strings.Join(skipReasons, "; "))
}

// Prune 保留最新 keep 个【有效】检查点，删除更旧的有效正式检查点文件。
//   - keep<=0：不限制数量，什么都不删；
//   - 无法通过校验的文件一律保留（损坏/半截必须在恢复时被明确发现，不能由清理静默销赃）；
//   - 任何情况下最新有效恢复点都不会被删除；
//   - 返回被删除文件路径，供调用方写日志。
func (s *Store) Prune(keep int) (removed []string, err error) {
	if keep <= 0 {
		return nil, nil
	}
	cands, err := s.ListCandidates()
	if err != nil {
		return nil, err
	}
	var valid []Candidate
	for _, cd := range cands {
		if _, lerr := s.LoadPath(cd.Path); lerr == nil {
			valid = append(valid, cd)
		}
	}
	if len(valid) <= keep {
		return nil, nil
	}
	stale := valid[:len(valid)-keep]
	for _, cd := range stale {
		if rerr := os.Remove(cd.Path); rerr != nil {
			return removed, rerr
		}
		removed = append(removed, cd.Path)
	}
	return removed, nil
}

// CleanupTempFiles 删除遗留的 .tmp 半截写入文件（它们永远不可能成为有效恢复点），
// 返回被删路径。
func (s *Store) CleanupTempFiles() ([]string, error) {
	tmps, err := s.TempFiles()
	if err != nil {
		return nil, err
	}
	var removed []string
	for _, p := range tmps {
		if err := os.Remove(p); err != nil {
			return removed, err
		}
		removed = append(removed, p)
	}
	return removed, nil
}

// DeleteAfter 严格删除所有 (step,era) 严格晚于给定边界的正式检查点文件，
// 用于显式回滚后清除“未来”的检查点，保证回滚点重新成为目录中的最新点，
// 下一次保存不会因为重名/排序问题被误判为旧点。
func (s *Store) DeleteAfter(step int, era uint64) ([]string, error) {
	cands, err := s.ListCandidates()
	if err != nil {
		return nil, err
	}
	var removed []string
	for _, cd := range cands {
		if lessKey(step, era, cd.Step, cd.Era) {
			if err := os.Remove(cd.Path); err != nil {
				return removed, err
			}
			removed = append(removed, cd.Path)
		}
	}
	return removed, nil
}

func joinSkipped(skipped []Skipped) string {
	parts := make([]string, len(skipped))
	for i, sk := range skipped {
		parts[i] = sk.String()
	}
	return strings.Join(parts, "; ")
}
