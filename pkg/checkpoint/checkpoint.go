package checkpoint

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const (
	Magic          = "ETCK"
	CurrentVersion = uint16(1)
)

var (
	ErrNotFound           = errors.New("checkpoint: checkpoint not found")
	ErrInvalidCheckpoint  = errors.New("checkpoint: invalid checkpoint container")
	ErrUnsupportedVersion = errors.New("checkpoint: unsupported checkpoint version")
	ErrPartial            = errors.New("checkpoint: checkpoint is only partially written")
	ErrCorrupt            = errors.New("checkpoint: checkpoint payload is corrupt")
)

// State is the complete information required to resume a deterministic run.
// The ordered membership and per-step completion bookkeeping are included in
// addition to model tensors, because resuming only weights would allow a
// reconnecting worker to contribute twice.
type State struct {
	Version          int       `json:"version"`
	Generation       int       `json:"generation"`
	NextStep         int       `json:"next_step"`
	TotalSteps       int       `json:"total_steps"`
	Params           []float64 `json:"params"`
	Momentum         []float64 `json:"momentum"`
	Target           []float64 `json:"target"`
	Members          []string  `json:"members"`
	CompletedWorkers []string  `json:"completed_workers"`
	RunCompleted     bool      `json:"run_completed"`
}

type Checkpointer struct {
	path string
}

func New(path string) *Checkpointer { return &Checkpointer{path: path} }

func (c *Checkpointer) Path() string { return c.path }

// Save writes to a unique temporary file and only renames after flush and
// fsync. A process killed while writing therefore leaves a *.tmp file behind
// but never replaces the previous committed checkpoint.
func (c *Checkpointer) Save(state State) error {
	if state.Version == 0 {
		state.Version = int(CurrentVersion)
	}
	if state.Version != int(CurrentVersion) {
		return fmt.Errorf("%w: writer version=%d, supported=%d",
			ErrUnsupportedVersion, state.Version, CurrentVersion)
	}
	if err := validateState(state); err != nil {
		return err
	}

	var compressed bytes.Buffer
	zw := gzip.NewWriter(&compressed)
	if _, err := zw.Write(mustMarshal(state)); err != nil {
		return fmt.Errorf("checkpoint: compress state: %w", err)
	}
	if err := zw.Close(); err != nil {
		return fmt.Errorf("checkpoint: finalize compression: %w", err)
	}
	payload := compressed.Bytes()
	checksum := sha256.Sum256(payload)

	dir := filepath.Dir(c.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("checkpoint: create directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".checkpoint-*.tmp")
	if err != nil {
		return fmt.Errorf("checkpoint: create temporary file: %w", err)
	}
	tmpName := tmp.Name()
	removeOnError := true
	defer func() {
		if removeOnError {
			_ = os.Remove(tmpName)
		}
	}()

	header := make([]byte, 0, 18)
	header = append(header, Magic...)
	header = binary.BigEndian.AppendUint16(header, CurrentVersion)
	header = binary.BigEndian.AppendUint64(header, uint64(len(payload)))
	if _, err := tmp.Write(header); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("checkpoint: write header: %w", err)
	}
	if _, err := tmp.Write(payload); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("checkpoint: write payload: %w", err)
	}
	if _, err := tmp.Write(checksum[:]); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("checkpoint: write checksum: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("checkpoint: fsync checkpoint: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("checkpoint: close checkpoint: %w", err)
	}
	if err := os.Rename(tmpName, c.path); err != nil {
		return fmt.Errorf("checkpoint: commit checkpoint: %w", err)
	}
	removeOnError = false
	return nil
}

// Load enforces framing before interpreting payload contents. A short frame
// that could result from an interrupted direct write is ErrPartial; a complete
// frame with a bad hash or unreadable JSON is ErrCorrupt.
func (c *Checkpointer) Load() (State, error) {
	data, err := os.ReadFile(c.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return State{}, fmt.Errorf("%w: %s", ErrNotFound, c.path)
		}
		return State{}, fmt.Errorf("checkpoint: read file: %w", err)
	}
	const headerSize = 4 + 2 + 8
	const checksumSize = sha256.Size
	if len(data) < headerSize {
		return State{}, fmt.Errorf("%w: file has %d bytes, header requires %d", ErrPartial, len(data), headerSize)
	}
	if string(data[:4]) != Magic {
		return State{}, fmt.Errorf("%w: bad magic %q", ErrInvalidCheckpoint, string(data[:4]))
	}
	version := binary.BigEndian.Uint16(data[4:6])
	if version != CurrentVersion {
		return State{}, fmt.Errorf("%w: checkpoint version=%d, supported=%d",
			ErrUnsupportedVersion, version, CurrentVersion)
	}
	payloadLen := int(binary.BigEndian.Uint64(data[6:headerSize]))
	need := headerSize + payloadLen + checksumSize
	if len(data) < need {
		return State{}, fmt.Errorf("%w: file has %d of %d advertised bytes", ErrPartial, len(data), need)
	}
	if len(data) > need {
		return State{}, fmt.Errorf("%w: file has %d unexpected trailing bytes", ErrCorrupt, len(data)-need)
	}
	payload := data[headerSize : headerSize+payloadLen]
	gotChecksum := data[headerSize+payloadLen:]
	wantChecksum := sha256.Sum256(payload)
	if !bytes.Equal(gotChecksum, wantChecksum[:]) {
		return State{}, fmt.Errorf("%w: checksum mismatch", ErrCorrupt)
	}

	zr, err := gzip.NewReader(bytes.NewReader(payload))
	if err != nil {
		return State{}, fmt.Errorf("%w: gzip header: %v", ErrCorrupt, err)
	}
	jsonPayload, err := io.ReadAll(zr)
	if err != nil {
		return State{}, fmt.Errorf("%w: gzip payload: %v", ErrCorrupt, err)
	}
	var state State
	if err := json.Unmarshal(jsonPayload, &state); err != nil {
		return State{}, fmt.Errorf("%w: state json: %v", ErrCorrupt, err)
	}
	if state.Version != int(CurrentVersion) {
		return State{}, fmt.Errorf("%w: embedded state version=%d", ErrUnsupportedVersion, state.Version)
	}
	if err := validateState(state); err != nil {
		return State{}, err
	}
	return state, nil
}

func mustMarshal(state State) []byte {
	data, err := json.Marshal(&state)
	if err != nil {
		panic(fmt.Sprintf("checkpoint: state must be JSON-encodable: %v", err))
	}
	return data
}

func validateState(state State) error {
	if state.Version != int(CurrentVersion) {
		return fmt.Errorf("%w: state version=%d", ErrUnsupportedVersion, state.Version)
	}
	if state.NextStep < 0 || state.TotalSteps < 0 || state.NextStep > state.TotalSteps {
		return fmt.Errorf("%w: step range next=%d total=%d", ErrCorrupt, state.NextStep, state.TotalSteps)
	}
	if len(state.Params) != len(state.Momentum) || len(state.Params) != len(state.Target) {
		return fmt.Errorf("%w: tensor lengths params=%d momentum=%d target=%d",
			ErrCorrupt, len(state.Params), len(state.Momentum), len(state.Target))
	}
	if len(state.Members) > 0 && len(state.Params) == 0 {
		return fmt.Errorf("%w: members exist without parameters", ErrCorrupt)
	}
	return nil
}
