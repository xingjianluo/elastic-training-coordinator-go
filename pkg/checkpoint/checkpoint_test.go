package checkpoint

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func testState() State {
	return State{
		Version:          int(CurrentVersion),
		Generation:       3,
		NextStep:         2,
		TotalSteps:       5,
		Params:           []float64{0.5, -1.25},
		Momentum:         []float64{0.25, -0.75},
		Target:           []float64{1, 2},
		Members:          []string{"w1", "w2"},
		CompletedWorkers: []string{"w1", "w2"},
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ckpt")
	cp := New(path)
	want := testState()
	if err := cp.Save(want); err != nil {
		t.Fatal(err)
	}
	got, err := cp.Load()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("members=%v next_step=%d decision=magic-version-payload-checksum-valid", got.Members, got.NextStep)
	if got.Generation != want.Generation || got.NextStep != want.NextStep || len(got.Params) != len(want.Params) {
		t.Fatalf("round trip mismatch: %+v", got)
	}
}

func TestRejectsPartialCorruptVersionAndInvalidContainer(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ckpt")
	cp := New(path)
	if err := cp.Save(testState()); err != nil {
		t.Fatal(err)
	}
	good, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	partialPath := filepath.Join(t.TempDir(), "partial")
	if err := os.WriteFile(partialPath, good[:len(good)-8], 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = New(partialPath).Load()
	t.Logf("decision=trailing-bytes-missing err=%v expected=partial", err)
	if !errors.Is(err, ErrPartial) {
		t.Fatalf("partial got %v", err)
	}

	corruptPath := filepath.Join(t.TempDir(), "corrupt")
	corrupt := append([]byte(nil), good...)
	corrupt[len(corrupt)-1] ^= 0xff
	if err := os.WriteFile(corruptPath, corrupt, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = New(corruptPath).Load()
	t.Logf("decision=checksum-mismatch err=%v expected=corrupt", err)
	if !errors.Is(err, ErrCorrupt) {
		t.Fatalf("corrupt got %v", err)
	}

	versionPath := filepath.Join(t.TempDir(), "version")
	versioned := append([]byte(nil), good...)
	binary.BigEndian.PutUint16(versioned[4:6], 99)
	if err := os.WriteFile(versionPath, versioned, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = New(versionPath).Load()
	t.Logf("decision=unsupported-version err=%v", err)
	if !errors.Is(err, ErrUnsupportedVersion) {
		t.Fatalf("version got %v", err)
	}

	invalidPath := filepath.Join(t.TempDir(), "invalid")
	if err := os.WriteFile(invalidPath, []byte("not-a-checkpoint"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = New(invalidPath).Load()
	t.Logf("decision=bad-magic err=%v", err)
	if !errors.Is(err, ErrInvalidCheckpoint) {
		t.Fatalf("invalid container got %v", err)
	}

	_, err = New(filepath.Join(t.TempDir(), "missing")).Load()
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing got %v", err)
	}
}

func TestSaveRejectsBadState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad")
	state := testState()
	state.Params = []float64{1}
	err := New(path).Save(state)
	t.Logf("decision=tensor-length-validation err=%v", err)
	if !errors.Is(err, ErrCorrupt) {
		t.Fatalf("bad state got %v", err)
	}
}
