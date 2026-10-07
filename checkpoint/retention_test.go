package checkpoint

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func mkState(step int, era uint64) *State {
	return &State{
		Version: FormatVersion, Era: era, Step: step,
		Params: []float64{float64(step), 2, 3}, Velocity: []float64{0.1, 0.2, 0.3},
		Members: []Member{{ID: "wa", Rank: 0}},
	}
}

func saveN(t *testing.T, s *Store, steps ...int) {
	t.Helper()
	for _, st := range steps {
		if err := s.Save(mkState(st, 1)); err != nil {
			t.Fatal(err)
		}
	}
}

func TestEntriesSortedAndPruneKeepsNewest(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	saveN(t, s, 2, 4, 6, 8)

	entries, err := s.Entries()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 4 {
		t.Fatalf("entries=%d want 4", len(entries))
	}
	for i, want := range []int{2, 4, 6, 8} {
		if entries[i].Step != want {
			t.Fatalf("entries[%d].Step=%d want %d", i, entries[i].Step, want)
		}
	}

	removed, err := s.Prune(2)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 2 {
		t.Fatalf("removed=%d want 2 (%v)", len(removed), removed)
	}
	entries, _ = s.Entries()
	if len(entries) != 2 || entries[0].Step != 6 || entries[1].Step != 8 {
		t.Fatalf("after prune entries=%+v, want steps [6 8]", entries)
	}
	// 被清理掉的点再显式加载必须报“缺失”，可区分于损坏。
	_, err = s.LoadPath(removed[0])
	if !errors.Is(err, ErrMissing) {
		t.Fatalf("pruned file load: got %v want ErrMissing", err)
	}
	t.Logf("pruned %v; kept steps [6 8]", removed)
}

func TestPruneNeverDeletesLatestEvenWithKeepBelowOne(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	saveN(t, s, 2, 4)
	// keep=0 会被钳制为 1：最新有效点是下次恢复要用的，绝不删除，
	// 清理后重启不能看起来像全新作业。
	removed, err := s.Prune(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 1 {
		t.Fatalf("removed=%v want exactly 1", removed)
	}
	entries, _ := s.Entries()
	if len(entries) != 1 || entries[0].Step != 4 {
		t.Fatalf("entries=%+v want only step 4", entries)
	}
	st, err := s.Load()
	if err != nil || st.Step != 4 {
		t.Fatalf("load after prune: st=%v err=%v", st, err)
	}
}

func TestLoadUsableFallsBackToEarlierValid(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	saveN(t, s, 2, 4)
	// 把最新的（step=4）翻一个字节 => 校验和损坏。
	latest, err := s.Latest()
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(latest)
	raw[20] ^= 0xFF
	if err := os.WriteFile(latest, raw, 0o600); err != nil {
		t.Fatal(err)
	}

	st, path, skipped, err := s.LoadUsable()
	if err != nil {
		t.Fatalf("LoadUsable: %v", err)
	}
	if st.Step != 2 {
		t.Fatalf("fell back to step=%d want 2", st.Step)
	}
	if len(skipped) != 1 {
		t.Fatalf("skipped=%v want 1 entry", skipped)
	}
	if KindOf(skipped[0]) != KindCorrupt {
		t.Fatalf("skipped reason=%v want corrupt", skipped[0])
	}
	t.Logf("fell back to %s; skipped reasons: %v", filepath.Base(path), skipped)
}

func TestLoadUsableAllBadReturnsNewestReason(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	saveN(t, s, 2, 4)
	entries, _ := s.Entries()
	for _, e := range entries {
		raw, _ := os.ReadFile(e.Path)
		raw[20] ^= 0xFF
		_ = os.WriteFile(e.Path, raw, 0o600)
	}
	_, _, skipped, err := s.LoadUsable()
	if !errors.Is(err, ErrCorrupt) {
		t.Fatalf("got %v want ErrCorrupt", err)
	}
	if len(skipped) != 2 {
		t.Fatalf("skipped=%d want 2", len(skipped))
	}
}

func TestLoadUsableEmptyAndOnlyTemp(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	if _, _, _, err := s.LoadUsable(); !errors.Is(err, ErrMissing) {
		t.Fatalf("empty dir: got %v want ErrMissing", err)
	}
	if _, err := s.SavePartial(mkState(3, 1), 8); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := s.LoadUsable(); !errors.Is(err, ErrPartialWrite) {
		t.Fatalf("only .tmp: got %v want ErrPartialWrite", err)
	}
}
