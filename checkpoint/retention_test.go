package checkpoint

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

func mkState(step int, era uint64) *State {
	return &State{
		Version: FormatVersion, Era: era, Step: step,
		Params:   []float64{1, 2, 3},
		Velocity: []float64{0.1, 0.2, 0.3},
		Members:  []Member{{ID: "wa", Rank: 0}},
	}
}

func saveN(t *testing.T, dir string, pairs ...[2]int) *Store {
	t.Helper()
	s := NewStore(dir)
	for _, p := range pairs {
		if err := s.Save(mkState(p[0], uint64(p[1]))); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func names(cands []Candidate) []string {
	out := make([]string, len(cands))
	for i, c := range cands {
		out[i] = filepath.Base(c.Path)
	}
	sort.Strings(out)
	return out
}

// TestListCandidatesOrdered 列出的候选按 (step,era) 升序，且忽略 .tmp/杂项文件。
func TestListCandidatesOrdered(t *testing.T) {
	dir := t.TempDir()
	s := saveN(t, dir, [2]int{4, 1}, [2]int{2, 1}, [2]int{2, 2}, [2]int{6, 1})
	// 一个遗留 .tmp 和一个不认识的文件都不应出现在候选里。
	if err := os.WriteFile(filepath.Join(dir, "ckpt-s0000001-e0000001.etc.tmp"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	cands, err := s.ListCandidates()
	if err != nil {
		t.Fatal(err)
	}
	got := names(cands)
	want := []string{
		"ckpt-s0000002-e0000001.etc",
		"ckpt-s0000002-e0000002.etc",
		"ckpt-s0000004-e0000001.etc",
		"ckpt-s0000006-e0000001.etc",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("candidates=%v want=%v", got, want)
	}
	t.Logf("candidates in (step,era) order: %v", got)
}

// TestPruneKeepsLatestN 保留上限：只删更旧有效点，最新点永删不掉。
func TestPruneKeepsLatestN(t *testing.T) {
	dir := t.TempDir()
	s := saveN(t, dir, [2]int{1, 1}, [2]int{2, 1}, [2]int{3, 1}, [2]int{4, 1}, [2]int{5, 1})

	removed, err := s.Prune(3)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 2 {
		t.Fatalf("removed=%d want 2 (%v)", len(removed), removed)
	}
	cands, _ := s.ListCandidates()
	got := names(cands)
	want := []string{
		"ckpt-s0000003-e0000001.etc",
		"ckpt-s0000004-e0000001.etc",
		"ckpt-s0000005-e0000001.etc",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("after prune=%v want=%v", got, want)
	}
	t.Logf("prune kept latest 3: %v removed=%v", got, removed)

	// 最新有效点必须还能直接恢复。
	st, err := s.Load()
	if err != nil {
		t.Fatalf("latest after prune: %v", err)
	}
	if st.Step != 5 {
		t.Fatalf("latest step=%d want 5", st.Step)
	}

	// 数量不超过上限时什么都不删。
	r2, err := s.Prune(3)
	if err != nil || len(r2) != 0 {
		t.Fatalf("second prune: removed=%v err=%v", r2, err)
	}
	// keep<=0 表示不限制。
	r3, err := s.Prune(0)
	if err != nil || len(r3) != 0 {
		t.Fatalf("keep<=0 prune: %v %v", r3, err)
	}
}

// TestPruneNeverDeletesLatestRecoveryPoint 即使 keep=1，最新点也必须保留。
func TestPruneNeverDeletesLatestRecoveryPoint(t *testing.T) {
	dir := t.TempDir()
	s := saveN(t, dir, [2]int{1, 1}, [2]int{2, 1})
	removed, err := s.Prune(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 1 {
		t.Fatalf("removed=%v", removed)
	}
	if st, err := s.Load(); err != nil || st.Step != 2 {
		t.Fatalf("latest must survive prune: st=%v err=%v", st, err)
	}
}

// TestLoadStepSelectsEraMax 回滚到指定 step 时取该 step 下 era 最大者。
func TestLoadStepSelectsEraMax(t *testing.T) {
	dir := t.TempDir()
	s := saveN(t, dir, [2]int{2, 1}, [2]int{2, 2}, [2]int{4, 1})
	st, path, err := s.LoadStep(2)
	if err != nil {
		t.Fatal(err)
	}
	if st.Era != 2 {
		t.Fatalf("era=%d want 2", st.Era)
	}
	t.Logf("LoadStep(2) picked %s", filepath.Base(path))

	_, _, err3 := s.LoadStep(3)
	if !errors.Is(err3, ErrCheckpointNotFound) {
		t.Fatalf("missing step: got %v want ErrCheckpointNotFound", err3)
	}
	if checkpointKind := KindOf(err3); checkpointKind != KindNotFound {
		t.Fatalf("kind=%v want KindNotFound", checkpointKind)
	}
}

// TestLoadLatestValidSkipsBadNewerPoints 最新点损坏时自动采用更早的可用点，
// 跳过原因被如实记录；全坏则 ErrNoValidCheckpoint。
func TestLoadLatestValidSkipsBadNewerPoints(t *testing.T) {
	dir := t.TempDir()
	s := saveN(t, dir, [2]int{1, 1}, [2]int{2, 1}, [2]int{3, 1})
	// 把最新点（step=3）就地损坏：翻载荷中间一个字节保持长度。
	p, _ := s.Latest()
	raw, _ := os.ReadFile(p)
	raw[20] ^= 0xFF
	if err := os.WriteFile(p, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	st, path, skipped, err := s.LoadLatestValid()
	if err != nil {
		t.Fatalf("LoadLatestValid: %v", err)
	}
	if st.Step != 2 {
		t.Fatalf("picked step=%d want 2", st.Step)
	}
	if len(skipped) != 1 || !errors.Is(skipped[0].Err, ErrCorrupt) {
		t.Fatalf("skipped=%v want one corrupt entry", skipped)
	}
	t.Logf("LoadLatestValid fell back to %s, skipped: %v", filepath.Base(path), skipped)

	// 全部正式点都损坏 -> ErrNoValidCheckpoint。
	cands, _ := s.ListCandidates()
	for _, c := range cands {
		b, _ := os.ReadFile(c.Path)
		for i := 16; i < len(b); i++ {
			b[i] ^= 0xFF
		}
		_ = os.WriteFile(c.Path, b, 0o600)
	}
	_, _, _, err = s.LoadLatestValid()
	if !errors.Is(err, ErrNoValidCheckpoint) {
		t.Fatalf("all corrupt: got %v want ErrNoValidCheckpoint", err)
	}
	t.Logf("all-bad directory correctly rejected: %v", err)
}

// TestLoadLatestValidEmptyDirReasons 空目录与仅有 .tmp 的原因可区分。
func TestLoadLatestValidEmptyDirReasons(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	if _, _, _, err := s.LoadLatestValid(); !errors.Is(err, ErrMissing) {
		t.Fatalf("empty dir: %v", err)
	}
	if _, err := s.SavePartial(mkState(1, 1), 10); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := s.LoadLatestValid(); !errors.Is(err, ErrPartialWrite) {
		t.Fatalf("only tmp: %v", err)
	}
}

// TestDeleteAfterClearsFuture 回滚后清除严格晚于边界的检查点。
func TestDeleteAfterClearsFuture(t *testing.T) {
	dir := t.TempDir()
	s := saveN(t, dir, [2]int{1, 1}, [2]int{2, 1}, [2]int{2, 2}, [2]int{3, 1})
	removed, err := s.DeleteAfter(2, 1) // 删除 (step,era) > (2,1)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 2 {
		t.Fatalf("removed=%v want 2", removed)
	}
	cands, _ := s.ListCandidates()
	got := names(cands)
	want := []string{"ckpt-s0000001-e0000001.etc", "ckpt-s0000002-e0000001.etc"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("remaining=%v want=%v", got, want)
	}
	t.Logf("DeleteAfter removed=%v remaining=%v", removed, got)
}

// TestCleanupTempFiles 删除遗留 .tmp。
func TestCleanupTempFiles(t *testing.T) {
	dir := t.TempDir()
	s := saveN(t, dir, [2]int{1, 1})
	if _, err := s.SavePartial(mkState(2, 1), 5); err != nil {
		t.Fatal(err)
	}
	tmps, err := s.CleanupTempFiles()
	if err != nil || len(tmps) != 1 {
		t.Fatalf("cleanup tmps=%v err=%v", tmps, err)
	}
	again, err := s.CleanupTempFiles()
	if err != nil || len(again) != 0 {
		t.Fatalf("second cleanup: %v %v", again, err)
	}
}
