package checkpoint

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func sampleState(step int) *State {
	return &State{
		Version:  FormatVersion,
		Era:      1,
		Step:     step,
		Params:   []float64{0.1, 0.2, 0.3},
		Velocity: []float64{0.01, 0.02, 0.03},
		Members:  []Member{{ID: "a", Rank: 0}, {ID: "b", Rank: 1}},
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	st := sampleState(3)
	if err := s.Save(st); err != nil {
		t.Fatal(err)
	}
	got, err := s.Load()
	if err != nil {
		t.Fatal(err)
	}
	if got.Step != 3 || got.Era != 1 || len(got.Members) != 2 {
		t.Fatalf("loaded state wrong: %+v", got)
	}
	for i := range st.Params {
		if got.Params[i] != st.Params[i] || got.Velocity[i] != st.Velocity[i] {
			t.Fatalf("param %d mismatch", i)
		}
	}
	t.Logf("round-trip checkpoint step=%d members=%v", got.Step, got.Members)
}

func TestLatestPicksHighestStep(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	for _, step := range []int{2, 8, 4} {
		if err := s.Save(sampleState(step)); err != nil {
			t.Fatal(err)
		}
	}
	st, err := s.Load()
	if err != nil {
		t.Fatal(err)
	}
	if st.Step != 8 {
		t.Fatalf("latest step=%d want 8", st.Step)
	}
}

func TestMissing(t *testing.T) {
	s := NewStore(t.TempDir())
	_, err := s.Load()
	if !errors.Is(err, ErrMissing) || KindOf(err) != KindMissing {
		t.Fatalf("got %v want ErrMissing", err)
	}
}

func TestPartialTempOnly(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	p, err := s.SavePartial(sampleState(1), 10) // 只写 10 字节 JSON
	if err != nil {
		t.Fatal(err)
	}
	// 直接读 .tmp 必须报 PartialWrite
	if _, err := s.LoadPath(p); !errors.Is(err, ErrPartialWrite) {
		t.Fatalf("LoadPath tmp: %v", err)
	}
	// Latest 在只有半截临时文件时也必须拒绝
	_, err = s.Latest()
	if !errors.Is(err, ErrPartialWrite) {
		t.Fatalf("Latest: %v want ErrPartialWrite", err)
	}
	if KindOf(err) != KindPartialWrite {
		t.Fatalf("kind=%v", KindOf(err))
	}
	t.Logf("partial temp file correctly rejected: %s", filepath.Base(p))
}

func TestTruncatedFinalIsPartialWrite(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	// 把一个 .tmp 半截改名成正式文件：头里声明了完整长度但载荷/校验和缺失。
	p, _ := s.SavePartial(sampleState(2), 20)
	final := filepath.Join(dir, "ckpt-s0000002-e0000001.etc")
	if err := os.Rename(p, final); err != nil {
		t.Fatal(err)
	}
	_, err := s.Load()
	if !errors.Is(err, ErrPartialWrite) {
		t.Fatalf("truncated final: %v want ErrPartialWrite", err)
	}
}

func TestBadMagic(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	// 写入长度 >=8 但魔数错误的文件。
	p, err := s.writeBadFile("ckpt-s0000001-e0000001.etc", []byte("NOTACKPT some random garbage data here padding"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.LoadPath(p); !errors.Is(err, ErrBadMagic) || KindOf(err) != KindBadMagic {
		t.Fatalf("got %v want ErrBadMagic", err)
	}
}

func TestVersionMismatch(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	st := sampleState(1)
	st.Version = 999
	p, err := s.writeBadFile("ckpt-s0000001-e0000001.etc", buildRaw(st))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.LoadPath(p); !errors.Is(err, ErrVersionMismatch) || KindOf(err) != KindVersionMismatch {
		t.Fatalf("got %v want ErrVersionMismatch", err)
	}
}

func TestCorruptChecksum(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	st := sampleState(1)
	raw := buildRaw(st)
	// 翻转载荷中间一个字节，保持长度不变 -> 校验和不符 -> Corrupt。
	raw[20] ^= 0xFF
	p, err := s.writeBadFile("ckpt-s0000001-e0000001.etc", raw)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.LoadPath(p); !errors.Is(err, ErrCorrupt) || KindOf(err) != KindCorrupt {
		t.Fatalf("got %v want ErrCorrupt", err)
	}
}

func TestValidCheckpointPreferredOverTemp(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)
	if err := s.Save(sampleState(5)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SavePartial(sampleState(99), 5); err != nil { // 遗留半截
		t.Fatal(err)
	}
	st, err := s.Load()
	if err != nil {
		t.Fatal(err)
	}
	if st.Step != 5 {
		t.Fatalf("got step %d, want valid ckpt step 5", st.Step)
	}
	tmps, _ := s.TempFiles()
	if len(tmps) != 1 {
		t.Fatalf("expected leftover temp discoverable, got %v", tmps)
	}
	t.Logf("leftover partial files kept visible: %v", tmps)
}
