package coordinator

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
)

// driveSteps 让固定 2 个成员跑完 n 步。
func driveSteps(t *testing.T, c *Coordinator, n int) {
	t.Helper()
	ctx := context.Background()
	for i := 0; i < n; i++ {
		runRound(t, ctx, c, []string{"wa", "wb"})
	}
}

// TestCheckpointResumeContinuesAndMatches 跑到一半（检查点已落盘），模拟进程
// 重启：用同一检查点目录 New 一个新协调者，成员重新 join 后接着跑完剩余步数，
// 最终参数/优化器速度必须与一次性单机基准位级一致。
func TestCheckpointResumeContinuesAndMatches(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir, 6)

	c1, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c1.Join(proto.JoinRequest{WorkerID: "wa"})
	c1.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c1, 3) // step=3；step=2 时已落检查点
	drainEvents(t, c1)
	if c1.Step() != 3 {
		t.Fatalf("step=%d", c1.Step())
	}

	// 显式再落一个 step=3 的检查点，确保重启从 step=3 恢复而不是 2。
	if _, err := c1.RequestCheckpoint("wa"); err != nil {
		t.Fatal(err)
	}

	// 模拟重启：New 会自动加载最新检查点。
	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("resume New failed: %v", err)
	}
	if c2.Step() != 3 {
		t.Fatalf("resumed step=%d want 3", c2.Step())
	}
	t.Logf("resumed at step=%d era=%d params=%v velocity=%v",
		c2.Step(), c2.Era(), c2.Params(), c2.velCopy())
	// 恢复后成员重新登记（相同 ID），成员信息也从检查点带回。
	c2.Join(proto.JoinRequest{WorkerID: "wa"})
	c2.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c2, 3) // 接着跑完 step 3,4,5
	if !c2.Done() || c2.Step() != 6 {
		t.Fatalf("after resume step=%d done=%v", c2.Step(), c2.Done())
	}
	assertCloseToBaseline(t, c2, 1e-9)
}

// TestRestartRejectsPartialCheckpoint 目录里只有写到一半的检查点时，
// 重启必须明确失败（ErrPartialWrite），不能静默全新启动。
func TestRestartRejectsPartialCheckpoint(t *testing.T) {
	dir := t.TempDir()
	st := &checkpoint.State{
		Version: checkpoint.FormatVersion, Era: 1, Step: 2,
		Params: []float64{1, 2, 3}, Velocity: []float64{0.1, 0.2, 0.3},
		Members: []checkpoint.Member{{ID: "wa", Rank: 0}},
	}
	store := checkpoint.NewStore(dir)
	tmp, err := store.SavePartial(st, 12)
	if err != nil {
		t.Fatal(err)
	}
	// 模拟崩溃发生在 rename 之后、内容写全之前：半截内容使用正式文件名。
	final := filepath.Join(dir, "ckpt-s0000002-e0000001.etc")
	if err := os.Rename(tmp, final); err != nil {
		t.Fatal(err)
	}
	_, err = New(testConfig(dir, 6))
	if !errors.Is(err, checkpoint.ErrPartialWrite) {
		t.Fatalf("restart on partial ckpt: got %v want ErrPartialWrite", err)
	}
	// 同样的半截内容若还停留在 .tmp（rename 前崩溃），也必须拒绝。
	tmp2, err := store.SavePartial(st, 12)
	if err != nil {
		t.Fatal(err)
	}
	_ = os.Remove(final)
	_, err = store.LoadPath(tmp2)
	if !errors.Is(err, checkpoint.ErrPartialWrite) {
		t.Fatalf("leftover .tmp: got %v want ErrPartialWrite", err)
	}
	if checkpoint.KindOf(err) != checkpoint.KindPartialWrite {
		t.Fatalf("kind=%v", checkpoint.KindOf(err))
	}
	t.Logf("restart correctly REFUSED partial checkpoint: %v", err)
}

// TestRestartRejectsCorruptAndBadVersion 损坏校验和、版本不符、魔数错误都要被
// 明确拒绝且原因可区分。
func TestRestartRejectsCorruptAndBadVersion(t *testing.T) {
	dir := t.TempDir()
	store := checkpoint.NewStore(dir)
	good := &checkpoint.State{
		Version: checkpoint.FormatVersion, Era: 1, Step: 2,
		Params: []float64{1, 2, 3}, Velocity: []float64{0.1, 0.2, 0.3},
		Members: []checkpoint.Member{{ID: "wa", Rank: 0}},
	}
	if err := store.Save(good); err != nil {
		t.Fatal(err)
	}
	p, err := store.Latest()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	// 1) 翻转载荷中间一个字节（保持长度）=> 校验和不符 => Corrupt
	corrupt := append([]byte(nil), raw...)
	corrupt[20] ^= 0xFF
	cp := filepath.Join(dir, "ckpt-s0000009-e0000001.etc")
	if err := os.WriteFile(cp, corrupt, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(testConfig(dir, 6)); !errors.Is(err, checkpoint.ErrCorrupt) {
		t.Fatalf("corrupt: got %v want ErrCorrupt", err)
	}
	t.Logf("corrupt checkpoint refused: %v", err)
	_ = os.Remove(cp)

	// 2) 坏魔数
	badMagicPath := filepath.Join(dir, "ckpt-s0000010-e0000001.etc")
	if err := os.WriteFile(badMagicPath, []byte("NOTACKPT padding padding padding padding"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(testConfig(dir, 6)); !errors.Is(err, checkpoint.ErrBadMagic) {
		t.Fatalf("bad magic: got %v want ErrBadMagic", err)
	}
	t.Logf("bad-magic checkpoint refused: %v", err)
	_ = os.Remove(badMagicPath)

	// 3) 版本不符：保存一份 step=11 的有效检查点，再就地把头部版本字段改成 42。
	verState := &checkpoint.State{
		Version: checkpoint.FormatVersion, Era: 1, Step: 11,
		Params: []float64{1, 2, 3}, Velocity: []float64{0.1, 0.2, 0.3},
	}
	vstore := checkpoint.NewStore(dir)
	if err := vstore.Save(verState); err != nil {
		t.Fatal(err)
	}
	verPath, err := vstore.Latest()
	if err != nil {
		t.Fatal(err)
	}
	vraw, err := os.ReadFile(verPath)
	if err != nil {
		t.Fatal(err)
	}
	binary.BigEndian.PutUint32(vraw[8:12], 42) // 仅改头部版本字段
	if err := os.WriteFile(verPath, vraw, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(testConfig(dir, 6)); !errors.Is(err, checkpoint.ErrVersionMismatch) {
		t.Fatalf("version: got %v want ErrVersionMismatch", err)
	}
	t.Logf("wrong-version checkpoint refused: %v", err)
}

// TestRestoreRejectsDimMismatch 检查点参数维度与当前作业不符也必须拒绝。
func TestRestoreRejectsDimMismatch(t *testing.T) {
	dir := t.TempDir()
	st := &checkpoint.State{
		Version: checkpoint.FormatVersion, Era: 1, Step: 1,
		Params: []float64{1, 2}, Velocity: []float64{0.1, 0.2}, // 维度 2，作业要 3
		Members: []checkpoint.Member{{ID: "wa", Rank: 0}},
	}
	if err := checkpoint.NewStore(dir).Save(st); err != nil {
		t.Fatal(err)
	}
	if _, err := New(testConfig(dir, 6)); !errors.Is(err, checkpoint.ErrCorrupt) {
		t.Fatalf("dim mismatch: got %v want ErrCorrupt", err)
	}
}
