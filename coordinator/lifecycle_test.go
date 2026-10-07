package coordinator

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
)

// TestGracefulStopMidCollectiveNoPartialCommit 在集合通信做到一半时停机：
// reduce-scatter 只收到 1/2 贡献，这一步绝不能被当成已完成写进检查点。
// 收尾检查点必须停在最后一个【完整提交】的步上，重启后从那里继续并对齐基准。
func TestGracefulStopMidCollectiveNoPartialCommit(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir, 6)

	c1, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	c1.Join(proto.JoinRequest{WorkerID: "wa"})
	c1.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c1, 2) // 完整跑完 2 步
	paramsAt2 := c1.Params()

	// 第 3 步开工：wa 拿到分工并提交 reduce-scatter，wb 还没交。
	ca := newCore(c1, "wa")
	na, err := ca.Next(ctx)
	if err != nil || na.Assignment == nil {
		t.Fatalf("wa next: %v", err)
	}
	if err := ca.DoSubmitOnly(ctx, na.Assignment); err != nil {
		t.Fatalf("wa submit: %v", err)
	}
	if c1.txn == nil || len(c1.txn.submitted) != 1 {
		t.Fatalf("expected in-flight step with 1/2 submits, got %+v", c1.txn)
	}

	// 集合通信中途停机。
	c1.RequestStop("test")
	rep := c1.Shutdown()
	if rep.Reason != ExitStopped {
		t.Fatalf("reason=%s want stopped", rep.Reason)
	}
	if rep.Step != 2 {
		t.Fatalf("report step=%d want 2 (in-flight step must not count)", rep.Step)
	}
	// 停机后不再接收新工作。
	if err := ca.DoSubmitOnly(ctx, na.Assignment); !errors.Is(err, ErrShuttingDown) {
		t.Fatalf("submit after stop: got %v want ErrShuttingDown", err)
	}
	nr, _ := newCore(c1, "wa").Next(ctx)
	if !nr.Stopped {
		t.Fatalf("next after stop should report Stopped")
	}
	if jr := c1.Join(proto.JoinRequest{WorkerID: "wc"}); jr.OK {
		t.Fatalf("new join after stop should be rejected")
	}

	// 落盘的最终检查点必须是 step=2 的状态，参数与停机前逐位一致。
	st, err := checkpoint.NewStore(dir).Load()
	if err != nil {
		t.Fatal(err)
	}
	if st.Step != 2 {
		t.Fatalf("final checkpoint step=%d want 2 (no partial commit)", st.Step)
	}
	if d := paramsDiff(st.Params, paramsAt2); d != 0 {
		t.Fatalf("checkpoint params diverged from pre-stop state: diff=%g", d)
	}
	drainEvents(t, c1)

	// 用同一检查点目录重启：从 step=2 继续，wa 半途的那份贡献不算数，
	// 整步重做，最终结果对齐单机基准（不丢步、不重复累加）。
	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("restart: %v", err)
	}
	if c2.Step() != 2 {
		t.Fatalf("resumed step=%d want 2", c2.Step())
	}
	c2.Join(proto.JoinRequest{WorkerID: "wa"})
	c2.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c2, 4)
	if !c2.Done() {
		t.Fatalf("not done after resume: step=%d", c2.Step())
	}
	assertCloseToBaseline(t, c2, 1e-9)
	drainEvents(t, c2)
}

// TestExitReportCompleted 跑满目标步数后，退出报告必须如实写明 completed。
func TestExitReportCompleted(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir, 4)
	c, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c.Join(proto.JoinRequest{WorkerID: "wa"})
	c.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c, 4)
	rep := c.Shutdown()
	if rep.Reason != ExitCompleted || rep.Step != 4 || rep.Steps != 4 {
		t.Fatalf("report=%+v want completed 4/4", rep)
	}
	if rep.CheckpointPath == "" {
		t.Fatalf("final checkpoint path empty")
	}
	// 幂等：再次 Shutdown 返回同一份报告。
	if rep2 := c.Shutdown(); rep2 != rep {
		t.Fatalf("shutdown not idempotent: %+v vs %+v", rep2, rep)
	}
	t.Logf("exit report: %s", rep)
}

// TestCheckpointRetentionAndResume 保留上限生效：目录里只留最新 N 份，
// 最新有效点始终可用于恢复，重启不会像全新作业。
func TestCheckpointRetentionAndResume(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir, 6)
	cfg.CheckpointEvery = 1
	cfg.CheckpointKeep = 2

	c, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c.Join(proto.JoinRequest{WorkerID: "wa"})
	c.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c, 6)
	rep := c.Shutdown()
	drainEvents(t, c)

	entries, err := checkpoint.NewStore(dir).Entries()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) > 2 {
		t.Fatalf("checkpoints grew beyond keep=2: %d files", len(entries))
	}
	if len(entries) == 0 || entries[len(entries)-1].Step != 6 {
		t.Fatalf("latest kept checkpoint = %+v, want step 6 present", entries)
	}
	t.Logf("retention keep=2: remaining entries=%+v", entries)

	// 从保留的点恢复：done 状态被带回，不是全新作业。
	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("resume after prune: %v", err)
	}
	if c2.Step() != 6 || !c2.Done() {
		t.Fatalf("after resume step=%d done=%v, want 6/true", c2.Step(), c2.Done())
	}
	assertCloseToBaseline(t, c2, 1e-9)
	_ = rep
}

// TestRollbackToEarlierCheckpoint 显式挑一个更早的有效检查点回滚，
// 从那里重新往下跑，结果同样对齐基准。
func TestRollbackToEarlierCheckpoint(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir, 6)
	cfg.CheckpointEvery = 2 // 检查点落在 step 2/4/6

	c1, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c1.Join(proto.JoinRequest{WorkerID: "wa"})
	c1.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c1, 6)
	drainEvents(t, c1)

	entries, err := checkpoint.NewStore(dir).Entries()
	if err != nil || len(entries) < 3 {
		t.Fatalf("entries=%+v err=%v", entries, err)
	}
	rollbackTo := entries[0].Path // step=2 的点
	t.Logf("rolling back to %s (step=%d)", rollbackTo, entries[0].Step)

	cfg2 := cfg
	cfg2.ResumeFrom = rollbackTo
	c2, err := New(cfg2)
	if err != nil {
		t.Fatalf("rollback resume: %v", err)
	}
	if c2.Step() != 2 {
		t.Fatalf("rolled back to step=%d want 2", c2.Step())
	}
	c2.Join(proto.JoinRequest{WorkerID: "wa"})
	c2.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c2, 4)
	if !c2.Done() {
		t.Fatalf("not done after rollback: step=%d", c2.Step())
	}
	assertCloseToBaseline(t, c2, 1e-9)
	drainEvents(t, c2)
}

// TestResumeFromUnusableRefused 显式指定的回滚点缺失/损坏时，
// 必须以可区分原因明确拒绝，绝不静默从头开始。
func TestResumeFromUnusableRefused(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir, 6)

	cfg.ResumeFrom = dir + "/ckpt-s0000003-e0000001.etc" // 不存在（可能已被清理）
	if _, err := New(cfg); !errors.Is(err, checkpoint.ErrMissing) {
		t.Fatalf("missing resume-from: got %v want ErrMissing", err)
	}

	// 写一个有效检查点再翻转一个字节 => 损坏。
	st := &checkpoint.State{
		Version: checkpoint.FormatVersion, Era: 1, Step: 3,
		Params: []float64{1, 2, 3}, Velocity: []float64{0.1, 0.2, 0.3},
	}
	store := checkpoint.NewStore(dir)
	if err := store.Save(st); err != nil {
		t.Fatal(err)
	}
	p, _ := store.Latest()
	raw, _ := readFile(t, p)
	raw[20] ^= 0xFF
	writeFile(t, p, raw)
	cfg.ResumeFrom = p
	if _, err := New(cfg); !errors.Is(err, checkpoint.ErrCorrupt) {
		t.Fatalf("corrupt resume-from: got %v want ErrCorrupt", err)
	}
	t.Logf("unusable resume-from correctly refused (missing -> ErrMissing, corrupt -> ErrCorrupt)")
}

// TestResumeFallbackToUsable 最新检查点损坏时，显式开启回退后
// 必须采用仍然可用的更早点并在日志里说明；不开回退则严格拒绝。
func TestResumeFallbackToUsable(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir, 6)
	cfg.CheckpointEvery = 2

	c1, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c1.Join(proto.JoinRequest{WorkerID: "wa"})
	c1.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c1, 4) // 检查点落在 step 2 和 4
	drainEvents(t, c1)

	// 损坏最新的（step=4）。
	latest, err := checkpoint.NewStore(dir).Latest()
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := readFile(t, latest)
	raw[20] ^= 0xFF
	writeFile(t, latest, raw)

	// 默认严格：拒绝。
	if _, err := New(cfg); !errors.Is(err, checkpoint.ErrCorrupt) {
		t.Fatalf("strict mode: got %v want ErrCorrupt", err)
	}
	// 显式回退：采用 step=2 的点继续，最终对齐基准。
	cfgF := cfg
	cfgF.ResumeFallback = true
	c2, err := New(cfgF)
	if err != nil {
		t.Fatalf("fallback resume: %v", err)
	}
	if c2.Step() != 2 {
		t.Fatalf("fallback resumed step=%d want 2", c2.Step())
	}
	foundFallbackLog := false
	for {
		select {
		case ev := <-c2.Events():
			t.Logf("  [%s] %s", ev.Kind, ev.Detail)
			if strings.Contains(ev.Detail, "fallback") {
				foundFallbackLog = true
			}
		default:
			goto drained
		}
	}
drained:
	if !foundFallbackLog {
		t.Fatalf("expected fallback decision in event log")
	}
	c2.Join(proto.JoinRequest{WorkerID: "wa"})
	c2.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c2, 4)
	assertCloseToBaseline(t, c2, 1e-9)
}

func readFile(t *testing.T, p string) ([]byte, error) {
	t.Helper()
	return os.ReadFile(p)
}

func writeFile(t *testing.T, p string, b []byte) {
	t.Helper()
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
}
