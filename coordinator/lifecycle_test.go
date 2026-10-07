package coordinator

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
	"github.com/xingjianluo/elastic-training-coordinator-go/worker"
)

// waitTerminal 等待终结或超时。
func waitTerminal(t *testing.T, c *Coordinator, d time.Duration) ExitStatus {
	t.Helper()
	select {
	case <-c.Terminal():
	case <-time.After(d):
		t.Fatalf("coordinator did not terminate within %s", d)
	}
	return c.ExitStatus()
}

// TestGracefulShutdownMidStepNoPartialCommit 在集合通信只做到一半时叫停：
// 最终检查点只能包含上一个完整提交步的状态，在途半成品绝不写入；
// 重启续训后最终结果与单机基准一致（不漏步、不重复累加）。
func TestGracefulShutdownMidStepNoPartialCommit(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 6)

	c1, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c1.Join(proto.JoinRequest{WorkerID: "wa"})
	c1.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c1, 2) // 完整跑完 step 0,1（step=2，已有 step=2 的周期检查点）
	ctx := context.Background()

	// 打开 step=2，只让 wa 提交 reduce-scatter，wb 不提交（集合通信做到一半）。
	cores := map[string]*worker.Core{}
	asg := map[string]*proto.StepAssignment{}
	for _, id := range []string{"wa", "wb"} {
		core := newCore(c1, id)
		cores[id] = core
		nr, err := core.Next(ctx)
		if err != nil || nr.Assignment == nil {
			t.Fatalf("%s next: %v %v", id, err, nr)
		}
		asg[id] = nr.Assignment
	}
	if _, err := cores["wa"].Submit(ctx, asg["wa"]); err != nil {
		t.Fatal(err)
	}
	paramsMid := c1.Params() // 此时应仍是 step=2 提交后的参数
	stepBefore := c1.Step()
	if stepBefore != 2 {
		t.Fatalf("step=%d want 2 before shutdown", stepBefore)
	}
	drainEvents(t, c1)

	// 任意时刻叫停。
	c1.Shutdown(ExitInterrupted, "test stop mid collective", time.Second)
	st := waitTerminal(t, c1, 2*time.Second)
	if st.Reason != ExitInterrupted {
		t.Fatalf("reason=%v want interrupted", st.Reason)
	}
	if st.Step != 2 || st.FinalCheckpoint == "" {
		t.Fatalf("status=%+v", st)
	}
	// 重启：从最终检查点恢复，在途步必须重来而不是被当成已完成。
	c2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if c2.Step() != 2 {
		t.Fatalf("resumed step=%d want 2 (in-flight half must not commit)", c2.Step())
	}
	if d := paramsDiff(c2.Params(), paramsMid); d > 0 {
		t.Fatalf("resumed params differ from last committed: %g", d)
	}
	c2.Join(proto.JoinRequest{WorkerID: "wa"})
	c2.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c2, 4) // step 2,3,4,5 全部重做一遍
	if !c2.Done() || c2.Step() != 6 {
		t.Fatalf("after resume step=%d done=%v", c2.Step(), c2.Done())
	}
	assertCloseToBaseline(t, c2, 1e-9)
	t.Logf("mid-collective shutdown: final ckpt at step=%d then restarted and matched baseline", st.Step)
}

// TestShutdownRejectsNewWorkAndIsIdempotent 停机开始后新工作被明确拒绝，
// 重复停机安全，且可在有限时间内终结。
func TestShutdownRejectsNewWorkAndIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 6)
	c, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c.Join(proto.JoinRequest{WorkerID: "wa"})
	c.Join(proto.JoinRequest{WorkerID: "wb"})
	// 打开一步但不完成，制造在途状态。
	ctx := context.Background()
	c0 := newCore(c, "wa")
	nr, _ := c0.Next(ctx)
	if nr.Assignment == nil {
		t.Fatal("expected open assignment")
	}
	c.Shutdown(ExitInterrupted, "stop", time.Second)
	c.Shutdown(ExitInterrupted, "stop again", time.Second) // 幂等
	st := waitTerminal(t, c, 2*time.Second)
	if st.Reason != ExitInterrupted {
		t.Fatalf("reason=%v", st.Reason)
	}
	// 在途步作废：重启后仍是 step=0。
	c2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if c2.Step() != 0 {
		t.Fatalf("step after abort=%d want 0", c2.Step())
	}
	// 停机后再提交贡献必须被明确拒绝。
	_, err = c.Submit(proto.SubmitRequest{WorkerID: "wa", Era: nr.Assignment.Era, Step: 0,
		Range: nr.Assignment.ParamRange, Chunk: make([]float64, 3), VectorLen: 3})
	if !errors.Is(err, ErrShuttingDown) {
		t.Fatalf("submit after stop: %v want ErrShuttingDown", err)
	}
	t.Logf("new work after shutdown rejected with shutting_down; status=%+v", st)
}

// TestCompletionReasonIsCompleted 不经过外部叫停、跑满目标步数时原因必须是 completed。
func TestCompletionReasonIsCompleted(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 4)
	sock := filepath.Join(dir, "e.sock")
	cfg.SocketPath = sock
	c, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	serveErr := make(chan error, 1)
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { serveErr <- c.Serve(parent) }()
	if !waitFor(func() bool { return waitSock(sock) }, 2*time.Second) {
		t.Fatal("socket never came up")
	}
	for _, id := range []string{"wa", "wb"} {
		w := worker.New(worker.Config{
			ID: id, SocketPath: sock, PollInterval: 2 * time.Millisecond,
			HeartbeatInterval: 20 * time.Millisecond, Logger: testLogger{t: t, id: id},
		})
		go func() { _ = w.Run(parent, true) }()
	}
	st := waitTerminal(t, c, 10*time.Second)
	if st.Reason != ExitCompleted {
		t.Fatalf("reason=%v want completed", st.Reason)
	}
	if st.Step != 4 || st.FinalCheckpoint == "" {
		t.Fatalf("status=%+v", st)
	}
	if err := <-serveErr; err != nil {
		t.Fatalf("serve: %v", err)
	}
	// socket 文件应被释放。
	if _, err := os.Stat(sock); !os.IsNotExist(err) {
		t.Fatalf("socket not released: %v", err)
	}
	t.Logf("completed naturally: %+v", st)
}

// TestMultipleStopRestartRoundsMatchBaseline 连续多轮“跑一段-停-重启”，
// 不漏步、不重复累加，最终对齐单机基准。
func TestMultipleStopRestartRoundsMatchBaseline(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 6)
	c := func() *Coordinator {
		cc, err := New(cfg) // 第一轮目录为空 => 全新；其余轮从最新点恢复
		if err != nil {
			t.Fatal(err)
		}
		return cc
	}()
	total := 0
	rounds := 0
	for total < cfg.Steps {
		c.Join(proto.JoinRequest{WorkerID: "wa"})
		c.Join(proto.JoinRequest{WorkerID: "wb"})
		run := 1
		if rounds == 1 {
			run = 2
		}
		if total+run > cfg.Steps {
			run = cfg.Steps - total
		}
		driveSteps(t, c, run)
		total += run
		rounds++
		if total < cfg.Steps {
			c.Shutdown(ExitInterrupted, fmt.Sprintf("round %d stop", rounds), time.Second)
			st := waitTerminal(t, c, 2*time.Second)
			if st.Step != total {
				t.Fatalf("round %d checkpoint step=%d want %d", rounds, st.Step, total)
			}
			c = (func() *Coordinator {
				cc, err := New(cfg)
				if err != nil {
					t.Fatal(err)
				}
				return cc
			})()
		}
	}
	if !c.Done() {
		t.Fatalf("not done after %d rounds, step=%d", rounds, c.Step())
	}
	assertCloseToBaseline(t, c, 1e-9)
	t.Logf("survived %d stop/restart rounds and matched baseline", rounds)
}

// TestResumeFromOlderCheckpointAndRetention 显式回滚到较早检查点重跑，
// 结果对齐基准；保留上限在运行中持续生效，最新点不被删。
func TestResumeFromOlderCheckpointAndRetention(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 6)
	cfg.MaxCheckpoints = 2
	c1, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c1.Join(proto.JoinRequest{WorkerID: "wa"})
	c1.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c1, 4) // step=4，期间 step=2,4 各落一个点（上限 2）
	cands, _ := c1.Store().ListCandidates()
	if len(cands) != 2 {
		t.Fatalf("retention: kept %d checkpoints want 2: %v", len(cands), cands)
	}
	t.Logf("retention kept %v", cands)
	if c1.Step() != 4 {
		t.Fatalf("step=%d", c1.Step())
	}

	// 回滚到 step=2（较早的有效点），重跑 2,3,4,5。
	rollbackCfg := cfg
	rs := 2
	rollbackCfg.ResumeStep = &rs
	c2, err := New(rollbackCfg)
	if err != nil {
		t.Fatalf("rollback New: %v", err)
	}
	if c2.Step() != 2 {
		t.Fatalf("rolled back to step=%d want 2", c2.Step())
	}
	// “未来”的 step=4 检查点必须已被清除，回滚点成为最新点。
	latest, err := c2.Store().Latest()
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(latest) != "ckpt-s0000002-e0000002.etc" {
		t.Fatalf("latest after rollback=%s", latest)
	}
	c2.Join(proto.JoinRequest{WorkerID: "wa"})
	c2.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c2, 4)
	if !c2.Done() || c2.Step() != 6 {
		t.Fatalf("after rollback step=%d done=%v", c2.Step(), c2.Done())
	}
	assertCloseToBaseline(t, c2, 1e-9)
}

// TestResumeMissingStepRejected 显式回滚到不存在的 step 要明确拒绝。
func TestResumeMissingStepRejected(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 6)
	c1, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c1.Join(proto.JoinRequest{WorkerID: "wa"})
	c1.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c1, 2)
	rollbackCfg := cfg
	rs := 5
	rollbackCfg.ResumeStep = &rs
	if _, err := New(rollbackCfg); !errors.Is(err, checkpoint.ErrCheckpointNotFound) {
		t.Fatalf("rollback missing step: got %v want ErrCheckpointNotFound", err)
	}
}

// TestAutoRecoverFallsBackToOlderValid 最新点损坏时 AutoRecover 自动采用更早
// 的可用点并记录跳过原因，绝不清零重来。
func TestAutoRecoverFallsBackToOlderValid(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 6)
	c1, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c1.Join(proto.JoinRequest{WorkerID: "wa"})
	c1.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c1, 4)
	wantStep4 := c1.Params()
	// 损坏最新点（step=4）。
	p, err := c1.Store().Latest()
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(p)
	raw[20] ^= 0xFF
	if err := os.WriteFile(p, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	// 严格模式：必须拒绝。
	if _, err := New(cfg); !errors.Is(err, checkpoint.ErrCorrupt) {
		t.Fatalf("strict mode: %v want ErrCorrupt", err)
	}
	// 自动恢复模式：回落到 step=2 的有效点继续，不静默清零。
	autoCfg := cfg
	autoCfg.AutoRecover = true
	c2, err := New(autoCfg)
	if err != nil {
		t.Fatalf("auto recover: %v", err)
	}
	if c2.Step() != 2 {
		t.Fatalf("auto recover step=%d want 2", c2.Step())
	}
	if d := paramsDiff(c2.Params(), wantStep4); d == 0 {
		t.Fatalf("auto recover unexpectedly returned step-4 params (fallback did not happen)")
	}
	t.Logf("auto-recover fell back to step=%d instead of silently restarting", c2.Step())
	c2.Join(proto.JoinRequest{WorkerID: "wa"})
	c2.Join(proto.JoinRequest{WorkerID: "wb"})
	driveSteps(t, c2, 4)
	assertCloseToBaseline(t, c2, 1e-9)
}
