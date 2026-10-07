package coordinator

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/worker"
)

// serveRound 用同一套参数、同一条 socket 路径跑一轮：
// 启动协调者与 2 个真实 worker（goroutine），跑到 targetStep 后请求停机，
// 断言协调者在有限时间内自然退出并返回退出报告。
func serveRound(t *testing.T, cfg Config, round, targetStep int, expectReason ExitReason) ExitReport {
	t.Helper()
	coord, err := New(cfg) // 自动从最新检查点恢复
	if err != nil {
		t.Fatalf("round %d: New: %v", round, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	serveDone := make(chan error, 1)
	go func() { serveDone <- coord.Serve(ctx) }()
	if !waitFor(func() bool { return waitSock(cfg.SocketPath) }, 5*time.Second) {
		t.Fatalf("round %d: socket never came up", round)
	}

	wctx, wcancel := context.WithCancel(context.Background())
	defer wcancel()
	errCh := make(chan error, 2)
	for _, id := range []string{"wa", "wb"} {
		w := worker.New(worker.Config{
			ID: id, SocketPath: cfg.SocketPath,
			PollInterval:      2 * time.Millisecond,
			HeartbeatInterval: 50 * time.Millisecond,
			Logger:            testLogger{t: t, id: id},
		})
		go func() { errCh <- w.Run(wctx, true) }()
	}

	if targetStep > 0 {
		// 跑到目标步数后主动叫停。
		if !waitFor(func() bool { return coord.Step() >= targetStep }, 15*time.Second) {
			t.Fatalf("round %d: never reached step %d (stuck at %d)", round, targetStep, coord.Step())
		}
		coord.RequestStop("test-driver")
	}
	// 有限时间内必须自然退出。
	select {
	case err := <-serveDone:
		if err != nil {
			t.Fatalf("round %d: serve: %v", round, err)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("round %d: coordinator did not exit within bounded time", round)
	}
	wcancel()
	for i := 0; i < 2; i++ {
		<-errCh // worker 随 ctx 取消退出，错误不限
	}

	rep, ok := coord.ExitReport()
	if !ok {
		t.Fatalf("round %d: no exit report", round)
	}
	if rep.Reason != expectReason {
		t.Fatalf("round %d: reason=%s want %s", round, rep.Reason, expectReason)
	}
	// 本地资源已释放：socket 文件不复存在，下一轮可立即复用同一路径。
	if _, err := os.Stat(cfg.SocketPath); !os.IsNotExist(err) {
		t.Fatalf("round %d: socket file not released: %v", round, err)
	}
	drainEvents(t, coord)
	t.Logf("round %d exit report: %s", round, rep)
	return rep
}

// TestServeStopRestartRoundsAlignBaseline 连续多轮“跑一段、停、用同一 socket
// 路径重启”，最终必须跑满目标步数、如实报告 completed，且参数/优化器状态
// 与不间断单机基准对齐（不漏步、不重复累加）。
func TestServeStopRestartRoundsAlignBaseline(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 6)
	cfg.SocketPath = filepath.Join(dir, "etc.sock")
	cfg.HeartbeatTimeout = 2 * time.Second
	cfg.StepTimeout = 2 * time.Second
	cfg.CheckpointEvery = 2
	cfg.CheckpointKeep = 3

	rep1 := serveRound(t, cfg, 1, 2, ExitStopped)
	if rep1.Step < 2 {
		t.Fatalf("round 1 stopped at step %d, want >=2", rep1.Step)
	}
	rep2 := serveRound(t, cfg, 2, 4, ExitStopped)
	if rep2.Step < rep1.Step {
		t.Fatalf("round 2 step %d went backwards vs round 1 step %d (lost work)", rep2.Step, rep1.Step)
	}
	rep3 := serveRound(t, cfg, 3, 0, ExitCompleted) // targetStep=0：不拦，跑满为止
	if rep3.Step != 6 {
		t.Fatalf("final step=%d want 6", rep3.Step)
	}

	// 最终检查点里的参数/优化器状态与单机基准对齐。
	final, err := New(cfg) // 从最终检查点恢复一个只读视图
	if err != nil {
		t.Fatal(err)
	}
	assertCloseToBaseline(t, final, 1e-9)

	// 保留上限生效：检查点数量不超过 keep=3。
	entries, err := final.Store().Entries()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) > 3 {
		t.Fatalf("checkpoint count %d exceeds keep=3", len(entries))
	}
	t.Logf("kept checkpoints after 3 rounds: %+v", entries)
}
