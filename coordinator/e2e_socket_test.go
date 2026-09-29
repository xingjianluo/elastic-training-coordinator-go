package coordinator

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/worker"
)

// TestEndToEndOverUnixSocket 通过真实 unix domain socket 跑多个并发 worker，
// 验证 transport IPC、心跳、集合通信与最终单机基准对齐。
func TestEndToEndOverUnixSocket(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(filepath.Join(dir, "ckpt"), 6)
	sock := filepath.Join(dir, "etc.sock")
	cfg.SocketPath = sock
	cfg.HeartbeatTimeout = 2 * time.Second
	cfg.StepTimeout = 2 * time.Second

	coord, err := NewNoResume(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		if err := coord.Serve(ctx); err != nil {
			t.Logf("serve: %v", err)
		}
	}()
	// 等 socket 就绪
	if !waitFor(func() bool { return waitSock(sock) }, 2*time.Second) {
		t.Fatal("socket never came up")
	}

	const world = 3
	errCh := make(chan error, world)
	for i := 0; i < world; i++ {
		id := "w" + string(rune('a'+i))
		w := worker.New(worker.Config{
			ID: id, SocketPath: sock,
			PollInterval:      2 * time.Millisecond,
			HeartbeatInterval: 20 * time.Millisecond,
			Logger:            testLogger{t: t, id: id},
		})
		go func() { errCh <- w.Run(ctx, true) }()
	}
	for i := 0; i < world; i++ {
		select {
		case e := <-errCh:
			if e != nil {
				t.Fatalf("worker error: %v", e)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("workers did not finish in time")
		}
	}
	if !coord.Done() {
		t.Fatal("training not done")
	}
	assertCloseToBaseline(t, coord, 1e-9)
}

type testLogger struct {
	t  *testing.T
	id string
}

func (l testLogger) Printf(format string, args ...any) {
	l.t.Logf("[worker %s] %s", l.id, fmt.Sprintf(format, args...))
}

func waitSock(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func waitFor(fn func() bool, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if fn() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return false
}
