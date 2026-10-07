package coordinator

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/nn"
)

// buildCmd 把 cmd/<name> 构建成真实可执行文件（测试真实进程生命周期用）。
func buildCmd(t *testing.T, dir, name string) string {
	t.Helper()
	bin := filepath.Join(dir, name)
	out, err := exec.Command("go", "build", "-o", bin, "../cmd/"+name).CombinedOutput()
	if err != nil {
		t.Fatalf("build %s: %v\n%s", name, err, out)
	}
	return bin
}

func startProc(t *testing.T, name string, logPath string, args ...string) *exec.Cmd {
	t.Helper()
	f, err := os.Create(logPath)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(name, args...)
	cmd.Stdout = f
	cmd.Stderr = f
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		f.Close()
	})
	return cmd
}

// waitExit 断言进程在 d 内自然退出（不是被 kill），返回是否如此。
func waitExit(cmd *exec.Cmd, d time.Duration) (exited bool, code int) {
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err == nil {
			return true, 0
		}
		if ee, ok := err.(*exec.ExitError); ok {
			return true, ee.ExitCode()
		}
		return true, -1
	case <-time.After(d):
		return false, -1
	}
}

func readLog(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestRealProcessGracefulStopAndResume 把真实协调者/worker 进程拉起来：
// 主动 SIGTERM 停机 -> 有限时间内自然退出、如实报告 stopped、释放 socket ->
// 用同一套参数同一 socket 路径立刻重启 -> worker 自动重连跑满 ->
// 如实报告 completed，最终检查点与单机基准对齐，检查点数量不超保留上限。
func TestRealProcessGracefulStopAndResume(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real-process test in -short mode")
	}
	dir := t.TempDir()
	etcd := buildCmd(t, dir, "etcd")
	etcworker := buildCmd(t, dir, "etcworker")

	sock := filepath.Join(dir, "e.sock")
	ckptDir := filepath.Join(dir, "ckpt")
	etcdArgs := []string{
		"-socket", sock, "-ckpt-dir", ckptDir,
		"-steps", "6", "-batch", "8", "-nin", "2", "-samples", "32",
		"-ckpt-every", "2", "-ckpt-keep", "3",
		"-heartbeat-timeout", "2s", "-step-timeout", "2s",
	}

	// 第一轮：拉起协调者 + 2 个真实 worker 进程。
	etcdLog1 := filepath.Join(dir, "etcd.1.log")
	p1 := startProc(t, etcd, etcdLog1, etcdArgs...)
	if !waitFor(func() bool { return waitSock(sock) }, 10*time.Second) {
		t.Fatal("socket never came up")
	}
	wa := startProc(t, etcworker, filepath.Join(dir, "wa.log"), "-socket", sock, "-id", "wa")
	wb := startProc(t, etcworker, filepath.Join(dir, "wb.log"), "-socket", sock, "-id", "wb")

	// 等到至少落了一份 step>=2 的检查点（说明确实跑起来了）。
	store := checkpoint.NewStore(ckptDir)
	ok := waitFor(func() bool {
		entries, err := store.Entries()
		if err != nil || len(entries) == 0 {
			return false
		}
		return entries[len(entries)-1].Step >= 2
	}, 30*time.Second)
	if !ok {
		t.Fatalf("no checkpoint with step>=2 appeared; etcd log:\n%s", readLog(t, etcdLog1))
	}

	// 主动停机：SIGTERM。必须在有限时间内自然退出（退出码 0）。
	if err := p1.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	exited, code := waitExit(p1, 10*time.Second)
	if !exited {
		t.Fatalf("coordinator did not exit within bounded time after SIGTERM; etcd log:\n%s\nwa log:\n%s",
			readLog(t, etcdLog1), readLog(t, filepath.Join(dir, "wa.log")))
	}
	if code != 0 {
		t.Fatalf("coordinator exit code=%d, want 0 (clean exit)", code)
	}
	log1 := readLog(t, etcdLog1)
	if !strings.Contains(log1, "reason=stopped") {
		t.Fatalf("first run must truthfully report stopped; log:\n%s", log1)
	}
	// 本地资源已释放：同一 socket 路径可立即复用。
	if _, err := os.Stat(sock); !os.IsNotExist(err) {
		t.Fatalf("socket file not released after exit: %v", err)
	}
	t.Logf("first run exited cleanly with reason=stopped")

	// 第二轮：同一套参数、同一条 socket 路径立刻重启；worker 自动重连接着跑。
	etcdLog2 := filepath.Join(dir, "etcd.2.log")
	p2 := startProc(t, etcd, etcdLog2, etcdArgs...)
	exited, code = waitExit(p2, 60*time.Second)
	if !exited {
		t.Fatalf("restarted coordinator did not finish; log:\n%s", readLog(t, etcdLog2))
	}
	if code != 0 {
		t.Fatalf("restarted coordinator exit code=%d; log:\n%s", code, readLog(t, etcdLog2))
	}
	log2 := readLog(t, etcdLog2)
	if !strings.Contains(log2, "restored from checkpoint") {
		t.Fatalf("second run must resume from checkpoint; log:\n%s", log2)
	}
	if !strings.Contains(log2, "reason=completed") {
		t.Fatalf("second run must report completed; log:\n%s", log2)
	}
	t.Logf("second run resumed and completed")

	// worker 看到 DONE 后应自行退出。
	for _, w := range []*exec.Cmd{wa, wb} {
		if exited, _ := waitExit(w, 10*time.Second); !exited {
			t.Logf("worker still running after completion (killing): pid=%d", w.Process.Pid)
			_ = w.Process.Kill()
		}
	}

	// 最终检查点与不间断单机基准对齐（不漏步、不重复累加）。
	st, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if st.Step != 6 {
		t.Fatalf("final checkpoint step=%d want 6", st.Step)
	}
	base := nn.RunBaseline(nn.BaselineConfig{
		NSamples: 32, NIn: 2, Seed: 20260930, Steps: 6, Batch: 8, LR: 0.05, Momentum: 0.9,
	})
	if d := paramsDiff(st.Params, base.Params); d > 1e-9 {
		t.Fatalf("final params diverge from baseline: diff=%g", d)
	}
	if d := paramsDiff(st.Velocity, base.Velocity); d > 1e-9 {
		t.Fatalf("final velocity diverges from baseline: diff=%g", d)
	}
	t.Logf("final checkpoint matches single-machine baseline (params=%v)", st.Params)

	// 保留上限：两轮下来目录里不超过 keep=3 份，且最新有效点在。
	entries, err := store.Entries()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 || len(entries) > 3 {
		t.Fatalf("checkpoint count=%d, want 1..3 (keep=3)", len(entries))
	}
	if entries[len(entries)-1].Step != 6 {
		t.Fatalf("latest kept checkpoint step=%d want 6", entries[len(entries)-1].Step)
	}
	t.Logf("retention keep=3 holds: %d checkpoint(s) kept", len(entries))
}
