package coordinator

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/nn"
)

// 本文件把【真实进程】拉起来验证生命周期：
//   - SIGTERM 后协调者在有限时间内自然退出、落最终检查点、释放 socket；
//   - 立即用同一 socket/ckpt 目录重启，接着跑并对齐单机基准；
//   - 集合通信中途停机不产生半提交；
//   - 连续多轮停机重启不漏步、不重复累加；
//   - 保留上限与从较早检查点回滚。
//
// 全部走固定参数、确定性合成数据，不需要联网或外部服务。

var (
	binOnce sync.Once
	binDir  string
	binErr  error

	procsMu  sync.Mutex
	allProcs []*procHandle
)

// TestMain 在包内所有测试结束后清理进程二进制缓存目录与残留子进程。
func TestMain(m *testing.M) {
	code := m.Run()
	procsMu.Lock()
	for _, p := range allProcs {
		p.killGroup()
	}
	procsMu.Unlock()
	if binDir != "" {
		_ = os.RemoveAll(binDir)
	}
	// 给被强杀的子进程一点时间关闭管道，避免残留读取协程继续写日志。
	time.Sleep(100 * time.Millisecond)
	os.Exit(code)
}

// buildBins 在测试临时目录构建两个进程二进制（每个 go test 调用只构建一次）。
func buildBins(t *testing.T) (etcd, etcworker string) {
	t.Helper()
	binOnce.Do(func() {
		var merr error
		binDir, merr = os.MkdirTemp("", "etc-e2e-bins-")
		if merr != nil {
			binErr = merr
			return
		}
		// 测试工作目录是 coordinator/，先定位模块根目录。
		rootOut, rootErr := exec.Command("go", "env", "GOMOD").Output()
		if rootErr != nil {
			binErr = rootErr
			return
		}
		modPath := strings.TrimSpace(string(rootOut))
		root := filepath.Dir(modPath)
		for _, tc := range []struct{ name, pkg string }{
			{"etcd", "./cmd/etcd"}, {"etcworker", "./cmd/etcworker"},
		} {
			out := filepath.Join(binDir, tc.name)
			cmd := exec.Command("go", "build", "-o", out, tc.pkg)
			cmd.Dir = root
			if b, err := cmd.CombinedOutput(); err != nil {
				binErr = fmt.Errorf("build %s: %v\n%s", tc.name, err, b)
				return
			}
		}
	})
	if binErr != nil {
		t.Fatal(binErr)
	}
	return filepath.Join(binDir, "etcd"), filepath.Join(binDir, "etcworker")
}

type procHandle struct {
	cmd   *exec.Cmd
	pgid  int
	t     *testing.T
	logMu sync.Mutex
	logs  []string
	done  chan struct{}

	finished chan struct{} // 所属测试结束后关闭
}

// signalGroup 向整个独立进程组发信号。
func (p *procHandle) signalGroup(sig os.Signal) error {
	return syscall.Kill(-p.pgid, sig.(syscall.Signal))
}

// killGroup 杀掉整个独立进程组（测试清理用）。
func (p *procHandle) killGroup() { _ = syscall.Kill(-p.pgid, syscall.SIGKILL) }

// logf 在测试结束后静默丢弃（避免残留子进程写日志导致 panic）。
func (p *procHandle) logf(line string) {
	select {
	case <-p.finished:
		return
	default:
	}
	p.t.Logf("  %s", line)
}

func (p *procHandle) LogLines() []string {
	p.logMu.Lock()
	defer p.logMu.Unlock()
	return append([]string(nil), p.logs...)
}

func (p *procHandle) LogContains(sub string) bool {
	p.logMu.Lock()
	defer p.logMu.Unlock()
	for _, l := range p.logs {
		if strings.Contains(l, sub) {
			return true
		}
	}
	return false
}

// tail 启动一个 goroutine，当日志里出现 sub 或 fn 返回 true 时关闭返回的通道。
func (p *procHandle) waitLog(fn func(string) bool, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		p.logMu.Lock()
		lines := append([]string(nil), p.logs...)
		p.logMu.Unlock()
		for _, l := range lines {
			if fn(l) {
				return true
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

func startProc(t *testing.T, name string, args ...string) *procHandle {
	t.Helper()
	cmd := exec.Command(name, args...)
	// 独立进程组：向协调者发 SIGINT/SIGTERM 时不会误杀 go test 自身
	// （否则测试进程与子进程同组，终端产生的中断信号会一并送达）。
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stdout = pw
	cmd.Stderr = pw
	h := &procHandle{cmd: cmd, done: make(chan struct{}), t: t, finished: make(chan struct{})}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	h.pgid = cmd.Process.Pid
	// 所属测试结束时强杀整个进程组并停止向已结束的测试回写日志。
	t.Cleanup(func() {
		h.killGroup()
		select {
		case <-h.done:
		case <-time.After(time.Second):
		}
		close(h.finished)
	})
	procsMu.Lock()
	allProcs = append(allProcs, h)
	procsMu.Unlock()
	// 读取协程：进程退出后由等待协程关闭写端，这里读到 EOF 自然结束。
	go func() {
		s := bufio.NewScanner(pr)
		s.Buffer(make([]byte, 1<<20), 1<<20)
		for s.Scan() {
			line := s.Text()
			h.logMu.Lock()
			h.logs = append(h.logs, line)
			h.logMu.Unlock()
			h.logf(line)
		}
		_ = pr.Close()
	}()
	// 等待协程：只负责回收进程，进程一退出就广播 done 并关闭写端（触发 EOF）。
	go func() {
		_ = cmd.Wait()
		_ = pw.Close()
		close(h.done)
	}()
	return h
}

func (p *procHandle) waitExit(t *testing.T, d time.Duration) {
	t.Helper()
	select {
	case <-p.done:
	case <-time.After(d):
		p.killGroup()
		t.Fatalf("process %d did not exit within %s", p.cmd.Process.Pid, d)
	}
}

// baseArgs 是所有进程场景共用的确定性作业参数。
func baseArgs(sock, ckpt string, extra ...string) []string {
	args := []string{
		"-socket", sock, "-ckpt-dir", ckpt,
		"-steps", "6", "-batch", "8", "-nin", "2", "-samples", "32",
		"-ckpt-every", "2",
		"-heartbeat-timeout", "5s", "-step-timeout", "5s",
		"-shutdown-timeout", "3s",
	}
	return append(args, extra...)
}

func workerArgs(workerBin, sock, id string, extra ...string) []string {
	args := []string{"-socket", sock, "-id", id}
	return append(args, extra...)
}

func waitSockFile(p string, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(p); err == nil {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

// waitForGone 等待 socket 文件消失（协调者释放本地资源的证据）。
func waitForGone(p string, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

// loadLatestParams 从目录最新有效检查点读参数（进程间唯一共享状态就是磁盘检查点）。
func loadLatestParams(t *testing.T, dir string) (step int, params, vel []float64) {
	t.Helper()
	st, err := checkpoint.NewStore(dir).Load()
	if err != nil {
		t.Fatal(err)
	}
	return st.Step, st.Params, st.Velocity
}

func assertFileBaseline(t *testing.T, dir string) {
	t.Helper()
	step, params, vel := loadLatestParams(t, dir)
	if step != 6 {
		t.Fatalf("final checkpoint step=%d want 6", step)
	}
	base := nn.RunBaseline(nn.BaselineConfig{
		NSamples: 32, NIn: 2, Seed: 20260930, Steps: 6, Batch: 8, LR: 0.05, Momentum: 0.9,
	})
	for i := range params {
		if d := abs(params[i] - base.Params[i]); d > 1e-9 {
			t.Fatalf("param %d got=%.12f baseline=%.12f", i, params[i], base.Params[i])
		}
		if d := abs(vel[i] - base.Velocity[i]); d > 1e-9 {
			t.Fatalf("vel %d got=%.12f baseline=%.12f", i, vel[i], base.Velocity[i])
		}
	}
	t.Logf("PROCESS restart final state MATCH baseline at step=%d params=%.10f", step, params)
}

// TestProcessSignalStopRestartAndFinish 真实进程：跑一段 -> SIGTERM ->
// 有限时间内自然退出（130）并释放 socket -> 同参数立即重启 -> 跑满（退出码 0），
// 最终检查点对齐单机基准；日志如实区分两种结束原因。
func TestProcessSignalStopRestartAndFinish(t *testing.T) {
	etcdBin, workerBin := buildBins(t)
	dir := t.TempDir()
	sock := filepath.Join(dir, "e.sock")
	ckpt := filepath.Join(dir, "ckpt")

	// 第一轮：拉起协调者 + 2 worker，观察到 step=2 已提交后发 SIGTERM。
	coord1 := startProc(t, etcdBin, baseArgs(sock, ckpt)...)
	if !waitSockFile(sock, 5*time.Second) {
		t.Fatal("coordinator socket never appeared")
	}
	var workers []*procHandle
	for _, id := range []string{"wa", "wb"} {
		w := startProc(t, workerBin, workerArgs(workerBin, sock, id)...)
		workers = append(workers, w)
	}
	if !coord1.waitLog(func(l string) bool {
		return strings.Contains(l, "=> COMMIT; step now=2/6")
	}, 10*time.Second) {
		t.Fatal("never reached step 2 commit")
	}
	if err := coord1.signalGroup(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	coord1.waitExit(t, 5*time.Second)
	if code := coord1.cmd.ProcessState.ExitCode(); code != 130 {
		t.Fatalf("first run exit code=%d want 130 (interrupted)", code)
	}
	if !coord1.LogContains("reason=interrupted") {
		t.Fatalf("logs do not report interrupted: %v", coord1.LogLines())
	}
	if !waitForGone(sock, 2*time.Second) {
		t.Fatal("socket not released after shutdown")
	}
	step, _, _ := loadLatestParams(t, ckpt)
	t.Logf("coordinator stopped cleanly at step=%d with socket released", step)
	// 杀掉第一轮 worker（第二轮重新拉起，模拟“同一套参数、同一条本地路径”）。
	for _, w := range workers {
		w.killGroup()
		<-w.done
	}

	// 第二轮：同参数、同一 socket/ckpt 路径重启，必须接着跑而不是从头来。
	coord2 := startProc(t, etcdBin, baseArgs(sock, ckpt)...)
	if !waitSockFile(sock, 5*time.Second) {
		t.Fatal("coordinator did not rebind socket after restart")
	}
	if !coord2.waitLog(func(l string) bool {
		return strings.Contains(l, "restored from checkpoint")
	}, 3*time.Second) {
		t.Fatal("restart did not restore from checkpoint")
	}
	for _, id := range []string{"wa", "wb"} {
		startProc(t, workerBin, workerArgs(workerBin, sock, id)...)
	}
	coord2.waitExit(t, 15*time.Second)
	if code := coord2.cmd.ProcessState.ExitCode(); code != 0 {
		t.Fatalf("second run exit code=%d want 0 (completed)", code)
	}
	if !coord2.LogContains("reason=completed") {
		t.Fatalf("logs do not report completed: %v", coord2.LogLines())
	}
	assertFileBaseline(t, ckpt)
}

// TestProcessStopMidCollectiveNoHalfCommit 在 reduce 已完成、allgather 未发出
// 的窗口内 SIGTERM：最终检查点停在上一个完整步，重启后该步整体重做，最终对齐。
func TestProcessStopMidCollectiveNoHalfCommit(t *testing.T) {
	etcdBin, workerBin := buildBins(t)
	dir := t.TempDir()
	sock := filepath.Join(dir, "e.sock")
	ckpt := filepath.Join(dir, "ckpt")

	coord := startProc(t, etcdBin, baseArgs(sock, ckpt, "-ckpt-every", "0")...)
	if !waitSockFile(sock, 5*time.Second) {
		t.Fatal("socket never appeared")
	}
	// worker 在 reduce 完成后、allgather 前停 1.5s，把“通信做到一半”窗口放大。
	var workers []*procHandle
	for _, id := range []string{"wa", "wb"} {
		w := startProc(t, workerBin, workerArgs(workerBin, sock, id, "-finish-delay", "1500ms")...)
		workers = append(workers, w)
	}
	// 等到第一步已 open（两个 worker 都会进入 reduce），随即停机。
	if !coord.waitLog(func(l string) bool { return strings.Contains(l, "open step=0 ") }, 10*time.Second) {
		t.Fatal("step 0 never opened")
	}
	time.Sleep(300 * time.Millisecond) // 让 reduce-scatter 大概率完成、allgather 尚在 delay
	if err := coord.signalGroup(os.Interrupt); err != nil {
		t.Fatal(err)
	}
	coord.waitExit(t, 5*time.Second)
	if !coord.LogContains("NOT complete => discarded, no partial commit") {
		t.Logf("shutdown happened to land outside the in-flight window; logs:")
		for _, l := range coord.LogLines() {
			t.Logf("   %s", l)
		}
	}
	for _, w := range workers {
		w.killGroup()
		<-w.done
	}
	// 最终检查点必须是 step=0（第一步没整体完成，绝不能写成 step=1）。
	step, _, _ := loadLatestParams(t, ckpt)
	if step != 0 {
		t.Fatalf("final checkpoint step=%d want 0 (in-flight step must not commit)", step)
	}
	t.Logf("mid-collective stop left final checkpoint at step=%d as required", step)

	// 立即重启续训（新 worker，无 delay），最终对齐基准。
	coord2 := startProc(t, etcdBin, baseArgs(sock, ckpt, "-ckpt-every", "2")...)
	if !waitSockFile(sock, 5*time.Second) {
		t.Fatal("restart socket never appeared")
	}
	for _, id := range []string{"wa", "wb"} {
		startProc(t, workerBin, workerArgs(workerBin, sock, id)...)
	}
	coord2.waitExit(t, 15*time.Second)
	if code := coord2.cmd.ProcessState.ExitCode(); code != 0 {
		t.Fatalf("restart exit=%d want 0", code)
	}
	assertFileBaseline(t, ckpt)
}

// TestProcessMultipleStopRestartRounds 真实进程连续三轮停机/重启，不漏步不重算。
func TestProcessMultipleStopRestartRounds(t *testing.T) {
	etcdBin, workerBin := buildBins(t)
	dir := t.TempDir()
	sock := filepath.Join(dir, "e.sock")
	ckpt := filepath.Join(dir, "ckpt")

	stopAt := []string{"step now=2/6", "step now=4/6"}
	wantStep := []int{2, 4}
	for round := 0; round < 2; round++ {
		coord := startProc(t, etcdBin, baseArgs(sock, ckpt)...)
		if !waitSockFile(sock, 5*time.Second) {
			t.Fatalf("round %d: no socket", round)
		}
		var ws []*procHandle
		for _, id := range []string{"wa", "wb"} {
			w := startProc(t, workerBin, workerArgs(workerBin, sock, id, "-finish-delay", "20ms")...)
			ws = append(ws, w)
		}
		if !coord.waitLog(func(l string) bool { return strings.Contains(l, "=> COMMIT; "+stopAt[round]) }, 12*time.Second) {
			t.Fatalf("round %d: target commit not reached", round)
		}
		if err := coord.signalGroup(os.Interrupt); err != nil {
			t.Fatal(err)
		}
		coord.waitExit(t, 5*time.Second)
		for _, w := range ws {
			w.killGroup()
			<-w.done
		}
		step, _, _ := loadLatestParams(t, ckpt)
		// 截停可能恰好命中目标步或其后一两个步（绝不会提前，漏步由最终基准断言兜底）。
		if step < wantStep[round] {
			t.Fatalf("round %d ckpt step=%d want at least %d", round, step, wantStep[round])
		}
		if !waitForGone(sock, 2*time.Second) {
			t.Fatalf("round %d socket not released", round)
		}
	}
	// 第三轮跑满。
	coord := startProc(t, etcdBin, baseArgs(sock, ckpt)...)
	if !waitSockFile(sock, 5*time.Second) {
		t.Fatal("final round: no socket")
	}
	for _, id := range []string{"wa", "wb"} {
		startProc(t, workerBin, workerArgs(workerBin, sock, id)...)
	}
	coord.waitExit(t, 15*time.Second)
	if code := coord.cmd.ProcessState.ExitCode(); code != 0 {
		t.Fatalf("final round exit=%d want 0", code)
	}
	assertFileBaseline(t, ckpt)
}

// TestProcessRetentionAndRollback 真实进程：保留上限生效；随后从较早检查点
// 显式回滚重启，重跑结果对齐基准。
func TestProcessRetentionAndRollback(t *testing.T) {
	etcdBin, workerBin := buildBins(t)
	dir := t.TempDir()
	sock := filepath.Join(dir, "e.sock")
	ckpt := filepath.Join(dir, "ckpt")

	coord := startProc(t, etcdBin, baseArgs(sock, ckpt, "-max-ckpt", "2")...)
	if !waitSockFile(sock, 5*time.Second) {
		t.Fatal("no socket")
	}
	for _, id := range []string{"wa", "wb"} {
		startProc(t, workerBin, workerArgs(workerBin, sock, id)...)
	}
	coord.waitExit(t, 15*time.Second)
	cands, err := checkpoint.NewStore(ckpt).ListCandidates()
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 2 {
		t.Fatalf("retention: %d checkpoints remain, want exactly 2: %v", len(cands), cands)
	}
	// 保留的必须是最新两个（step=4,6），恢复点 step=6 绝不能被删掉。
	if cands[0].Step != 4 || cands[1].Step != 6 {
		t.Fatalf("retention kept steps %d,%d want 4,6", cands[0].Step, cands[1].Step)
	}
	t.Logf("retention kept exactly latest 2 checkpoints: %v", cands)

	// 用 -resume-step=4 回滚（max-ckpt 放大，避免重跑过程中把回滚点清掉）。
	rb := startProc(t, etcdBin, baseArgs(sock, ckpt, "-max-ckpt", "10", "-resume-step", "4")...)
	if !waitSockFile(sock, 5*time.Second) {
		t.Fatal("rollback: no socket")
	}
	if !rb.waitLog(func(l string) bool { return strings.Contains(l, "explicit rollback to step=4") }, 3*time.Second) {
		t.Fatal("rollback log missing")
	}
	for _, id := range []string{"wa", "wb"} {
		startProc(t, workerBin, workerArgs(workerBin, sock, id)...)
	}
	rb.waitExit(t, 15*time.Second)
	if code := rb.cmd.ProcessState.ExitCode(); code != 0 {
		t.Fatalf("rollback run exit=%d want 0", code)
	}
	assertFileBaseline(t, ckpt)
}
