// Command etcd (elastic-training-coordinator-daemon) 启动本地弹性训练协调者。
//
// 生命周期语义：
//   - 跑满 -steps：在最终状态落检查点、释放 socket 后以退出码 0 干净结束；
//   - 收到 SIGINT/SIGTERM 或本地 stop RPC：在【最后一个完整完成的步】上落最终
//     检查点（在途半成品作废，绝不半提交），在 -shutdown-timeout 内干净退出，
//     退出码 130（被中途叫停）；
//   - 随后用同一 -socket/-ckpt-dir 与相同参数重新拉起，会自动接着跑，结果与不
//     中断一次跑完位级一致。
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/coordinator"
)

// exitInterrupted 是“被中途叫停但已干净结束”时的进程退出码。
const exitInterrupted = 130

func main() {
	socket := flag.String("socket", "/tmp/etc.sock", "unix socket 路径")
	ckptDir := flag.String("ckpt-dir", "./checkpoints", "检查点目录")
	steps := flag.Int("steps", 6, "训练步数")
	batch := flag.Int("batch", 8, "每步样本数")
	nIn := flag.Int("nin", 2, "输入维度")
	nSamples := flag.Int("samples", 32, "数据集样本数")
	lr := flag.Float64("lr", 0.05, "学习率")
	momentum := flag.Float64("momentum", 0.9, "momentum")
	hbTimeout := flag.Duration("heartbeat-timeout", 2*time.Second, "心跳超时（判定死亡）")
	stepTimeout := flag.Duration("step-timeout", 5*time.Second, "单步等待超时（区分掉队/死亡）")
	ckptEvery := flag.Int("ckpt-every", 2, "每多少步落一次检查点（0=关闭，停机/完成时总会落一个）")
	noResume := flag.Bool("no-resume", false, "忽略已有检查点全新启动")
	maxCkpt := flag.Int("max-ckpt", 3, "检查点目录保留的有效检查点上限（<=0 不限制）")
	resumeStep := flag.Int("resume-step", -1, "显式从该 step 的较早检查点回滚续训（<0=取最新点）")
	autoRecover := flag.Bool("auto-recover", false, "最新点损坏/半截时自动采用更早的可用点（默认严格拒绝）")
	shutdownTimeout := flag.Duration("shutdown-timeout", 5*time.Second, "优雅停机最长时间")
	completionGrace := flag.Duration("completion-grace", 200*time.Millisecond, "跑满后等待落后 worker 收到 Done 的宽限（<0 不等）")
	flag.Parse()

	cfg := coordinator.Config{
		SocketPath:       *socket,
		CheckpointDir:    *ckptDir,
		NSamples:         *nSamples,
		NIn:              *nIn,
		Seed:             20260930,
		Steps:            *steps,
		Batch:            *batch,
		LR:               *lr,
		Momentum:         *momentum,
		HeartbeatTimeout: *hbTimeout,
		StepTimeout:      *stepTimeout,
		CheckpointEvery:  *ckptEvery,
		MaxCheckpoints:   *maxCkpt,
		AutoRecover:      *autoRecover,
		ShutdownTimeout:  *shutdownTimeout,
		CompletionGrace:  *completionGrace,
	}
	if *resumeStep >= 0 {
		rs := *resumeStep
		cfg.ResumeStep = &rs
	}
	var (
		coord *coordinator.Coordinator
		err   error
	)
	if *noResume {
		coord, err = coordinator.NewNoResume(cfg)
	} else {
		coord, err = coordinator.New(cfg)
	}
	if err != nil {
		log.Fatalf("coordinator init failed: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Printf("received signal %s: graceful shutdown (final checkpoint, no new work)", sig)
		coord.Shutdown(coordinator.ExitInterrupted, "signal "+sig.String(), *shutdownTimeout)
	}()
	// 把协调者的判定依据（成员列表/分片区间/每步决策/失败原因）打到日志。
	go func() {
		for ev := range coord.Events() {
			log.Printf("[%s] %s", ev.Kind, ev.Detail)
		}
	}()
	log.Printf("coordinator serving on %s (steps=%d; workers join dynamically; max-ckpt=%d)",
		*socket, steps, *maxCkpt)
	serveErr := coord.Serve(ctx)
	cancel()
	status := coord.ExitStatus()
	log.Printf("coordinator exited: reason=%s step=%d/%d era=%d finalCheckpoint=%s detail=%q",
		status.Reason, status.Step, status.TargetSteps, status.Era,
		status.FinalCheckpoint, status.Detail)
	if serveErr != nil {
		log.Fatalf("serve failed: %v", serveErr)
	}
	if status.Reason == coordinator.ExitInterrupted {
		// 如实区分“被中途叫停”与“跑满”，便于编排脚本判断与立即重启。
		os.Exit(exitInterrupted)
	}
}
