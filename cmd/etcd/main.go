// Command etcd (elastic-training-coordinator-daemon) 启动本地弹性训练协调者。
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
	ckptEvery := flag.Int("ckpt-every", 2, "每多少步落一次检查点（0=关闭）")
	noResume := flag.Bool("no-resume", false, "忽略已有检查点全新启动")
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
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		cancel()
	}()
	// 把协调者的判定依据（成员列表/分片区间/每步决策/失败原因）打到日志。
	go func() {
		for ev := range coord.Events() {
			log.Printf("[%s] %s", ev.Kind, ev.Detail)
		}
	}()
	log.Printf("coordinator serving on %s (steps=%d; workers join dynamically)", *socket, steps)
	if err := coord.Serve(ctx); err != nil {
		log.Fatalf("serve failed: %v", err)
	}
	log.Printf("coordinator exited cleanly")
}
