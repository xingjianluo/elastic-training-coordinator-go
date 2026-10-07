// Command etcworker 启动一个本地训练 worker。
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"strconv"
	"syscall"

	"github.com/xingjianluo/elastic-training-coordinator-go/worker"
)

func main() {
	socket := flag.String("socket", "/tmp/etc.sock", "unix socket 路径")
	id := flag.String("id", "", "worker id（默认 worker-<pid>）")
	noReconnect := flag.Bool("no-reconnect", false, "连接中断后不自动重连")
	finishDelay := flag.Duration("finish-delay", 0, "reduce 完成、allgather 之前的确定性停顿（停机演练用）")
	flag.Parse()

	wid := *id
	if wid == "" {
		wid = "worker-" + strconv.Itoa(os.Getpid())
	}
	w := worker.New(worker.Config{ID: wid, SocketPath: *socket, FinishDelay: *finishDelay})
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		cancel()
	}()
	if err := w.Run(ctx, !*noReconnect); err != nil {
		log.Fatalf("worker failed: %v", err)
	}
}
