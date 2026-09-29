package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/cluster"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/sharding"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/training"
)

const (
	dimension   = 12
	totalSteps  = 4
	workerCount = 3
)

type shardRange = sharding.Range

func main() {
	if len(os.Args) > 1 && os.Args[1] == "worker" {
		if err := runWorker(os.Args[2:]); err != nil {
			fail(err)
		}
		return
	}
	if err := runCoordinator(); err != nil {
		fail(err)
	}
}

func runCoordinator() error {
	dir, err := os.MkdirTemp("", "elastic-training-demo-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	socketPath := filepath.Join(dir, "coordinator.sock")
	checkpointPath := filepath.Join(dir, "training.ckpt")

	cfg := cluster.Config{
		SocketPath:       socketPath,
		Dimension:        dimension,
		TotalSteps:       totalSteps,
		InitialWorkers:   workerCount,
		StepTimeout:      5 * time.Second,
		HeartbeatTimeout: 30 * time.Second,
		CheckpointPath:   checkpointPath,
		CheckpointEvery:  1,
	}
	coordinator := cluster.NewCoordinator(cfg)
	listener, err := cluster.ListenUnix(socketPath)
	if err != nil {
		return err
	}
	serveErr := make(chan error, 1)
	go func() { serveErr <- coordinator.Serve(listener) }()

	executable, err := os.Executable()
	if err != nil {
		return err
	}
	children := make([]*exec.Cmd, 0, workerCount)
	for i := range workerCount {
		id := fmt.Sprintf("worker-%d", i)
		cmd := exec.Command(executable, "worker",
			"--socket", socketPath,
			"--id", id,
			"--heartbeat", "20ms",
		)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			return err
		}
		children = append(children, cmd)
	}
	for _, cmd := range children {
		if err := cmd.Wait(); err != nil {
			return fmt.Errorf("worker process failed: %w", err)
		}
	}

	// Run the idempotency check in another real local process as well.
	reconnect := exec.Command(executable, "worker",
		"--socket", socketPath,
		"--id", "worker-0",
		"--heartbeat", "20ms",
	)
	reconnect.Stdout = os.Stdout
	reconnect.Stderr = os.Stderr
	if err := reconnect.Run(); err != nil {
		return fmt.Errorf("completed reconnect process failed: %w", err)
	}

	got, step, gen, plan := coordinator.Snapshot()
	want := training.Baseline(dimension, totalSteps, 0.05, 0.9)
	maxDiff := 0.0
	for i := range got {
		maxDiff = math.Max(maxDiff, math.Abs(got[i]-want[i]))
	}
	fmt.Printf("socket=%s\ncheckpoint=%s\n", socketPath, checkpointPath)
	fmt.Printf("process_workers=%d\nmembers=%v\nranges=%v\n", workerCount, plan.Workers, plan.Ranges)
	fmt.Printf("generation=%d completed_steps=%d max_abs_diff=%.3e\n", gen, step, maxDiff)
	if maxDiff > 1e-12 {
		return errors.New("distributed result does not match the deterministic single-node baseline")
	}

	if err := coordinator.Close(); err != nil {
		return err
	}
	if err := listener.Close(); err != nil {
		return err
	}
	return <-serveErr
}

func runWorker(args []string) error {
	fs := flag.NewFlagSet("worker", flag.ContinueOnError)
	socketPath := fs.String("socket", "", "coordinator Unix socket path")
	id := fs.String("id", "", "stable worker ID")
	heartbeat := fs.Duration("heartbeat", 20*time.Millisecond, "heartbeat interval")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *socketPath == "" || *id == "" {
		return errors.New("worker requires --socket and --id")
	}
	gradient := func(_ string, _, _ int, owned shardRange, params []float64) []float64 {
		target := training.NewModel(dimension, 0.05, 0.9).Target
		return training.Gradient(params, target[owned.Start:owned.End])
	}
	return cluster.NewWorker(*id, gradient, cluster.WithHeartbeat(*heartbeat)).Run(*socketPath)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "demo failed:", err)
	os.Exit(1)
}
