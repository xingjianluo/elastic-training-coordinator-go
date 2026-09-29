package cluster

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/membership"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/sharding"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/training"
)

func testLogger(t *testing.T) *log.Logger {
	t.Helper()
	return log.New(os.Stdout, "  "+t.Name()+": ", log.LstdFlags|log.Lmicroseconds)
}

func localGradient(params, target []float64) []float64 {
	return training.Gradient(params, target)
}

func contributeAll(t *testing.T, c *Coordinator, workers []string) {
	t.Helper()
	params, step, gen, plan := c.Snapshot()
	t.Logf("step=%d generation=%d members=%v ranges=%v decision=awaiting-one-gradient-per-member",
		step, gen, plan.Workers, plan.Ranges)
	order := append([]string(nil), workers...)
	rand.New(rand.NewPCG(uint64(step+1), uint64(gen+7))).Shuffle(len(order), func(i, j int) {
		order[i], order[j] = order[j], order[i]
	})
	model := training.NewModel(c.cfg.Dimension, 0.05, 0.9)
	target := model.Target
	errCh := make(chan error, len(order))
	results := make(chan []float64, len(order))
	for _, id := range order {
		go func(id string) {
			g := localGradient(params, target)
			out, _, code, err := c.Contribute(context.Background(), id, gen, step, g)
			if err != nil {
				errCh <- err
				return
			}
			t.Logf("worker=%s response=%s decision=gradient-accepted-or-committed", id, code)
			if code != CodeOK {
				errCh <- &unexpectedCodeError{id: id, code: code}
				return
			}
			results <- out
		}(id)
	}
	for range order {
		select {
		case err := <-errCh:
			t.Fatalf("contribute failed: %v", err)
		case <-results:
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for step commit")
		}
	}
}

type unexpectedCodeError struct{ id, code string }

func (e *unexpectedCodeError) Error() string { return "worker " + e.id + " got code " + e.code }

func almostEqual(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch got %d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("index %d got %.15f want %.15f", i, got[i], want[i])
		}
	}
}

func TestCoordinatorDeterministicAcrossShardCounts(t *testing.T) {
	const dim, steps = 11, 6
	for _, n := range []int{1, 2, 4} {
		t.Run("workers-"+string(rune('0'+n)), func(t *testing.T) {
			workers := make([]string, n)
			for i := range workers {
				workers[i] = "w" + string(rune('a'+i))
			}
			c := NewCoordinator(Config{
				Dimension:      dim,
				TotalSteps:     steps,
				InitialWorkers: n,
				Logger:         testLogger(t),
			})
			defer c.Close()
			for _, id := range workers {
				c.pending[id] = time.Now()
			}
			c.mu.Lock()
			c.activatePendingLocked(time.Now())
			c.mu.Unlock()

			t.Logf("members=%v decision=fixed-worker-count shard-count=%d", workers, n)
			for range steps {
				contributeAll(t, c, workers)
			}
			got, step, _, plan := c.Snapshot()
			t.Logf("members=%v ranges=%v final_step=%d decision=compare-with-single-node-baseline",
				plan.Workers, plan.Ranges, step)
			almostEqual(t, got, training.Baseline(dim, steps, 0.05, 0.9))
		})
	}
}

func TestDuplicateContributionDoesNotDoubleCount(t *testing.T) {
	c := NewCoordinator(Config{Dimension: 3, TotalSteps: 1, InitialWorkers: 1, Logger: testLogger(t)})
	defer c.Close()
	c.mu.Lock()
	c.pending["w0"] = time.Now()
	c.activatePendingLocked(time.Now())
	c.mu.Unlock()
	params, step, gen, _ := c.Snapshot()
	g := localGradient(params, training.NewModel(3, 0.05, 0.9).Target)

	first, _, code, err := c.Contribute(context.Background(), "w0", gen, step, g)
	if err != nil || code != CodeOK {
		t.Fatalf("first contribution: code=%s err=%v", code, err)
	}
	second, _, secondCode, err := c.Contribute(context.Background(), "w0", gen, step, g)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("members=[w0] step=%d first=%s duplicate=%s decision=duplicate-must-not-reapply",
		step, code, secondCode)
	if secondCode != CodeAlreadyCompleted {
		t.Fatalf("duplicate code=%s want %s", secondCode, CodeAlreadyCompleted)
	}
	for i := range first {
		if math.Abs(first[i]-second[i]) > 1e-15 {
			t.Fatalf("duplicate changed parameters: %v vs %v", first, second)
		}
	}
}

func activateTestWorkers(t *testing.T, c *Coordinator, workers []string) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, id := range workers {
		c.pending[id] = time.Now()
	}
	c.activatePendingLocked(time.Now())
}

func addTestWorker(c *Coordinator, id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.active[id] = &runtimeMember{id: id, lastBeat: time.Now()}
	c.reshardLocked(time.Now(), "test-join:"+id)
}

func gradientAtSnapshot(c *Coordinator) ([]float64, int, int) {
	params, step, gen, _ := c.Snapshot()
	target := training.NewModel(c.cfg.Dimension, 0.05, 0.9).Target
	return training.Gradient(params, target), gen, step
}

func waitForPending(c *Coordinator, want []string) {
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		got, _ := c.Diagnose()
		if len(got) == len(want) {
			return
		}
		time.Sleep(time.Millisecond)
	}
}

func TestElasticJoinLeaveRetriesStepAndMatchesBaseline(t *testing.T) {
	const dim, steps = 7, 3
	c := NewCoordinator(Config{
		Dimension:      dim,
		TotalSteps:     steps,
		InitialWorkers: 2,
		Logger:         testLogger(t),
	})
	defer c.Close()
	activateTestWorkers(t, c, []string{"wa", "wb"})

	contributeAll(t, c, []string{"wa", "wb"})

	// Failure in the middle of step 1: wa's partial gradient must be discarded
	// when wb is declared dead. The same optimizer step is retried in gen 1.
	g, gen, step := gradientAtSnapshot(c)
	result := make(chan struct {
		params []float64
		serial uint64
		code   string
		err    error
	}, 1)
	go func() {
		p, serial, code, err := c.Contribute(context.Background(), "wa", gen, step, g)
		result <- struct {
			params []float64
			serial uint64
			code   string
			err    error
		}{p, serial, code, err}
	}()
	waitForPending(c, []string{"wb"})
	c.FailWorker("wb")
	first := <-result
	if first.err != nil || first.code != CodeGenerationRestart {
		t.Fatalf("leave response code=%s err=%v", first.code, first.err)
	}
	_, newStep, newGen, plan := c.Snapshot()
	t.Logf("members=%v ranges=%v step=%d generation=%d decision=partial-gradient-discarded-after-leaf",
		plan.Workers, plan.Ranges, newStep, newGen)
	if newStep != 1 || newGen != 1 {
		t.Fatalf("after leave step=%d gen=%d, want step=1 gen=1", newStep, newGen)
	}
	addTestWorker(c, "wc")
	_, retryStep, retryGen, retryPlan := c.Snapshot()
	t.Logf("members=%v ranges=%v step=%d generation=%d decision=join-causes-reshard-before-remaining-steps",
		retryPlan.Workers, retryPlan.Ranges, retryStep, retryGen)
	contributeAll(t, c, []string{"wa", "wc"})
	contributeAll(t, c, []string{"wa", "wc"})

	got, finalStep, finalGen, finalPlan := c.Snapshot()
	t.Logf("members=%v ranges=%v step=%d generation=%d decision=compare-elastic-result-with-baseline",
		finalPlan.Workers, finalPlan.Ranges, finalStep, finalGen)
	almostEqual(t, got, training.Baseline(dim, steps, 0.05, 0.9))
}

func errOrCode(code string, err error) error {
	if err != nil {
		return err
	}
	if code != CodeOK {
		return fmt.Errorf("unexpected code %s", code)
	}
	return nil
}

func TestSlowWorkerIsDistinguishedFromDeadWorker(t *testing.T) {
	c := NewCoordinator(Config{
		Dimension:        2,
		TotalSteps:       1,
		InitialWorkers:   2,
		StepTimeout:      25 * time.Millisecond,
		HeartbeatTimeout: 500 * time.Millisecond,
		Logger:           testLogger(t),
	})
	defer c.Close()
	activateTestWorkers(t, c, []string{"fast", "slow"})
	g, gen, step := gradientAtSnapshot(c)
	stopHeartbeat := make(chan struct{})
	go func() {
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stopHeartbeat:
				return
			case <-ticker.C:
				_ = c.Heartbeat("slow")
			}
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_, _, code, err := c.Contribute(ctx, "fast", gen, step, g)
	close(stopHeartbeat)
	missing, state := c.Diagnose()
	t.Logf("members=[fast slow] missing=%v diagnosis=%s response-code=%s err=%v decision=heartbeat-fresh-means-slow-not-dead",
		missing, state, code, err)
	if !errors.Is(err, ErrSlowWorker) || state != membership.StateSlow {
		t.Fatalf("expected slow worker, got state=%s err=%v", state, err)
	}
}

func TestHeartbeatTimeoutIsDeadAndRebuildsGeneration(t *testing.T) {
	c := NewCoordinator(Config{
		Dimension:        2,
		TotalSteps:       1,
		InitialWorkers:   2,
		StepTimeout:      20 * time.Millisecond,
		HeartbeatTimeout: 40 * time.Millisecond,
		Logger:           testLogger(t),
	})
	defer c.Close()
	activateTestWorkers(t, c, []string{"alive", "dead"})
	g, gen, step := gradientAtSnapshot(c)
	stopAliveHeartbeat := make(chan struct{})
	go func() {
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stopAliveHeartbeat:
				return
			case <-ticker.C:
				_ = c.Heartbeat("alive")
			}
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, _, code, err := c.Contribute(ctx, "alive", gen, step, g)
	close(stopAliveHeartbeat)
	_, nextStep, nextGen, plan := c.Snapshot()
	t.Logf("members=%v ranges=%v step=%d generation=%d code=%s err=%v decision=no-heartbeat-past-hard-deadline-is-dead",
		plan.Workers, plan.Ranges, nextStep, nextGen, code, err)
	if err != nil || code != CodeGenerationRestart {
		t.Fatalf("expected generation restart, code=%s err=%v", code, err)
	}
	if nextStep != step || nextGen != gen+1 || len(plan.Workers) != 1 || plan.Workers[0] != "alive" {
		t.Fatalf("unexpected rebuilt view: step=%d gen=%d plan=%+v", nextStep, nextGen, plan)
	}
}

func TestCheckpointResumeContinuesAndMatchesBaseline(t *testing.T) {
	const dim, totalSteps = 7, 5
	workers := []string{"wa", "wb"}
	cfg := Config{
		Dimension:        dim,
		TotalSteps:       totalSteps,
		InitialWorkers:   len(workers),
		StepTimeout:      time.Second,
		HeartbeatTimeout: 10 * time.Second,
		CheckpointPath:   filepath.Join(t.TempDir(), "training.ckpt"),
		CheckpointEvery:  1,
		Logger:           testLogger(t),
	}
	first := NewCoordinator(cfg)
	activateTestWorkers(t, first, workers)
	contributeAll(t, first, workers)
	contributeAll(t, first, workers)
	params, savedStep, savedGen, savedPlan := first.Snapshot()
	t.Logf("members=%v ranges=%v next_step=%d generation=%d decision=checkpoint-contains-weights-optimizer-and-members",
		savedPlan.Workers, savedPlan.Ranges, savedStep, savedGen)
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	loaded, err := checkpoint.New(cfg.CheckpointPath).Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.NextStep != 2 || loaded.Generation != savedGen {
		t.Fatalf("loaded step=%d gen=%d", loaded.NextStep, loaded.Generation)
	}
	t.Logf("loaded members=%v next_step=%d decision=valid-magic-version-checksum-allows-restore",
		loaded.Members, loaded.NextStep)

	resumed := ResumeCoordinator(cfg, loaded)
	defer resumed.Close()
	resumedParams, resumedStep, resumedGen, resumedPlan := resumed.Snapshot()
	almostEqual(t, resumedParams, params)
	t.Logf("resumed members=%v ranges=%v next_step=%d generation=%d decision=continue-from-next-step-without-replaying-step",
		resumedPlan.Workers, resumedPlan.Ranges, resumedStep, resumedGen)
	for range totalSteps - 2 {
		contributeAll(t, resumed, workers)
	}
	got, finalStep, _, finalPlan := resumed.Snapshot()
	t.Logf("members=%v ranges=%v final_step=%d decision=resumed-result-compared-with-baseline",
		finalPlan.Workers, finalPlan.Ranges, finalStep)
	almostEqual(t, got, training.Baseline(dim, totalSteps, 0.05, 0.9))
}

func TestUnixSocketWorkersAndCompletedReconnect(t *testing.T) {
	const dim, steps, n = 8, 3, 3
	socketPath := filepath.Join(t.TempDir(), "coordinator.sock")
	cfg := Config{
		SocketPath:       socketPath,
		Dimension:        dim,
		TotalSteps:       steps,
		InitialWorkers:   n,
		StepTimeout:      5 * time.Second,
		HeartbeatTimeout: 10 * time.Second,
		Logger:           testLogger(t),
	}
	c := NewCoordinator(cfg)
	defer c.Close()
	listener, err := ListenUnix(socketPath)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	serveErr := make(chan error, 1)
	go func() { serveErr <- c.Serve(listener) }()

	gradient := func(_ string, _, _ int, owned sharding.Range, params []float64) []float64 {
		target := training.NewModel(dim, 0.05, 0.9).Target
		return training.Gradient(params, target[owned.Start:owned.End])
	}
	errCh := make(chan error, n)
	for i := range n {
		id := fmt.Sprintf("worker-%d", i)
		go func() { errCh <- NewWorker(id, gradient, WithHeartbeat(10*time.Millisecond)).Run(socketPath) }()
	}
	for range n {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
	got, finalStep, _, plan := c.Snapshot()
	t.Logf("socket-members=%v ranges=%v final_step=%d decision=unix-socket-ipc-completed",
		plan.Workers, plan.Ranges, finalStep)
	almostEqual(t, got, training.Baseline(dim, steps, 0.05, 0.9))

	// A completed worker process can reconnect, but it receives an idempotent
	// completion response and must not submit another gradient.
	if err := NewWorker("worker-0", gradient).Run(socketPath); err != nil {
		t.Fatalf("completed reconnect: %v", err)
	}
	afterReconnect, stepAfterReconnect, _, _ := c.Snapshot()
	t.Logf("reconnect-worker=worker-0 step=%d decision=already-completed-is-idempotent", stepAfterReconnect)
	almostEqual(t, afterReconnect, got)
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	if err := <-serveErr; err != nil {
		t.Fatal(err)
	}
}

func TestContributeShardMatchesBaselineAndRejectsMisalignment(t *testing.T) {
	const dim, steps, n = 9, 2, 3
	workers := []string{"wa", "wb", "wc"}
	c := NewCoordinator(Config{
		Dimension:      dim,
		TotalSteps:     steps,
		InitialWorkers: n,
		Logger:         testLogger(t),
	})
	defer c.Close()
	activateTestWorkers(t, c, workers)

	for range steps {
		params, step, gen, plan := c.Snapshot()
		target := training.NewModel(dim, 0.05, 0.9).Target
		t.Logf("members=%v ranges=%v step=%d decision=each-worker-submits-only-owned-gradient-shard",
			plan.Workers, plan.Ranges, step)
		errCh := make(chan error, n)
		for i, id := range workers {
			go func(i int, id string) {
				r := plan.Ranges[i]
				g := training.Gradient(params[r.Start:r.End], target[r.Start:r.End])
				_, _, code, err := c.ContributeShard(context.Background(), id, gen, step, r, g)
				errCh <- errOrCode(code, err)
			}(i, id)
		}
		for range n {
			if err := <-errCh; err != nil {
				t.Fatal(err)
			}
		}
	}
	got, finalStep, _, plan := c.Snapshot()
	t.Logf("members=%v ranges=%v step=%d decision=allgathered-shards-compared-with-baseline",
		plan.Workers, plan.Ranges, finalStep)
	almostEqual(t, got, training.Baseline(dim, steps, 0.05, 0.9))
}

func TestContributeShardRejectsWrongRangeAndLength(t *testing.T) {
	const dim = 6
	workers := []string{"wa", "wb"}
	c := NewCoordinator(Config{Dimension: dim, TotalSteps: 1, InitialWorkers: 2, Logger: testLogger(t)})
	defer c.Close()
	activateTestWorkers(t, c, workers)
	params, step, gen, plan := c.Snapshot()
	target := training.NewModel(dim, 0.05, 0.9).Target
	goodRange := plan.Ranges[0]

	wrongRange := sharding.Range{Start: goodRange.Start, End: goodRange.End + 1}
	_, _, code, err := c.ContributeShard(context.Background(), "wa", gen, step, wrongRange,
		training.Gradient(params[goodRange.Start:goodRange.End], target[goodRange.Start:goodRange.End]))
	t.Logf("members=%v ranges=%v sent=%v decision=wrong-range-must-be-rejected code=%s err=%v",
		plan.Workers, plan.Ranges, wrongRange, code, err)
	if code != CodeMisalignedShards {
		t.Fatalf("wrong range code=%s", code)
	}

	wrongLengthValues := make([]float64, goodRange.Len()+1)
	_, _, code, err = c.ContributeShard(context.Background(), "wa", gen, step, goodRange, wrongLengthValues)
	t.Logf("members=%v ranges=%v sent-range=%v sent-length=%d decision=wrong-length-must-be-rejected code=%s err=%v",
		plan.Workers, plan.Ranges, goodRange, len(wrongLengthValues), code, err)
	if code != CodeMisalignedShards {
		t.Fatalf("wrong length code=%s", code)
	}
}

func contributeShardsAll(t *testing.T, c *Coordinator, workers []string) {
	t.Helper()
	params, step, gen, plan := c.Snapshot()
	target := training.NewModel(c.cfg.Dimension, 0.05, 0.9).Target
	t.Logf("members=%v ranges=%v step=%d generation=%d decision=allgather-one-gradient-shard-per-worker",
		plan.Workers, plan.Ranges, step, gen)
	order := append([]string(nil), workers...)
	rand.New(rand.NewPCG(uint64(step+17), uint64(gen+23))).Shuffle(len(order), func(i, j int) {
		order[i], order[j] = order[j], order[i]
	})
	errCh := make(chan error, len(order))
	for _, id := range order {
		go func(id string) {
			index := slices.Index(plan.Workers, id)
			if index < 0 {
				errCh <- fmt.Errorf("worker %s missing from plan", id)
				return
			}
			r := plan.Ranges[index]
			g := training.Gradient(params[r.Start:r.End], target[r.Start:r.End])
			_, _, code, err := c.ContributeShard(context.Background(), id, gen, step, r, g)
			errCh <- errOrCode(code, err)
		}(id)
	}
	for range order {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
}

func TestShardAllGatherDeterministicAcrossShardCounts(t *testing.T) {
	const dim, steps = 11, 3
	for _, n := range []int{1, 2, 4} {
		t.Run("shards-"+string(rune('0'+n)), func(t *testing.T) {
			workers := make([]string, n)
			for i := range workers {
				workers[i] = "w" + string(rune('a'+i))
			}
			c := NewCoordinator(Config{Dimension: dim, TotalSteps: steps, InitialWorkers: n, Logger: testLogger(t)})
			defer c.Close()
			activateTestWorkers(t, c, workers)
			for range steps {
				contributeShardsAll(t, c, workers)
			}
			got, finalStep, _, plan := c.Snapshot()
			t.Logf("members=%v ranges=%v final_step=%d decision=sharded-allgather-compared-with-baseline",
				plan.Workers, plan.Ranges, finalStep)
			almostEqual(t, got, training.Baseline(dim, steps, 0.05, 0.9))
		})
	}
}

func TestShardCollectiveFailureDiscardsPartialShardAndMatchesBaseline(t *testing.T) {
	const dim, steps = 7, 3
	c := NewCoordinator(Config{Dimension: dim, TotalSteps: steps, InitialWorkers: 2, Logger: testLogger(t)})
	defer c.Close()
	activateTestWorkers(t, c, []string{"wa", "wb"})
	contributeShardsAll(t, c, []string{"wa", "wb"})

	params, step, gen, plan := c.Snapshot()
	target := training.NewModel(dim, 0.05, 0.9).Target
	waRange := plan.Ranges[slices.Index(plan.Workers, "wa")]
	waGrad := training.Gradient(params[waRange.Start:waRange.End], target[waRange.Start:waRange.End])
	result := make(chan string, 1)
	go func() {
		_, _, code, err := c.ContributeShard(context.Background(), "wa", gen, step, waRange, waGrad)
		if err != nil {
			result <- err.Error()
			return
		}
		result <- code
	}()
	waitForPending(c, []string{"wb"})
	c.FailWorker("wb")
	if code := <-result; code != CodeGenerationRestart {
		t.Fatalf("failure during shard collective got %s", code)
	}
	addTestWorker(c, "wc")
	_, retryStep, retryGen, retryPlan := c.Snapshot()
	t.Logf("members=%v ranges=%v step=%d generation=%d decision=partial-shard-discarded-after-mid-collective-failure",
		retryPlan.Workers, retryPlan.Ranges, retryStep, retryGen)
	contributeShardsAll(t, c, []string{"wa", "wc"})
	contributeShardsAll(t, c, []string{"wa", "wc"})
	got, finalStep, _, finalPlan := c.Snapshot()
	t.Logf("members=%v ranges=%v final_step=%d decision=recovered-sharded-run-compared-with-baseline",
		finalPlan.Workers, finalPlan.Ranges, finalStep)
	almostEqual(t, got, training.Baseline(dim, steps, 0.05, 0.9))
}
