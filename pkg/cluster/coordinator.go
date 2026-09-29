package cluster

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/collective"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/membership"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/sharding"
	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/training"
)

var (
	ErrSlowWorker         = errors.New("cluster: slow worker exceeded the step deadline but is still heartbeating")
	ErrWorkerDead         = errors.New("cluster: worker is dead because its heartbeat deadline elapsed")
	ErrDuplicateWorker    = errors.New("cluster: duplicate worker identity")
	ErrCoordinatorClosed  = errors.New("cluster: coordinator is closed")
	ErrInvalidRequest     = errors.New("cluster: invalid request")
	ErrInvalidShardRange  = errors.New("cluster: invalid shard range")
	ErrMixedContributions = errors.New("cluster: full-replica and shard contributions cannot be mixed in one step")
)

type Config struct {
	SocketPath       string
	Dimension        int
	TotalSteps       int
	InitialWorkers   int
	StepTimeout      time.Duration
	HeartbeatTimeout time.Duration
	CheckpointPath   string
	CheckpointEvery  int
	Logger           *log.Logger
}

type Coordinator struct {
	cfg Config
	cp  *checkpoint.Checkpointer
	log *log.Logger

	mu          sync.Mutex
	waiters     map[chan struct{}]struct{}
	closedCh    chan struct{}
	closed      bool
	model       *training.Model
	gen         int
	step        int
	total       int
	dim         int
	active      map[string]*runtimeMember
	pending     map[string]time.Time
	pendingConn map[string]uint64
	plan        sharding.Plan

	contrib      map[string][]float64
	contribShard map[string]shardContribution
	stepStarted  time.Time
	commitSerial uint64
	runDone      bool
	connSerial   uint64
}

type runtimeMember struct {
	id         string
	lastBeat   time.Time
	connection uint64
}

type shardContribution struct {
	shard  sharding.Range
	values []float64
}

func NewCoordinator(cfg Config) *Coordinator {
	return newCoordinator(cfg, checkpoint.State{}, false)
}

func ResumeCoordinator(cfg Config, state checkpoint.State) *Coordinator {
	return newCoordinator(cfg, state, true)
}

func newCoordinator(cfg Config, state checkpoint.State, restored bool) *Coordinator {
	if cfg.Dimension <= 0 {
		panic(fmt.Sprintf("cluster: dimension must be positive, got %d", cfg.Dimension))
	}
	if cfg.TotalSteps < 0 {
		panic(fmt.Sprintf("cluster: total steps cannot be negative: %d", cfg.TotalSteps))
	}
	if cfg.InitialWorkers <= 0 {
		cfg.InitialWorkers = 1
	}
	if cfg.StepTimeout <= 0 {
		cfg.StepTimeout = 250 * time.Millisecond
	}
	if cfg.HeartbeatTimeout <= 0 {
		cfg.HeartbeatTimeout = cfg.StepTimeout * 3
	}
	if cfg.CheckpointEvery == 0 {
		cfg.CheckpointEvery = 1
	}
	if cfg.Logger == nil {
		cfg.Logger = log.New(os.Stderr, "", log.LstdFlags|log.Lmicroseconds)
	}
	c := &Coordinator{
		cfg:          cfg,
		log:          cfg.Logger,
		dim:          cfg.Dimension,
		total:        cfg.TotalSteps,
		active:       map[string]*runtimeMember{},
		pending:      map[string]time.Time{},
		pendingConn:  map[string]uint64{},
		contrib:      map[string][]float64{},
		contribShard: map[string]shardContribution{},
	}
	if cfg.CheckpointPath != "" {
		c.cp = checkpoint.New(cfg.CheckpointPath)
	}
	c.waiters = map[chan struct{}]struct{}{}
	c.closedCh = make(chan struct{})
	now := time.Now()
	if restored {
		c.loadStateLocked(state, now)
	} else {
		c.model = training.NewModel(cfg.Dimension, 0.05, 0.9)
		c.stepStarted = now
	}
	go c.reapLoop()
	return c
}

func (c *Coordinator) loadStateLocked(state checkpoint.State, now time.Time) {
	if len(state.Params) != c.dim {
		panic(fmt.Sprintf("cluster: restored dimension %d does not match config %d", len(state.Params), c.dim))
	}
	c.model = &training.Model{
		Params:   append([]float64(nil), state.Params...),
		Momentum: append([]float64(nil), state.Momentum...),
		Target:   append([]float64(nil), state.Target...),
		LR:       0.05,
		Mu:       0.9,
	}
	c.gen = state.Generation
	c.step = state.NextStep
	c.total = state.TotalSteps
	c.runDone = state.RunCompleted
	c.active = map[string]*runtimeMember{}
	for _, id := range state.Members {
		c.active[id] = &runtimeMember{id: id, lastBeat: now, connection: 0}
	}
	if c.total > 0 && len(c.active) > 0 {
		view, err := membership.NewView(c.gen, c.dim, now, state.Members)
		if err != nil {
			panic(fmt.Sprintf("cluster: restored members cannot form a plan: %v", err))
		}
		c.plan = view.Plan
	}
	c.contrib = map[string][]float64{}
	c.contribShard = map[string]shardContribution{}
	c.stepStarted = now
	c.log.Printf("restore decision: generation=%d next_step=%d run_done=%v members=%v",
		c.gen, c.step, c.runDone, state.Members)
	c.logPlanLocked("restored-members")
}

func (c *Coordinator) Serve(l net.Listener) error {
	for {
		conn, err := l.Accept()
		if err != nil {
			c.mu.Lock()
			closed := c.closed
			c.mu.Unlock()
			if closed {
				return nil
			}
			return err
		}
		go c.serveConn(conn)
	}
}

func (c *Coordinator) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	close(c.closedCh)
	c.broadcastLocked()
	c.mu.Unlock()
	return nil
}

func (c *Coordinator) serveConn(conn net.Conn) {
	defer conn.Close()
	workerID := ""
	connID := uint64(0)
	owns := false
	defer func() {
		if owns {
			c.handleDisconnect(workerID, connID)
		}
	}()

	dec := json.NewDecoder(conn)
	enc := json.NewEncoder(conn)
	for {
		var req Request
		if err := dec.Decode(&req); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, net.ErrClosed) {
				return
			}
			c.log.Printf("decode error from %s: %v", workerID, err)
			return
		}
		now := time.Now()
		if req.Type == "join" {
			workerID = req.WorkerID
			connID = c.joinConnection(req.WorkerID, now)
			owns = true
		}
		if workerID == "" || req.WorkerID != workerID {
			_ = enc.Encode(errorResponse(CodeInvalidRequest, "a worker must join before other requests and cannot change identity"))
			return
		}
		resp := c.handleRequest(req, connID, now)
		resp.ReqID = req.Seq
		if err := enc.Encode(&resp); err != nil {
			return
		}
		if resp.RunDone || resp.Code == CodeInvalidRequest || resp.Code == CodeAlreadyCompleted {
			return
		}
	}
}

func (c *Coordinator) joinConnection(id string, now time.Time) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0
	}
	c.connSerial++
	connID := c.connSerial
	if m, ok := c.active[id]; ok {
		m.lastBeat = now
		m.connection = connID
		c.log.Printf("join decision: worker=%s generation=%d reconnect=true members=%v", id, c.gen, c.sortedActiveLocked())
		return connID
	}
	if c.runDone {
		c.log.Printf("join decision: worker=%s rejected-reason=run-already-completed", id)
		return connID
	}
	if !c.planBuiltLocked() && len(c.active) == 0 {
		c.pending[id] = now
		c.pendingConn[id] = connID
		if len(c.pending) >= c.cfg.InitialWorkers {
			c.activatePendingLocked(now)
		} else {
			c.log.Printf("join decision: worker=%s state=pending count=%d/%d", id, len(c.pending), c.cfg.InitialWorkers)
			c.broadcastLocked()
			return connID
		}
	} else {
		delete(c.pending, id)
		c.active[id] = &runtimeMember{id: id, lastBeat: now, connection: connID}
		c.reshardLocked(now, "join:"+id)
	}
	c.log.Printf("join decision: worker=%s generation=%d reconnect=false members=%v", id, c.gen, c.sortedActiveLocked())
	return connID
}

func (c *Coordinator) activatePendingLocked(now time.Time) {
	ids := make([]string, 0, len(c.pending))
	for id := range c.pending {
		ids = append(ids, id)
	}
	view, err := membership.NewView(c.gen, c.dim, now, ids)
	if err != nil {
		c.log.Printf("activation rejected: reason=%v ids=%v", err, ids)
		return
	}
	c.plan = view.Plan
	c.active = make(map[string]*runtimeMember, len(ids))
	for _, m := range view.Members {
		c.active[m.ID] = &runtimeMember{id: m.ID, lastBeat: now, connection: c.pendingConn[m.ID]}
	}
	c.pending = map[string]time.Time{}
	c.pendingConn = map[string]uint64{}
	c.stepStarted = now
	c.log.Printf("activation decision: generation=%d members=%v", c.gen, c.sortedActiveLocked())
	c.logPlanLocked("initial-shards")
	c.broadcastLocked()
}

func (c *Coordinator) handleRequest(req Request, connID uint64, now time.Time) Response {
	switch req.Type {
	case "join":
		return c.currentViewResponse(connID, now)
	case "heartbeat":
		return c.heartbeat(req.WorkerID, connID, now)
	case "contribute":
		return c.contributeOverNetwork(req, connID, now)
	default:
		return errorResponse(CodeInvalidRequest, "unknown request type "+req.Type)
	}
}

func (c *Coordinator) currentViewResponse(connID uint64, now time.Time) Response {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.runDone {
		return Response{Code: CodeAlreadyCompleted, Message: "training run already completed", Generation: c.gen, Step: c.step, RunDone: true, Values: c.snapshotParamsLocked()}
	}
	for !c.closed && !c.planBuiltLocked() {
		if err := c.waitCtxLocked(context.Background()); err != nil {
			return errorResponse(CodeWorkerDead, err.Error())
		}
	}
	if c.closed {
		return errorResponse(CodeWorkerDead, "coordinator closed")
	}
	if m := c.active[workerForConn(c, connID)]; m != nil {
		m.lastBeat = now
	}
	return c.shardResponseLocked(workerForConn(c, connID), CodeOK, "", c.snapshotParamsLocked())
}

// workerForConn finds an ID while holding the mutex. Connection ownership is
// normally passed explicitly; this small helper only serves the join reply.
func workerForConn(c *Coordinator, connID uint64) string {
	for id, m := range c.active {
		if m.connection == connID {
			return id
		}
	}
	return ""
}

func (c *Coordinator) heartbeat(id string, connID uint64, now time.Time) Response {
	c.mu.Lock()
	defer c.mu.Unlock()
	m, ok := c.active[id]
	if !ok || m.connection != connID {
		return Response{Code: CodeGenerationRestart, Message: "worker is not in the current generation", Generation: c.gen, Step: c.step, Plan: c.plan}
	}
	m.lastBeat = now
	return Response{Code: CodeOK, Generation: c.gen, Step: c.step, Plan: c.plan}
}

func (c *Coordinator) contributeOverNetwork(req Request, connID uint64, now time.Time) Response {
	c.mu.Lock()
	m, ok := c.active[req.WorkerID]
	if !ok || m.connection != connID {
		gen := c.gen
		plan := c.plan
		step := c.step
		c.mu.Unlock()
		return Response{Code: CodeGenerationRestart, Message: "worker is not in current generation", Generation: gen, Step: step, Plan: plan}
	}
	wantRange, err := c.plan.ShardFor(req.WorkerID)
	if err != nil {
		message := err.Error()
		c.mu.Unlock()
		return errorResponse(CodeMisalignedShards, message)
	}
	if req.Shard != wantRange {
		message := fmt.Sprintf("worker %q sent shard %v but current plan owns %v", req.WorkerID, req.Shard, wantRange)
		c.mu.Unlock()
		return errorResponse(CodeMisalignedShards, message)
	}
	if len(req.Values) != wantRange.Len() {
		message := fmt.Sprintf("worker %q sent %d values for shard %v, expected %d",
			req.WorkerID, len(req.Values), wantRange, wantRange.Len())
		c.mu.Unlock()
		return errorResponse(CodeMisalignedShards, message)
	}
	m.lastBeat = now
	c.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), c.cfg.StepTimeout)
	defer cancel()
	params, _, respCode, err := c.ContributeShard(ctx, req.WorkerID, req.Generation, req.Step, req.Shard, req.Values)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		if errors.Is(err, ErrSlowWorker) {
			return c.shardResponseLocked(req.WorkerID, CodeSlowWorker, err.Error(), params)
		}
		if errors.Is(err, context.DeadlineExceeded) {
			missing := c.pendingContributorsLocked()
			return c.shardResponseLocked(req.WorkerID, CodeSlowWorker,
				fmt.Sprintf("step deadline exceeded; pending workers=%v", missing), params)
		}
		return c.shardResponseLocked(req.WorkerID, CodeWorkerDead, err.Error(), params)
	}
	return c.shardResponseLocked(req.WorkerID, respCode, "", params)
}

func (c *Coordinator) shardResponseLocked(workerID, code, message string, full []float64) Response {
	resp := Response{
		Code:       code,
		Message:    message,
		Generation: c.gen,
		Step:       c.step,
		Plan:       c.plan,
		RunDone:    c.runDone || c.step >= c.total,
		Committed:  code == CodeOK,
	}
	if r, err := c.plan.ShardFor(workerID); err == nil && len(full) == c.plan.Dimension {
		resp.Values = append([]float64(nil), full[r.Start:r.End]...)
	}
	return resp
}

// Contribute submits one full local gradient replica. It blocks until either
// all current members have contributed and one optimizer step commits, or a
// membership change invalidates the generation. Partial gradients are always
// cleared on invalidation, so they can never be committed after a worker joins
// or leaves.
func (c *Coordinator) Contribute(ctx context.Context, id string, gen, step int, gradient []float64) ([]float64, uint64, string, error) {
	if len(gradient) != c.dim {
		return nil, 0, CodeInvalidRequest, fmt.Errorf("%w: gradient dimension %d, model dimension %d",
			ErrInvalidRequest, len(gradient), c.dim)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for !c.closed && !c.planBuiltLocked() && !c.runDone {
		if err := c.waitCtxLocked(ctx); err != nil {
			return nil, 0, CodeWorkerDead, err
		}
	}
	if c.closed {
		return nil, 0, CodeWorkerDead, ErrCoordinatorClosed
	}
	if c.runDone || step < c.step || gen < c.gen {
		return c.snapshotParamsLocked(), c.commitSerial, CodeAlreadyCompleted, nil
	}
	m, ok := c.active[id]
	if !ok {
		return c.snapshotParamsLocked(), c.commitSerial, CodeGenerationRestart, nil
	}
	if gen != c.gen || step != c.step {
		return c.snapshotParamsLocked(), c.commitSerial, CodeGenerationRestart, nil
	}
	if len(c.contribShard) > 0 {
		return nil, 0, CodeInvalidRequest, ErrMixedContributions
	}
	if _, duplicate := c.contrib[id]; duplicate {
		c.log.Printf("contribute decision: worker=%s generation=%d step=%d rejected-reason=duplicate", id, gen, step)
		return c.snapshotParamsLocked(), c.commitSerial, CodeAlreadyCompleted, nil
	}

	entry := c.commitSerial
	entryGen := c.gen
	entryStep := c.step
	c.contrib[id] = append([]float64(nil), gradient...)
	m.lastBeat = time.Now()
	c.log.Printf("contribute decision: worker=%s generation=%d step=%d accepted=true pending=%v",
		id, gen, step, c.pendingContributorsLocked())
	if len(c.contrib) == len(c.active) && len(c.contribShard) == 0 {
		c.commitStepLocked()
	}
	for !c.closed {
		if c.runDone || c.commitSerial != entry {
			return c.snapshotParamsLocked(), c.commitSerial, CodeOK, nil
		}
		if c.gen != entryGen || c.step != entryStep {
			return c.snapshotParamsLocked(), c.commitSerial, CodeGenerationRestart, nil
		}
		if err := c.waitCtxLocked(ctx); err != nil {
			missing := c.pendingContributorsLocked()
			state := c.diagnoseLocked(time.Now())
			if state == membership.StateSlow {
				return nil, entry, CodeSlowWorker, fmt.Errorf("%w: pending=%v", ErrSlowWorker, missing)
			}
			return nil, entry, CodeWorkerDead, fmt.Errorf("%w: pending=%v", ErrWorkerDead, missing)
		}
	}
	return nil, entry, CodeWorkerDead, ErrCoordinatorClosed
}

// ContributeShard submits only the gradient range owned by the worker. The
// coordinator validates the range against the current plan and allgathers all
// owned shards before applying one optimizer step.
func (c *Coordinator) ContributeShard(ctx context.Context, id string, gen, step int, r sharding.Range, gradient []float64) ([]float64, uint64, string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for !c.closed && !c.planBuiltLocked() && !c.runDone {
		if err := c.waitCtxLocked(ctx); err != nil {
			return nil, 0, CodeWorkerDead, err
		}
	}
	if c.closed {
		return nil, 0, CodeWorkerDead, ErrCoordinatorClosed
	}
	if c.runDone || step < c.step || gen < c.gen {
		return c.snapshotParamsLocked(), c.commitSerial, CodeAlreadyCompleted, nil
	}
	m, ok := c.active[id]
	if !ok {
		return c.snapshotParamsLocked(), c.commitSerial, CodeGenerationRestart, nil
	}
	if gen != c.gen || step != c.step {
		return c.snapshotParamsLocked(), c.commitSerial, CodeGenerationRestart, nil
	}
	want, err := c.plan.ShardFor(id)
	if err != nil || r != want || len(gradient) != want.Len() {
		return nil, 0, CodeMisalignedShards, fmt.Errorf("%w: worker=%s sent=%v len=%d plan=%v",
			ErrInvalidShardRange, id, r, len(gradient), want)
	}
	if len(c.contrib) > 0 {
		return nil, 0, CodeInvalidRequest, ErrMixedContributions
	}
	if _, duplicate := c.contribShard[id]; duplicate {
		c.log.Printf("contribute decision: worker=%s generation=%d step=%d rejected-reason=duplicate-shard", id, gen, step)
		return c.snapshotParamsLocked(), c.commitSerial, CodeAlreadyCompleted, nil
	}

	entry := c.commitSerial
	entryGen := c.gen
	entryStep := c.step
	c.contribShard[id] = shardContribution{shard: r, values: append([]float64(nil), gradient...)}
	m.lastBeat = time.Now()
	c.log.Printf("contribute decision: worker=%s generation=%d step=%d shard=%v accepted=true pending=%v",
		id, gen, step, r, c.pendingContributorsLocked())
	if len(c.contribShard) == len(c.active) && len(c.contrib) == 0 {
		c.commitStepLocked()
	}
	for !c.closed {
		if c.runDone || c.commitSerial != entry {
			return c.snapshotParamsLocked(), c.commitSerial, CodeOK, nil
		}
		if c.gen != entryGen || c.step != entryStep {
			return c.snapshotParamsLocked(), c.commitSerial, CodeGenerationRestart, nil
		}
		if err := c.waitCtxLocked(ctx); err != nil {
			missing := c.pendingContributorsLocked()
			state := c.diagnoseLocked(time.Now())
			if state == membership.StateSlow {
				return nil, entry, CodeSlowWorker, fmt.Errorf("%w: pending=%v", ErrSlowWorker, missing)
			}
			return nil, entry, CodeWorkerDead, fmt.Errorf("%w: pending=%v", ErrWorkerDead, missing)
		}
	}
	return nil, entry, CodeWorkerDead, ErrCoordinatorClosed
}

func (c *Coordinator) commitStepLocked() {
	ids := c.sortedActiveLocked()
	var grad []float64
	mode := "allreduce"
	if len(c.contribShard) > 0 {
		mode = "allgather"
		shards := make(map[string][]float64, len(ids))
		for _, id := range ids {
			shards[id] = c.contribShard[id].values
		}
		assembled, err := collective.AllGather(shards, c.plan)
		if err != nil {
			c.log.Printf("commit rejected: generation=%d collective=allgather reason=%v members=%v", c.gen, err, ids)
			return
		}
		grad = assembled
	} else {
		replicas := make(map[string][]float64, len(ids))
		for _, id := range ids {
			replicas[id] = c.contrib[id]
		}
		reduced, err := collective.AllReduce(replicas, collective.OpSum)
		if err != nil {
			c.log.Printf("commit rejected: generation=%d collective=allreduce reason=%v members=%v", c.gen, err, ids)
			return
		}
		grad = reduced
	}
	n := float64(len(ids))
	if mode == "allreduce" {
		for i := range grad {
			grad[i] /= n
		}
	}
	c.model.Apply(grad)
	c.step++
	c.commitSerial++
	c.contrib = map[string][]float64{}
	c.contribShard = map[string]shardContribution{}
	c.stepStarted = time.Now()
	c.log.Printf("commit decision: generation=%d completed_step=%d member_count=%d collective=%s average_denominator=%v serial=%d",
		c.gen, c.step-1, len(ids), mode, map[bool]float64{true: n, false: 1}[mode == "allreduce"], c.commitSerial)
	if c.step >= c.total {
		c.runDone = true
	}
	if err := c.saveCheckpointLocked(); err != nil {
		c.log.Printf("checkpoint decision: step=%d rejected-reason=%v", c.step, err)
	}
	if c.runDone {
		c.log.Printf("run decision: completed=true final_members=%v", ids)
	}
	c.broadcastLocked()
}

func (c *Coordinator) handleDisconnect(id string, connID uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if id == "" {
		return
	}
	if m, ok := c.active[id]; ok && m.connection == connID && !c.runDone {
		delete(c.active, id)
		c.log.Printf("failure decision: worker=%s reason=transport-closed state=dead generation=%d", id, c.gen)
		c.reshardLocked(time.Now(), "leave:"+id)
		return
	}
	if _, pending := c.pending[id]; pending {
		delete(c.pending, id)
		c.log.Printf("failure decision: worker=%s reason=transport-closed state=pending", id)
		c.broadcastLocked()
	}
}

// FailWorker deterministically simulates a hard worker crash in tests. A soft
// straggler is instead a live worker that keeps heartbeating but withholds its
// contribution.
func (c *Coordinator) FailWorker(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.active[id]; !ok {
		return
	}
	delete(c.active, id)
	c.reshardLocked(time.Now(), "simulated-crash:"+id)
}

func (c *Coordinator) Heartbeat(id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	m, ok := c.active[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrWorkerDead, id)
	}
	m.lastBeat = time.Now()
	return nil
}

func (c *Coordinator) reshardLocked(now time.Time, reason string) {
	ids := make([]string, 0, len(c.active))
	for id := range c.active {
		ids = append(ids, id)
	}
	c.gen++
	c.contrib = map[string][]float64{}
	c.contribShard = map[string]shardContribution{}
	if len(ids) == 0 {
		c.plan = sharding.Plan{}
		c.stepStarted = now
		c.log.Printf("reshard decision: reason=%s generation=%d members=[] waiting-for-replacement=true", reason, c.gen)
		c.broadcastLocked()
		return
	}
	view, err := membership.NewView(c.gen, c.dim, now, ids)
	if err != nil {
		c.log.Printf("reshard rejected: reason=%v members=%v", err, ids)
		c.plan = sharding.Plan{}
		c.broadcastLocked()
		return
	}
	c.plan = view.Plan
	for _, id := range ids {
		c.active[id].lastBeat = now
	}
	c.stepStarted = now
	c.log.Printf("reshard decision: reason=%s generation=%d members=%v", reason, c.gen, ids)
	c.logPlanLocked("elastic-shards")
	if err := c.saveCheckpointLocked(); err != nil {
		c.log.Printf("checkpoint decision: generation=%d rejected-reason=%v", c.gen, err)
	}
	c.broadcastLocked()
}

func (c *Coordinator) reapLoop() {
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		c.mu.Lock()
		if c.closed {
			c.mu.Unlock()
			return
		}
		now := time.Now()
		var dead []string
		for id, m := range c.active {
			if now.Sub(m.lastBeat) >= c.cfg.HeartbeatTimeout {
				dead = append(dead, id)
			}
		}
		if len(dead) == 0 {
			c.mu.Unlock()
			continue
		}
		for _, id := range dead {
			delete(c.active, id)
			c.log.Printf("failure decision: worker=%s reason=heartbeat-timeout state=dead", id)
		}
		c.reshardLocked(now, "heartbeat-timeout:"+strings.Join(dead, ","))
		c.mu.Unlock()
	}
}

func (c *Coordinator) Diagnose() (missing []string, state membership.State) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.pendingContributorsLocked(), c.diagnoseLocked(time.Now())
}

func (c *Coordinator) diagnoseLocked(now time.Time) membership.State {
	state := membership.StateAlive
	for _, id := range c.pendingContributorsLocked() {
		m := c.active[id]
		if m == nil {
			return membership.StateDead
		}
		switch {
		case now.Sub(m.lastBeat) >= c.cfg.HeartbeatTimeout:
			return membership.StateDead
		case now.Sub(c.stepStarted) >= c.cfg.StepTimeout:
			state = membership.StateSlow
		}
	}
	return state
}

func (c *Coordinator) WaitUntilReady(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	for !c.closed && !c.planBuiltLocked() && !c.runDone {
		if err := c.waitCtxLocked(ctx); err != nil {
			return err
		}
	}
	if c.closed {
		return ErrCoordinatorClosed
	}
	return nil
}

func (c *Coordinator) Snapshot() ([]float64, int, int, sharding.Plan) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.snapshotParamsLocked(), c.step, c.gen, c.plan
}

func (c *Coordinator) Members() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sortedActiveLocked()
}

func (c *Coordinator) SaveCheckpoint() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.saveCheckpointLocked()
}

func (c *Coordinator) CheckpointState() checkpoint.State {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stateLocked()
}

func completedWorkersLocked(runCompleted bool, members []string) []string {
	if !runCompleted {
		return nil
	}
	return append([]string(nil), members...)
}

func (c *Coordinator) shouldCheckpointLocked() bool {
	if c.runDone {
		return true
	}
	every := c.cfg.CheckpointEvery
	if every <= 0 {
		every = 1
	}
	return c.step > 0 && c.step%every == 0
}

func (c *Coordinator) stateLocked() checkpoint.State {
	members := c.sortedActiveLocked()
	return checkpoint.State{
		Version:          int(checkpoint.CurrentVersion),
		Generation:       c.gen,
		NextStep:         c.step,
		TotalSteps:       c.total,
		Params:           c.snapshotParamsLocked(),
		Momentum:         append([]float64(nil), c.model.Momentum...),
		Target:           append([]float64(nil), c.model.Target...),
		Members:          members,
		CompletedWorkers: completedWorkersLocked(c.runDone, members),
		RunCompleted:     c.runDone,
	}
}

func (c *Coordinator) saveCheckpointLocked() error {
	if c.cp == nil || !c.shouldCheckpointLocked() {
		return nil
	}
	state := c.stateLocked()
	if err := c.cp.Save(state); err != nil {
		return err
	}
	c.log.Printf("checkpoint decision: saved=true path=%s generation=%d next_step=%d members=%v",
		c.cp.Path(), state.Generation, state.NextStep, state.Members)
	return nil
}

func (c *Coordinator) waitCtxLocked(ctx context.Context) error {
	ch := make(chan struct{}, 1)
	c.waiters[ch] = struct{}{}
	c.mu.Unlock()
	select {
	case <-ch:
		c.mu.Lock()
		delete(c.waiters, ch)
		return nil
	case <-ctx.Done():
		c.mu.Lock()
		delete(c.waiters, ch)
		return ctx.Err()
	case <-c.closedCh:
		c.mu.Lock()
		delete(c.waiters, ch)
		return ErrCoordinatorClosed
	}
}

func (c *Coordinator) broadcastLocked() {
	for ch := range c.waiters {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

func (c *Coordinator) planBuiltLocked() bool { return len(c.plan.Workers) > 0 }

func (c *Coordinator) sortedActiveLocked() []string {
	ids := make([]string, 0, len(c.active))
	for id := range c.active {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

func (c *Coordinator) pendingContributorsLocked() []string {
	if !c.planBuiltLocked() || c.runDone {
		return nil
	}
	missing := make([]string, 0)
	for _, id := range c.plan.Workers {
		_, full := c.contrib[id]
		_, partial := c.contribShard[id]
		if !full && !partial {
			missing = append(missing, id)
		}
	}
	return missing
}

func (c *Coordinator) snapshotParamsLocked() []float64 {
	if c.model == nil {
		return nil
	}
	return append([]float64(nil), c.model.Params...)
}

func (c *Coordinator) logPlanLocked(reason string) {
	for i, id := range c.plan.Workers {
		r := c.plan.Ranges[i]
		c.log.Printf("shard decision: reason=%s generation=%d worker=%s range=[%d,%d) length=%d",
			reason, c.gen, id, r.Start, r.End, r.Len())
	}
}

func errorResponse(code, message string) Response {
	return Response{Code: code, Message: message}
}

// ListenUnix creates and removes a local Unix-domain socket. It uses only
// local filesystem/IPC capabilities and never opens a network endpoint.
func ListenUnix(socketPath string) (net.Listener, error) {
	if socketPath == "" {
		return nil, fmt.Errorf("%w: empty unix socket path", ErrInvalidRequest)
	}
	if err := os.MkdirAll(filepath.Dir(socketPath), 0o755); err != nil {
		return nil, err
	}
	if err := os.RemoveAll(socketPath); err != nil {
		return nil, err
	}
	l, err := net.Listen("unix", socketPath)
	if err != nil {
		return nil, err
	}
	return l, nil
}
