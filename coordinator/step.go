package coordinator

import (
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/nn"
	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
	"github.com/xingjianluo/elastic-training-coordinator-go/shard"
)

// openStepLocked 惰性打开下一个待执行步骤，并记录规划判定依据。
func (c *Coordinator) openStepLocked() *stepTxn {
	if c.txn != nil || c.done {
		return c.txn
	}
	ids := c.activeIDsLocked()
	world := len(ids)
	if world == 0 {
		return nil
	}
	paramPlan, err := shard.NewPlan(c.cfg.NIn+1, world)
	if err != nil {
		// world<=参数维度时不会发生；防御性记录。
		c.log(EvFailure, fmt.Sprintf("open step %d rejected: %v", c.step, err))
		return nil
	}
	samplePlan, err := shard.NewPlan(c.cfg.Batch, world)
	if err != nil {
		c.log(EvFailure, fmt.Sprintf("open step %d rejected: %v", c.step, err))
		return nil
	}
	sr := make([]interval, world)
	for r := 0; r < world; r++ {
		siv := samplePlan.IntervalOf(r)
		sr[r] = interval{Start: siv.Start, End: siv.End}
	}
	t := &stepTxn{
		Era:          c.era,
		Step:         c.step,
		OpenedAt:     c.clock.Now(),
		WorldSize:    world,
		RankToID:     append([]string(nil), ids...),
		SampleRanges: sr,
		submitted:    map[int][]float64{},
		reduced:      map[int][]float64{},
		gathered:     map[int][]float64{},
		gatheredVel:  map[int][]float64{},
		submittedBy:  map[int]bool{},
		gatheredBy:   map[int]bool{},
	}
	c.txn = t
	ranges := make([]string, world)
	pranges := make([]string, world)
	for r := 0; r < world; r++ {
		ranges[r] = fmt.Sprintf("rank%d=%s", r, sr[r])
		pranges[r] = fmt.Sprintf("rank%d=%s", r, paramPlan.IntervalOf(r))
	}
	off := (c.step * c.cfg.Batch) % c.cfg.NSamples
	c.log(EvPlan, fmt.Sprintf(
		"open step=%d era=%d world=%d; batch sample offset=%d; sample shards {%v}; param shards {%v}",
		c.step, c.era, world, off, joinComma(ranges), joinComma(pranges)))
	return c.txn
}

func joinComma(xs []string) string {
	out := ""
	for i, x := range xs {
		if i > 0 {
			out += ", "
		}
		out += x
	}
	return out
}

// rankOfIDLocked 返回 worker 在当前 era 的 canonical rank；不在则 -1。
func (t *stepTxn) rankOfID(id string) int {
	for r, x := range t.RankToID {
		if x == id {
			return r
		}
	}
	return -1
}

// Next 返回当前待执行步骤分工。重复调用幂等：在途步骤期间返回同一份分工。
func (c *Coordinator) Next(req proto.NextRequest) proto.NextResponse {
	c.mu.Lock()
	defer c.mu.Unlock()

	m, ok := c.members[req.WorkerID]
	if !ok {
		c.log(EvFailure, fmt.Sprintf("next from unknown worker %q rejected", req.WorkerID))
		return proto.NextResponse{}
	}
	m.LastSeen = c.clock.Now()

	if c.done {
		c.log(EvDecision, fmt.Sprintf("worker %q asked next: training complete (no further work)", req.WorkerID))
		return proto.NextResponse{Done: true}
	}
	t := c.openStepLocked()
	if t == nil {
		c.log(EvDecision, "no active members; cannot open step")
		return proto.NextResponse{}
	}
	rank := t.rankOfID(req.WorkerID)

	// 只把该 rank 的样本数据放进分配（参数仍给全量，worker 需要算完整局部梯度）。
	off := (t.Step * c.cfg.Batch) % c.cfg.NSamples
	sr := t.SampleRanges[rank]
	flat := make([]float64, 0, (sr.End-sr.Start)*(c.cfg.NIn+1))
	for k := sr.Start; k < sr.End; k++ {
		s := c.ds.Samples[(off+k)%c.cfg.NSamples]
		flat = append(flat, s.X...)
		flat = append(flat, s.Y)
	}
	plan, _ := shard.NewPlan(c.cfg.NIn+1, t.WorldSize)
	a := &proto.StepAssignment{
		Era:         t.Era,
		Step:        t.Step,
		Params:      append([]float64(nil), c.params...),
		WorldSize:   t.WorldSize,
		Rank:        rank,
		ShardCount:  t.WorldSize,
		ParamRange:  plan.IntervalOf(rank),
		SampleRange: shard.Interval{Start: sr.Start, End: sr.End},
		Samples:     flat,
		NIn:         c.cfg.NIn,
		BatchSize:   c.cfg.Batch,
		LR:          c.cfg.LR,
		Momentum:    c.cfg.Momentum,
	}
	c.log(EvDecision, fmt.Sprintf(
		"assign step=%d era=%d to worker=%q rank=%d sampleShard=%s paramShard=%s",
		t.Step, t.Era, req.WorkerID, rank, sr, plan.IntervalOf(rank)))
	return proto.NextResponse{Assignment: a}
}

// checkContrib 校验一条贡献消息的通用前置条件，返回 rank 与在途事务。
func (c *Coordinator) checkContrib(workerID string, era uint64, step int) (*stepTxn, int, error) {
	m, ok := c.members[workerID]
	if !ok {
		return nil, 0, ErrUnknownWorker
	}
	m.LastSeen = c.clock.Now()
	if c.done {
		return nil, 0, ErrTrainingComplete
	}
	if c.txn == nil || era != c.era || step != c.step {
		// 成员变动推进 era 后，旧 era 的在途贡献一律拒绝（整步重来，不部分提交）。
		return nil, 0, fmt.Errorf("%w: got era=%d step=%d, current era=%d step=%d",
			ErrStaleEra, era, step, c.era, c.step)
	}
	rank := c.txn.rankOfID(workerID)
	if rank < 0 {
		return nil, 0, ErrUnknownWorker
	}
	return c.txn, rank, nil
}

// Submit 接收 reduce-scatter 阶段的局部分片。
// 同一 worker 重发（完成后重连/网络重试）时幂等返回此前结果，绝不重复累加。
func (c *Coordinator) Submit(req proto.SubmitRequest) (proto.SubmitResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	t, rank, err := c.checkContrib(req.WorkerID, req.Era, req.Step)
	if err != nil {
		if errors.Is(err, ErrStaleEra) {
			c.log(EvFailure, fmt.Sprintf("submit rejected stale: worker=%q era=%d step=%d",
				req.WorkerID, req.Era, req.Step))
		}
		return proto.SubmitResponse{}, err
	}
	// 区间必须和协调者的分片规划逐片对齐，否则明确拒绝并给出原因。
	plan, _ := shard.NewPlan(c.cfg.NIn+1, t.WorldSize)
	if err := plan.ValidateRange(rank, req.Range); err != nil {
		c.log(EvFailure, fmt.Sprintf("submit misaligned: worker=%q rank=%d got=%s: %v",
			req.WorkerID, rank, req.Range, err))
		return proto.SubmitResponse{}, fmt.Errorf("%w: %v", ErrShardMisaligned, err)
	}
	// reduce-scatter：每个 worker 必须提交完整局部梯度向量。
	if req.VectorLen != plan.N || len(req.Chunk) != plan.N {
		return proto.SubmitResponse{}, fmt.Errorf(
			"%w: worker=%q full-gradient len=%d vectorLen=%d, want full len=%d",
			ErrShardMisaligned, req.WorkerID, len(req.Chunk), req.VectorLen, plan.N)
	}

	if t.submittedBy[rank] {
		c.log(EvContribution, fmt.Sprintf(
			"DEDUP submit step=%d era=%d worker=%q rank=%d: duplicate contribution ignored, returning cached result",
			req.Step, req.Era, req.WorkerID, rank))
		ready := len(t.reduced) == t.WorldSize
		resp := proto.SubmitResponse{
			Era:       t.Era,
			Step:      t.Step,
			Range:     req.Range,
			Ready:     ready,
			Duplicate: true,
		}
		if ready {
			resp.Chunk = append([]float64(nil), t.reduced[rank]...)
			resp.VelocityChunk = append([]float64(nil), c.vel[req.Range.Start:req.Range.End]...)
		} else {
			resp.Chunk = append([]float64(nil), t.submitted[rank]...)
		}
		return resp, nil
	}

	chunk := append([]float64(nil), req.Chunk...)
	t.submitted[rank] = chunk
	t.submittedBy[rank] = true
	c.log(EvContribution, fmt.Sprintf(
		"accept submit step=%d era=%d rank=%d range=%s (%d/%d workers submitted)",
		t.Step, t.Era, rank, req.Range, len(t.submitted), t.WorldSize))

	// 集齐后做 canonical（rank 升序）reduce-scatter：
	// 先把各 worker 的完整局部梯度按 rank 0..world-1 顺序累加成全局梯度
	// （canonical 顺序保证位级结果与到达顺序无关），再按规划切成各 rank 的片。
	if len(t.submitted) == t.WorldSize {
		full := make([]float64, plan.N)
		for k := 0; k < t.WorldSize; k++ {
			seg := t.submitted[k]
			for i := 0; i < plan.N; i++ {
				full[i] += seg[i]
			}
		}
		for r := 0; r < t.WorldSize; r++ {
			iv := plan.IntervalOf(r)
			t.reduced[r] = append([]float64(nil), full[iv.Start:iv.End]...)
		}
		c.log(EvPlan, fmt.Sprintf(
			"reduce-scatter step=%d complete in canonical rank order; reduced shard lengths=%v",
			t.Step, shardLengths(plan, t.WorldSize)))
	}

	// 未集齐时回传原始分片与 Ready=false，worker 轮询等待其余成员。
	reduced, has := t.reduced[rank]
	resp := proto.SubmitResponse{Era: t.Era, Step: t.Step, Range: req.Range, Ready: has}
	if has {
		resp.Chunk = reduced
		resp.VelocityChunk = append([]float64(nil), c.vel[req.Range.Start:req.Range.End]...)
	} else {
		resp.Chunk = chunk
	}
	return resp, nil
}

func shardLengths(plan *shard.Plan, world int) []int {
	out := make([]int, world)
	for r := 0; r < world; r++ {
		out[r] = plan.IntervalOf(r).Len()
	}
	return out
}

// Allgather 接收优化器更新后的分片，集齐后 canonical 拼接、提交步骤并推进。
// 重复提交同样幂等：已提交过的 rank 重发不会被第二次计入。
func (c *Coordinator) Allgather(req proto.AllgatherRequest) (proto.AllgatherResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	t, rank, err := c.checkContrib(req.WorkerID, req.Era, req.Step)
	if err != nil {
		return proto.AllgatherResponse{}, err
	}
	plan, _ := shard.NewPlan(c.cfg.NIn+1, t.WorldSize)
	if err := plan.ValidateRange(rank, req.Range); err != nil {
		c.log(EvFailure, fmt.Sprintf("allgather misaligned: worker=%q rank=%d got=%s: %v",
			req.WorkerID, rank, req.Range, err))
		return proto.AllgatherResponse{}, fmt.Errorf("%w: %v", ErrShardMisaligned, err)
	}
	if len(req.Chunk) != req.Range.Len() {
		return proto.AllgatherResponse{}, fmt.Errorf("%w: chunk len=%d but range %s needs %d",
			ErrShardMisaligned, len(req.Chunk), req.Range, req.Range.Len())
	}
	if len(req.VelocityChunk) != req.Range.Len() {
		return proto.AllgatherResponse{}, fmt.Errorf("%w: velocity chunk len=%d but range %s needs %d",
			ErrShardMisaligned, len(req.VelocityChunk), req.Range, req.Range.Len())
	}

	committed := false
	if t.gatheredBy[rank] {
		// 幂等：完成本阶段后重连/重发，直接返回当前已知结果，不重复计入。
		c.log(EvContribution, fmt.Sprintf(
			"DEDUP allgather step=%d worker=%q rank=%d: duplicate ignored", t.Step, req.WorkerID, rank))
	} else {
		t.gathered[rank] = append([]float64(nil), req.Chunk...)
		t.gatheredVel[rank] = append([]float64(nil), req.VelocityChunk...)
		t.gatheredBy[rank] = true
	}

	if len(t.gathered) == t.WorldSize && !c.done {
		full := make([]float64, c.cfg.NIn+1)
		fullVel := make([]float64, c.cfg.NIn+1)
		for r := 0; r < t.WorldSize; r++ {
			iv := plan.IntervalOf(r)
			copy(full[iv.Start:iv.End], t.gathered[r])
			copy(fullVel[iv.Start:iv.End], t.gatheredVel[r])
		}
		c.params = full
		c.vel = fullVel
		c.step++
		c.txn = nil
		committed = true
		model := nn.Model{Params: c.params, NIn: c.cfg.NIn}
		loss := model.Loss(c.ds)
		c.log(EvDecision, fmt.Sprintf(
			"allgather step=%d complete in canonical rank order => COMMIT; step now=%d/%d loss=%.10f",
			req.Step, c.step, c.cfg.Steps, loss))
		if c.step >= c.cfg.Steps {
			c.done = true
			c.log(EvDecision, fmt.Sprintf("training reached %d steps => DONE", c.cfg.Steps))
		}
		if c.cfg.CheckpointEvery > 0 && c.step%c.cfg.CheckpointEvery == 0 {
			c.saveCheckpointLocked("periodic")
		}
	}

	resp := proto.AllgatherResponse{
		Era: t.Era, Step: t.Step, Committed: committed,
		Params:   append([]float64(nil), c.params...),
		Velocity: append([]float64(nil), c.vel...),
	}
	return resp, nil
}

// SlowWorker 描述一个被判定为掉队（慢但还活着）的 worker。
type SlowWorker struct {
	ID     string
	Rank   int
	Reason string
}

// CheckTimeouts 在给定时刻做一次超时判定，供故障检测器/确定性测试调用：
//   - 心跳超过 HeartbeatTimeout => 死亡：移出成员、推进 era、在途步骤作废重来；
//   - 仍有心跳但当前步骤超过 StepTimeout 未完成 => 掉队（straggler）：
//     不删成员、不丢步骤，只在事件中说明，等待其追上。
//
// 两者返回值与日志原因可明确区分。dead 返回被判定死亡并移除的 worker。
func (c *Coordinator) CheckTimeouts(now time.Time) (slow []SlowWorker, dead []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// 先判死：心跳超时即认为真挂了。
	for id, m := range c.members {
		if now.Sub(m.LastSeen) > c.cfg.HeartbeatTimeout {
			dead = append(dead, id)
		}
	}
	sort.Strings(dead)
	for _, id := range dead {
		ls := c.members[id].LastSeen
		delete(c.members, id)
		c.log(EvFailure, fmt.Sprintf(
			"timeout classification: worker=%q DEAD (no heartbeat for %s > %s, last seen %s)",
			id, now.Sub(ls), c.cfg.HeartbeatTimeout, ls.Format(time.RFC3339Nano)))
		c.advanceEraLocked(fmt.Sprintf("worker %q heartbeat timeout => DEAD", id))
	}
	// 再判慢：在途步骤等待过久，但成员心跳仍新鲜 => 掉队而非死亡。
	if c.txn != nil {
		for r, id := range c.txn.RankToID {
			m, alive := c.members[id]
			if !alive {
				continue
			}
			// 该 rank 两阶段都完成的不算慢。
			if c.txn.gatheredBy[r] {
				continue
			}
			wait := now.Sub(c.txn.OpenedAt)
			if wait > c.cfg.StepTimeout && now.Sub(m.LastSeen) <= c.cfg.HeartbeatTimeout {
				slow = append(slow, SlowWorker{
					ID: id, Rank: r,
					Reason: fmt.Sprintf("straggler: step open for %s > %s but heartbeat fresh (age %s <= %s) => SLOW not DEAD",
						wait, c.cfg.StepTimeout, now.Sub(m.LastSeen), c.cfg.HeartbeatTimeout),
				})
			}
		}
		sort.Slice(slow, func(i, j int) bool { return slow[i].ID < slow[j].ID })
		for _, s := range slow {
			c.log(EvDecision, "timeout classification: worker="+s.ID+" "+s.Reason)
		}
	}
	return slow, dead
}
