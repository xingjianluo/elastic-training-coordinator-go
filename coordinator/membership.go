package coordinator

import (
	"fmt"
	"sort"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
)

func (c *Coordinator) log(kind EventKind, detail string) {
	ev := Event{Time: c.clock.Now(), Kind: kind, Detail: detail}
	select {
	case c.events <- ev:
	default: // 事件通道满时丢弃日志，绝不阻塞训练
	}
}

// Events 返回判定日志事件通道。
func (c *Coordinator) Events() <-chan Event { return c.events }

// Membership 返回当前成员快照（不改变任何状态）。
func (c *Coordinator) Membership() proto.Membership {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.membershipLocked()
}

func (c *Coordinator) membershipLocked() proto.Membership {
	ids := c.activeIDsLocked()
	ms := make([]proto.Member, len(ids))
	for i, id := range ids {
		ms[i] = proto.Member{ID: id, Rank: i}
	}
	return proto.Membership{Era: c.era, Step: c.step, Members: ms}
}

// Join 处理 worker 加入。
// 语义：
//   - 全新 ID：登记为成员；若当前正有在途步骤，则进入新 era，当前步骤作废重开，
//     保证该 worker 从下一步/重开步骤起参与，绝不丢梯度也不重复累加。
//   - 已存在 ID（断线重连，或训练完成后的重连）：幂等返回当前成员快照，
//     不产生第二次贡献，不推进 era。
func (c *Coordinator) Join(req proto.JoinRequest) proto.JoinResponse {
	c.mu.Lock()
	defer c.mu.Unlock()

	if req.WorkerID == "" {
		c.log(EvFailure, "join rejected: empty worker id")
		return proto.JoinResponse{OK: false, Membership: c.membershipLocked()}
	}
	now := c.clock.Now()
	if m, ok := c.members[req.WorkerID]; ok {
		m.LastSeen = now
		c.log(EvMembership, fmt.Sprintf(
			"worker %q reconnected (idempotent join); era=%d step=%d members=%v",
			req.WorkerID, c.era, c.step, c.activeIDsLocked()))
		return proto.JoinResponse{OK: true, Membership: c.membershipLocked()}
	}
	if c.stopping {
		c.log(EvDecision, fmt.Sprintf("join from new worker %q rejected: coordinator is shutting down", req.WorkerID))
		return proto.JoinResponse{OK: false, Membership: c.membershipLocked()}
	}

	c.members[req.WorkerID] = &memberState{ID: req.WorkerID, JoinEra: c.era + 1, LastSeen: now}
	c.advanceEraLocked(fmt.Sprintf("worker %q joined", req.WorkerID))
	return proto.JoinResponse{OK: true, Membership: c.membershipLocked()}
}

// Leave 处理 worker 主动退出。若步骤在途则推进 era 并重开。
func (c *Coordinator) Leave(workerID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.members[workerID]; !ok {
		return
	}
	delete(c.members, workerID)
	c.advanceEraLocked(fmt.Sprintf("worker %q left", workerID))
}

// MarkDead 供故障检测器/测试把某个 worker 标记为死亡（心跳超时）。
func (c *Coordinator) MarkDead(workerID, reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.members[workerID]; !ok {
		return
	}
	delete(c.members, workerID)
	c.advanceEraLocked(fmt.Sprintf("worker %q removed as dead: %s", workerID, reason))
}

// advanceEraLocked 使旧在途步骤失效并进入下一个 era。
// 旧 era 收到的任何贡献都会被 ErrStaleEra 拒绝，旧贡献全部作废、不会部分提交，
// 新 era 以当前参数重新开始同一步骤，因此既不丢梯度（整步重来）也不重复累加。
func (c *Coordinator) advanceEraLocked(reason string) {
	c.era++
	if c.txn != nil {
		c.log(EvDecision, fmt.Sprintf(
			"era advanced -> %d due to %s; in-flight step %d ABORTED and will be retried (no partial commit, no double count)",
			c.era, reason, c.txn.Step))
		c.txn = nil
	} else {
		c.log(EvMembership, fmt.Sprintf("era advanced -> %d due to %s (no in-flight step)", c.era, reason))
	}
	ids := c.activeIDsLocked()
	ranks := make([]proto.Member, len(ids))
	for i, id := range ids {
		ranks[i] = proto.Member{ID: id, Rank: i}
	}
	c.log(EvMembership, fmt.Sprintf("membership of era %d: %v (rank by sorted worker id)", c.era, ids))
}

// heartbeat 刷新 worker 的最后可见时间。
func (c *Coordinator) heartbeat(workerID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	m, ok := c.members[workerID]
	if !ok {
		return ErrUnknownWorker
	}
	m.LastSeen = c.clock.Now()
	return nil
}

// ReapByClock 按心跳超时清理成员，返回被清理的 worker。
// 调用方（故障检测器）用它区分“慢”和“死”：
// 这里只处理真的心跳超时（死），慢节点由步骤级超时单独判定，不删成员。
func (c *Coordinator) ReapByClock(now time.Time) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var dead []string
	for id, m := range c.members {
		if now.Sub(m.LastSeen) > c.cfg.HeartbeatTimeout {
			dead = append(dead, id)
		}
	}
	sort.Strings(dead)
	for _, id := range dead {
		last := c.members[id].LastSeen
		delete(c.members, id)
		c.advanceEraLocked(fmt.Sprintf(
			"worker %q heartbeat timed out after %s (last seen %s) => DEAD",
			id, c.cfg.HeartbeatTimeout, last.Format(time.RFC3339Nano)))
	}
	return dead
}

// snapshotLocked 构造检查点状态（调用时持锁）。
func (c *Coordinator) snapshotLocked() *checkpoint.State {
	ids := c.activeIDsLocked()
	ms := make([]checkpoint.Member, len(ids))
	for i, id := range ids {
		ms[i] = checkpoint.Member{ID: id, Rank: i}
	}
	return &checkpoint.State{
		Version:   checkpoint.FormatVersion,
		Era:       c.era,
		Step:      c.step,
		Params:    append([]float64(nil), c.params...),
		Velocity:  append([]float64(nil), c.vel...),
		Members:   ms,
		UpdatedAt: c.clock.Now(),
	}
}

// SnapshotState 导出当前检查点状态。
func (c *Coordinator) SnapshotState() *checkpoint.State {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.snapshotLocked()
}

// Params 返回当前参数的拷贝。
func (c *Coordinator) Params() []float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]float64(nil), c.params...)
}

// Step 返回已完成步数。
func (c *Coordinator) Step() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.step
}

// Era 返回当前时代号。
func (c *Coordinator) Era() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.era
}

// Done 返回训练是否全部完成。
func (c *Coordinator) Done() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.done
}
