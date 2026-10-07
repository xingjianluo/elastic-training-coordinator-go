package coordinator

import (
	"fmt"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
)

// ExitReason 如实说明一次进程运行是如何结束的。
type ExitReason string

const (
	// ExitRunning 尚未结束（仍在运行）。
	ExitRunning ExitReason = "running"
	// ExitCompleted 跑满目标步数后正常干净结束。
	ExitCompleted ExitReason = "completed"
	// ExitInterrupted 被外部叫停（stop 请求 / 终止信号 / ctx 取消），
	// 已在最后一个完整完成的状态落盘后干净退出。
	ExitInterrupted ExitReason = "interrupted"
)

// ExitStatus 是一次运行结束时可如实汇报的状态。
type ExitStatus struct {
	Reason          ExitReason
	Detail          string
	Step            int
	TargetSteps     int
	Era             uint64
	FinalCheckpoint string
	FinishedAt      time.Time
}

// defaultShutdownTimeout 是优雅停机的默认最长时间。
const defaultShutdownTimeout = 5 * time.Second

// ErrShuttingDown 停机流程开始后，新到达的非幂等工作请求一律被拒绝。
var ErrShuttingDown = errShuttingDown

// loadForResume 按配置选择恢复检查点：
//   - ResumeStep>=0：显式回滚到指定 step（该 step 下 era 最大者）；
//   - AutoRecover=true：最新点无效时自动采用更早的可用点，跳过原因如实返回；
//   - 默认（严格模式）：只认最新正式点，半截/损坏/版本不符直接返回可区分错误。
func (c *Coordinator) loadForResume() (st *checkpoint.State, path string, skipped []checkpoint.Skipped, err error) {
	if c.cfg.ResumeStep != nil {
		st, path, err = c.store.LoadStep(*c.cfg.ResumeStep)
		return st, path, nil, err
	}
	if c.cfg.AutoRecover {
		return c.store.LoadLatestValid()
	}
	// 严格模式：与历史行为一致，最新点是什么就校验什么，绝不静默换点/重来。
	p, err := c.store.Latest()
	if err != nil {
		return nil, "", nil, err
	}
	st, err = c.store.LoadPath(p)
	if err != nil {
		return nil, "", nil, err
	}
	return st, p, nil, nil
}

// afterRestoreLocked 记录恢复日志；显式回滚时删除“未来”检查点，使回滚点
// 重新成为目录中的最新点。构造期调用，此时协调者尚未对外服务。
func (c *Coordinator) afterRestoreLocked(path string, skipped []checkpoint.Skipped, st *checkpoint.State) {
	for _, sk := range skipped {
		c.log(EvCheckpoint, fmt.Sprintf("recovery skipped unusable checkpoint %s (reason=%v)",
			sk.Path, sk.Err))
	}
	c.log(EvCheckpoint, fmt.Sprintf("restored from checkpoint: era=%d step=%d members=%v path=%s",
		st.Era, st.Step, memberNames(st.Members), path))
	if c.cfg.ResumeStep != nil {
		removed, err := c.store.DeleteAfter(st.Step, st.Era)
		if err != nil {
			c.log(EvFailure, fmt.Sprintf("rollback cleanup of later checkpoints failed: %v", err))
		} else {
			c.log(EvLifecycle, fmt.Sprintf(
				"explicit rollback to step=%d: removed %d later checkpoint(s) %v",
				st.Step, len(removed), removed))
		}
	}
}

// Shutdown 触发优雅停机的【状态固化】阶段，可在任意时刻、从任意 goroutine
// 调用，幂等。它会：停止接收新工作 -> 作废尚未整体完成的在途步骤（绝不半提交）
// -> 在最后一个【完整完成】的状态上落最终检查点 -> 按保留上限清理旧点 ->
// 标记生命周期终结（Terminal 通道关闭）。整个过程不超过 timeout（<=0 用配置/
// 默认值）。
//
// 本方法只固化状态，不直接关闭本地 socket；传输资源的释放在 Serve 返回前统一
// 完成（完成路径可留短暂宽限让落后 worker 收到 Done），这样进程内直接调用时
// 不依赖 Serve 也能完成终结。
func (c *Coordinator) Shutdown(reason ExitReason, detail string, timeout time.Duration) {
	c.shutdownOnce.Do(func() {
		if timeout <= 0 {
			timeout = c.cfg.ShutdownTimeout
		}
		if timeout <= 0 {
			timeout = defaultShutdownTimeout
		}
		finalized := make(chan struct{})
		go func() {
			defer close(finalized)
			c.finalizeForShutdown(reason, detail)
		}()
		select {
		case <-finalized:
			c.markTerminal()
		case <-time.After(timeout):
			// 固化超过上限（典型为磁盘卡死）：不能让进程无限等下去。
			// 终结标记放后台，调用方据此进入资源释放并在兜底后退出进程。
			c.log(EvFailure, fmt.Sprintf(
				"graceful shutdown finalization did not finish within %s; releasing resources and exiting anyway",
				timeout))
			go c.markTerminal()
		}
	})
}

// finalizeForShutdown 在锁内完成停机前的状态固化。
func (c *Coordinator) finalizeForShutdown(reason ExitReason, detail string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stopping = true
	c.exitReason = reason
	if detail != "" {
		c.exitDetail = detail
	}
	if c.txn != nil {
		c.log(EvLifecycle, fmt.Sprintf(
			"shutdown (%s): in-flight step=%d era=%d is NOT complete => discarded, no partial commit",
			reason, c.txn.Step, c.txn.Era))
		c.txn = nil
	}
	c.log(EvLifecycle, fmt.Sprintf(
		"shutdown (%s) begins: no new work accepted; saving final checkpoint at last fully committed step=%d/%d era=%d",
		reason, c.step, c.cfg.Steps, c.era))
	// 总是在最后完整完成的状态上落一个最终检查点（哪怕刚跑满、或一步都还没跑）。
	path, err := c.saveCheckpointLocked("final checkpoint on graceful shutdown (" + string(reason) + ")")
	if err != nil {
		c.log(EvFailure, fmt.Sprintf("final checkpoint save failed: %v", err))
	} else {
		c.finalCkptPath = path
		c.finalCkptStep = c.step
	}
	c.pruneAndSweepLocked("graceful shutdown")
}

// markTerminal 标记生命周期终结（只生效一次）：汇总退出状态并写日志。
func (c *Coordinator) markTerminal() {
	c.terminalOnce.Do(func() {
		c.mu.Lock()
		if c.exitReason == ExitRunning {
			c.exitReason = ExitInterrupted
		}
		st := ExitStatus{
			Reason:          c.exitReason,
			Detail:          c.exitDetail,
			Step:            c.step,
			TargetSteps:     c.cfg.Steps,
			Era:             c.era,
			FinalCheckpoint: c.finalCkptPath,
			FinishedAt:      c.clock.Now(),
		}
		c.mu.Unlock()
		c.log(EvLifecycle, fmt.Sprintf(
			"coordinator TERMINATED reason=%s step=%d/%d era=%d finalCheckpoint=%s detail=%q",
			st.Reason, st.Step, st.TargetSteps, st.Era, st.FinalCheckpoint, st.Detail))
		close(c.terminalCh)
	})
}

// RequestStop 实现 stop RPC：登记一次外部停机请求，先回送回执，再异步优雅停机。
func (c *Coordinator) RequestStop(req proto.StopRequest) (proto.StopResponse, error) {
	c.mu.Lock()
	alreadyStopping := c.stopping
	step, era := c.step, c.era
	c.mu.Unlock()
	reason := req.Reason
	if reason == "" {
		reason = "stop request over local IPC"
	}
	if !alreadyStopping {
		c.log(EvLifecycle, "stop request received: "+reason)
		go c.Shutdown(ExitInterrupted, reason, c.cfg.ShutdownTimeout)
	}
	return proto.StopResponse{OK: true, Step: step, Era: era}, nil
}

// ExitStatus 返回当前/最终退出状态快照。
func (c *Coordinator) ExitStatus() ExitStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	return ExitStatus{
		Reason:          c.exitReason,
		Detail:          c.exitDetail,
		Step:            c.step,
		TargetSteps:     c.cfg.Steps,
		Era:             c.era,
		FinalCheckpoint: c.finalCkptPath,
		FinishedAt:      c.clock.Now(),
	}
}

// Terminal 返回生命周期终结（最终检查点已落、socket 已释放）后关闭的通道。
func (c *Coordinator) Terminal() <-chan struct{} { return c.terminalCh }

// Completed 返回训练跑满目标步数后关闭的通道（最终检查点此刻已同步落盘）。
func (c *Coordinator) Completed() <-chan struct{} { return c.completeCh }

// signalCompletedLocked 把退出原因记为“跑满”并广播完成信号（调用时持锁）。
func (c *Coordinator) signalCompletedLocked() {
	c.completeOnce.Do(func() {
		c.exitReason = ExitCompleted
		c.exitDetail = fmt.Sprintf("reached target %d steps", c.cfg.Steps)
		// 记下完成瞬间在册的成员：只要它们都通过 next 拿到 Done，就可以干净关 socket。
		c.completionMembers = map[string]struct{}{}
		for id := range c.members {
			c.completionMembers[id] = struct{}{}
		}
		close(c.completeCh)
		if len(c.completionMembers) == 0 {
			c.completionOnce.Do(func() { close(c.completionCh) })
		}
	})
}

// noteDeliveredDoneLocked 标记某成员已收到 Done；若完成集合已全部确认则广播（调用时持锁）。
func (c *Coordinator) noteDeliveredDoneLocked(id string) {
	if c.completionMembers == nil {
		return
	}
	if _, ok := c.completionMembers[id]; !ok {
		return
	}
	if m := c.members[id]; m != nil && !m.deliveredDone {
		m.deliveredDone = true
	}
	for id := range c.completionMembers {
		m := c.members[id]
		if m == nil || !m.deliveredDone {
			return
		}
	}
	c.completionOnce.Do(func() { close(c.completionCh) })
}

// CompletionConfirmed 返回“完成时在册的所有 worker 都已收到 Done”后关闭的通道。
func (c *Coordinator) CompletionConfirmed() <-chan struct{} { return c.completionCh }
