package coordinator

import (
	"errors"
	"fmt"

	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
)

// ErrShuttingDown 协调者正在优雅停机：不再接收任何新工作/新贡献。
var ErrShuttingDown = errors.New("coordinator: shutting down, not accepting new work")

// ExitReason 如实说明本次进程结束的原因。
type ExitReason string

const (
	// ExitCompleted 跑满目标步数后自然结束。
	ExitCompleted ExitReason = "completed"
	// ExitStopped 被外部叫停（停止请求/终止信号）后优雅结束。
	ExitStopped ExitReason = "stopped"
)

// ExitReport 是进程结束前给出的最终说明。
type ExitReport struct {
	Reason ExitReason
	// Step 退出时【已完整提交】的步数；在途未完成的一步绝不被计入。
	Step  int
	Steps int
	// CheckpointPath 收尾时落盘的最终检查点（下次启动从它恢复）。
	CheckpointPath string
}

func (r ExitReport) String() string {
	return fmt.Sprintf("reason=%s step=%d/%d checkpoint=%s", r.Reason, r.Step, r.Steps, r.CheckpointPath)
}

// RequestStop 请求优雅停机（幂等）：立即不再接收新工作；
// 真正的收尾（在最新完整状态上落检查点、释放资源）由 Serve 的退出路径完成。
func (c *Coordinator) RequestStop(requester string) proto.StopResponse {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.stopping {
		c.stopping = true
		inflight := -1
		if c.txn != nil {
			inflight = c.txn.Step
		}
		c.log(EvDecision, fmt.Sprintf(
			"stop requested by %s: no new work accepted; completed step=%d/%d; in-flight step=%v will NOT be committed (no partial commit)",
			requester, c.step, c.cfg.Steps, inflight))
	}
	return proto.StopResponse{OK: true, Era: c.era, Step: c.step}
}

// Stopping 返回是否已进入停机流程。
func (c *Coordinator) Stopping() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stopping
}

// finalizeLocked 收尾：在【已完整完成】的最新状态上落最终检查点并生成退出报告。
// 幂等；在途事务（若有）被丢弃，其部分贡献绝不写入检查点。调用时持锁。
func (c *Coordinator) finalizeLocked() {
	if c.exit != nil {
		return
	}
	reason := ExitStopped
	if c.done {
		reason = ExitCompleted
	}
	if c.txn != nil {
		c.log(EvDecision, fmt.Sprintf(
			"finalizing: discarding in-flight step=%d era=%d (%d/%d submits, %d/%d gathers) — only fully committed steps are checkpointed",
			c.txn.Step, c.txn.Era, len(c.txn.submitted), c.txn.WorldSize, len(c.txn.gathered), c.txn.WorldSize))
		c.txn = nil
	}
	path, err := c.saveCheckpointLocked("final: " + string(reason))
	if err != nil {
		// 落盘失败也要如实报告；路径留空。
		c.log(EvFailure, fmt.Sprintf("final checkpoint failed: %v", err))
	}
	rep := ExitReport{Reason: reason, Step: c.step, Steps: c.cfg.Steps, CheckpointPath: path}
	c.exit = &rep
	c.log(EvDecision, "exit report: "+rep.String())
}

// Shutdown 同步执行与 Serve 退出路径完全相同的收尾（进程内测试/嵌入场景用）。
// 幂等：重复调用返回同一份报告。
func (c *Coordinator) Shutdown() ExitReport {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stopping = true
	c.finalizeLocked()
	return *c.exit
}

// ExitReport 返回退出报告；尚未退出时 ok=false。
func (c *Coordinator) ExitReport() (ExitReport, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.exit == nil {
		return ExitReport{}, false
	}
	return *c.exit, true
}
