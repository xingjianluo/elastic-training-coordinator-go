package coordinator

import (
	"fmt"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/nn"
	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
)

// saveCheckpointLocked 原子落盘当前参数/优化器状态/成员信息。调用时持锁。
func (c *Coordinator) saveCheckpointLocked(reason string) (string, error) {
	st := c.snapshotLocked()
	path := c.store.FinalPath(st)
	if err := c.store.Save(st); err != nil {
		c.log(EvFailure, fmt.Sprintf("checkpoint save failed (%s): %v", reason, err))
		return "", err
	}
	c.log(EvCheckpoint, fmt.Sprintf(
		"checkpoint saved (%s): era=%d step=%d members=%v params=%d path=%s (atomic tmp+rename)",
		reason, st.Era, st.Step, c.activeIDsLocked(), len(st.Params), path))
	return path, nil
}

// RequestCheckpoint 触发一次显式检查点落盘。
func (c *Coordinator) RequestCheckpoint(workerID string) (proto.CheckpointResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.members[workerID]; !ok {
		return proto.CheckpointResponse{}, ErrUnknownWorker
	}
	path, err := c.saveCheckpointLocked("explicit request by " + workerID)
	if err != nil {
		return proto.CheckpointResponse{OK: false}, err
	}
	return proto.CheckpointResponse{Era: c.era, Step: c.step, Path: path, OK: true}, nil
}

// Store 暴露检查点存储（测试故障演练用）。
func (c *Coordinator) Store() *checkpoint.Store { return c.store }

// Baseline 返回与当前配置一致的单机基准，供测试/校验做数值对齐。
func (c *Coordinator) Baseline() *nn.BaselineResult {
	return nn.RunBaseline(nn.BaselineConfig{
		NSamples: c.cfg.NSamples,
		NIn:      c.cfg.NIn,
		Seed:     c.cfg.Seed,
		Steps:    c.cfg.Steps,
		Batch:    c.cfg.Batch,
		LR:       c.cfg.LR,
		Momentum: c.cfg.Momentum,
	})
}
