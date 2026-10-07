package proto

import (
	"encoding/json"

	"github.com/xingjianluo/elastic-training-coordinator-go/shard"
)

// 消息类型常量（本地 IPC 上 JSON 消息的 type 字段）。
const (
	MsgJoin       = "join"
	MsgHeartbeat  = "heartbeat"
	MsgLeave      = "leave"
	MsgNext       = "next"
	MsgSubmit     = "submit"
	MsgAllgather  = "allgather"
	MsgCheckpoint = "checkpoint"
	MsgStop       = "stop"
)

// Marshal / Unmarshal 把强类型负载编进 transport.Envelope 的原始 JSON。
func Marshal(msgType string, payload any) ([]byte, error) {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return b, nil
}

// Membership 是一次成员快照。
type Membership struct {
	Era     uint64   `json:"era"`
	Step    int      `json:"step"`
	Members []Member `json:"members"`
}

// Member 描述一个 worker。
type Member struct {
	ID   string `json:"id"`
	Rank int    `json:"rank"`
}

// StepAssignment 描述一个训练步骤的分工。
type StepAssignment struct {
	Era        uint64    `json:"era"`
	Step       int       `json:"step"`
	Params     []float64 `json:"params"`
	WorldSize  int       `json:"world_size"`
	Rank       int       `json:"rank"`
	ShardCount int       `json:"shard_count"`
	// ParamRange 该 worker 本步负责的参数分片区间。
	ParamRange  shard.Interval `json:"param_range"`
	SampleRange shard.Interval `json:"sample_range"`
	Samples     []float64      `json:"samples"` // 展平的样本，每行 NIn+1
	NIn         int            `json:"n_in"`
	BatchSize   int            `json:"batch_size"` // 整个 batch 的样本数（梯度归一化用）
	LR          float64        `json:"lr"`
	Momentum    float64        `json:"momentum"`
}

// SubmitRequest 是 reduce-scatter 阶段提交。
// Chunk 为该 worker 算出的【完整局部梯度向量】（长度=参数维度）；
// Range 表示该 rank 负责（最终归约并更新）的参数分片区间。
type SubmitRequest struct {
	WorkerID  string         `json:"worker_id"`
	Era       uint64         `json:"era"`
	Step      int            `json:"step"`
	Rank      int            `json:"rank"`
	VectorLen int            `json:"vector_len"`
	Range     shard.Interval `json:"range"`
	Chunk     []float64      `json:"chunk"`
}

// SubmitResponse 是 reduce-scatter 完成后该 worker 的分片结果。
type SubmitResponse struct {
	Era           uint64         `json:"era"`
	Step          int            `json:"step"`
	Range         shard.Interval `json:"range"`
	Chunk         []float64      `json:"chunk"`          // canonical 求和后的梯度分片
	VelocityChunk []float64      `json:"velocity_chunk"` // 该分片当前优化器速度
	Ready         bool           `json:"ready"`          // 是否已集齐全部贡献完成 reduce
	Duplicate     bool           `json:"duplicate"`
}

// AllgatherRequest 是 allgather 阶段提交。
type AllgatherRequest struct {
	WorkerID      string         `json:"worker_id"`
	Era           uint64         `json:"era"`
	Step          int            `json:"step"`
	Rank          int            `json:"rank"`
	Range         shard.Interval `json:"range"`
	Chunk         []float64      `json:"chunk"` // 更新后的参数分片
	VelocityChunk []float64      `json:"velocity_chunk"`
}

// AllgatherResponse 返回当前全量参数、全量优化器速度与本步是否已提交。
type AllgatherResponse struct {
	Era       uint64    `json:"era"`
	Step      int       `json:"step"`
	Committed bool      `json:"committed"`
	Params    []float64 `json:"params"`
	Velocity  []float64 `json:"velocity"`
}

// JoinRequest / JoinResponse
type JoinRequest struct {
	WorkerID string `json:"worker_id"`
}

type JoinResponse struct {
	OK         bool       `json:"ok"`
	Membership Membership `json:"membership"`
}

// HeartbeatRequest / HeartbeatResponse
type HeartbeatRequest struct {
	WorkerID string `json:"worker_id"`
}

type HeartbeatResponse struct {
	OK bool `json:"ok"`
}

// LeaveRequest
type LeaveRequest struct {
	WorkerID string `json:"worker_id"`
}

// NextRequest / NextResponse
type NextRequest struct {
	WorkerID string `json:"worker_id"`
}

type NextResponse struct {
	// Done 表示全部训练步骤已完成。
	Done bool `json:"done"`
	// Stopped 表示协调者正在优雅停机，不再下发新工作。
	Stopped    bool            `json:"stopped,omitempty"`
	Assignment *StepAssignment `json:"assignment,omitempty"`
}

// CheckpointRequest / Response
type CheckpointRequest struct {
	WorkerID string `json:"worker_id"`
}

type CheckpointResponse struct {
	Era  uint64 `json:"era"`
	Step int    `json:"step"`
	Path string `json:"path"`
	OK   bool   `json:"ok"`
}

// StopRequest 请求协调者优雅停机（幂等）。
type StopRequest struct {
	Requester string `json:"requester"`
}

// StopResponse 确认已进入停机流程；Step 为当时已完整提交的步数。
type StopResponse struct {
	OK   bool   `json:"ok"`
	Era  uint64 `json:"era"`
	Step int    `json:"step"`
}
