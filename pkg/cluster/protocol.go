package cluster

import "github.com/xingjianluo/elastic-training-coordinator-go/pkg/sharding"

const (
	CodeOK                = "ok"
	CodeGenerationRestart = "generation_restart"
	CodeSlowWorker        = "slow_worker"
	CodeWorkerDead        = "worker_dead"
	CodeAlreadyCompleted  = "already_completed"
	CodeDuplicate         = "duplicate"
	CodeInvalidShardCount = "invalid_shard_count"
	CodeMisalignedShards  = "misaligned_shards"
	CodeInvalidRequest    = "invalid_request"
)

type Request struct {
	Seq        uint64         `json:"seq"`
	WorkerID   string         `json:"worker_id"`
	Type       string         `json:"type"`
	Generation int            `json:"generation"`
	Step       int            `json:"step"`
	Shard      sharding.Range `json:"shard"`
	Values     []float64      `json:"values,omitempty"`
	Dimension  int            `json:"dimension"`
	TotalSteps int            `json:"total_steps"`
}

type Response struct {
	ReqID      uint64        `json:"req_id"`
	Code       string        `json:"code"`
	Message    string        `json:"message,omitempty"`
	Generation int           `json:"generation"`
	Step       int           `json:"step"`
	Plan       sharding.Plan `json:"plan"`
	Values     []float64     `json:"values,omitempty"`
	Committed  bool          `json:"committed"`
	RunDone    bool          `json:"run_done"`
}
