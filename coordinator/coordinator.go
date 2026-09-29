package coordinator

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/checkpoint"
	"github.com/xingjianluo/elastic-training-coordinator-go/nn"
	"github.com/xingjianluo/elastic-training-coordinator-go/transport"
)

// 协调者侧可区分的失败原因。
var (
	// ErrUnknownWorker 发消息的 worker 没加入过当前集群。
	ErrUnknownWorker = errors.New("coordinator: unknown worker, must join first")
	// ErrStaleEra worker 消息来自旧时代（成员变动后未重新领取步骤）。
	ErrStaleEra = errors.New("coordinator: message from a stale era")
	// ErrDuplicateContrib 同一 (era,step,rank) 的重复贡献（幂等返回缓存，不重复累加）。
	ErrDuplicateContrib = errors.New("coordinator: duplicate contribution for this step")
	// ErrStepNotOpen 当前没有对该 worker 开放的步骤。
	ErrStepNotOpen = errors.New("coordinator: step is not open for contributions")
	// ErrShardMisaligned worker 提交的分片区间与协调者规划不一致。
	ErrShardMisaligned = errors.New("coordinator: shard range misaligned with plan")
	// ErrTrainingComplete 全部步骤已完成。
	ErrTrainingComplete = errors.New("coordinator: training already complete")
	// ErrNoActiveMembers 当前没有任何活跃 worker。
	ErrNoActiveMembers = errors.New("coordinator: no active members")
)

// Clock 抽象时间源，便于测试中确定性地推进超时判定。
type Clock interface {
	Now() time.Time
}

type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

// EventKind 是判定日志事件类型。
type EventKind string

const (
	EvMembership   EventKind = "membership"
	EvPlan         EventKind = "plan"
	EvDecision     EventKind = "decision"
	EvContribution EventKind = "contribution"
	EvCheckpoint   EventKind = "checkpoint"
	EvFailure      EventKind = "failure"
)

// Event 是一步判定依据的结构化日志。
type Event struct {
	Time   time.Time
	Kind   EventKind
	Detail string
}

// Config 协调者配置。
type Config struct {
	SocketPath       string
	CheckpointDir    string
	NSamples         int
	NIn              int
	Seed             int64
	Steps            int
	Batch            int
	LR               float64
	Momentum         float64
	HeartbeatTimeout time.Duration
	StepTimeout      time.Duration
	CheckpointEvery  int
}

// memberState 是一个 worker 的运行时状态。
type memberState struct {
	ID       string
	JoinEra  uint64
	LastSeen time.Time
}

// stepTxn 是当前开放步骤的在途台账。
type stepTxn struct {
	Era       uint64
	Step      int
	OpenedAt  time.Time
	WorldSize int
	// rank -> workerID（rank 按成员 ID 排序 canonical 分配）
	RankToID []string
	// rank -> 该 rank 负责的数据样本区间
	SampleRanges []interval
	// reduce-scatter 阶段：rank -> 原始局部分片
	submitted map[int][]float64
	// reduce-scatter canonical 求和结果：rank -> 分片
	reduced map[int][]float64
	// allgather 阶段：rank -> 更新后分片
	gathered map[int][]float64
	// allgather 阶段：rank -> 更新后的优化器速度分片
	gatheredVel map[int][]float64
	// 标记已提交过的 (worker)，用于重连幂等
	submittedBy map[int]bool
	gatheredBy  map[int]bool
}

type interval struct{ Start, End int }

func (i interval) String() string { return fmt.Sprintf("[%d,%d)", i.Start, i.End) }

// Coordinator 弹性训练协调者。
type Coordinator struct {
	cfg Config

	mu      sync.Mutex
	params  []float64
	vel     []float64
	era     uint64
	step    int // 已完成步数
	ds      *nn.Dataset
	members map[string]*memberState
	txn     *stepTxn
	done    bool

	store  *checkpoint.Store
	server *transport.Server
	events chan Event
	clock  Clock
}

// New 创建协调者并从最新检查点恢复（若存在）。
// 全新启动时：无检查点 -> 零参数；只有半截/损坏检查点 -> 返回可区分错误，
// 不会静默当成全新启动，也不会当成有效检查点继续。
func New(cfg Config) (*Coordinator, error) {
	c := &Coordinator{
		cfg:     cfg,
		ds:      nn.NewDataset(cfg.NSamples, cfg.NIn, cfg.Seed),
		members: map[string]*memberState{},
		store:   checkpoint.NewStore(cfg.CheckpointDir),
		events:  make(chan Event, 1024),
		clock:   realClock{},
	}
	c.params = make([]float64, cfg.NIn+1)
	c.vel = make([]float64, cfg.NIn+1)

	st, err := c.store.Load()
	if err != nil {
		if errors.Is(err, checkpoint.ErrMissing) {
			return c, nil // 全新启动
		}
		return nil, err // 半截/坏魔数/版本不符/损坏：明确拒绝
	}
	if err := c.restoreLocked(st); err != nil {
		return nil, err
	}
	c.log(EvCheckpoint, fmt.Sprintf("restored from checkpoint: era=%d step=%d members=%v",
		st.Era, st.Step, memberNames(st.Members)))
	return c, nil
}

// NewNoResume 创建协调者且忽略已有检查点（测试用）。
func NewNoResume(cfg Config) (*Coordinator, error) {
	c := &Coordinator{
		cfg:     cfg,
		ds:      nn.NewDataset(cfg.NSamples, cfg.NIn, cfg.Seed),
		members: map[string]*memberState{},
		store:   checkpoint.NewStore(cfg.CheckpointDir),
		events:  make(chan Event, 1024),
		clock:   realClock{},
	}
	c.params = make([]float64, cfg.NIn+1)
	c.vel = make([]float64, cfg.NIn+1)
	return c, nil
}

// SetClock 注入时间源（测试用，须在启动服务前调用）。
func (c *Coordinator) SetClock(cl Clock) {
	c.mu.Lock()
	c.clock = cl
	c.mu.Unlock()
}

func (c *Coordinator) restoreLocked(st *checkpoint.State) error {
	if st.Version != checkpoint.FormatVersion {
		return fmt.Errorf("%w: state version=%d", checkpoint.ErrVersionMismatch, st.Version)
	}
	if len(st.Params) != c.cfg.NIn+1 || len(st.Velocity) != c.cfg.NIn+1 {
		return fmt.Errorf("%w: checkpoint params dim=%d, want %d",
			checkpoint.ErrCorrupt, len(st.Params), c.cfg.NIn+1)
	}
	if st.Step < 0 || st.Step > c.cfg.Steps {
		return fmt.Errorf("%w: step %d out of range [0,%d]", checkpoint.ErrCorrupt, st.Step, c.cfg.Steps)
	}
	c.params = append([]float64(nil), st.Params...)
	c.vel = append([]float64(nil), st.Velocity...)
	c.era = st.Era
	c.step = st.Step
	c.done = st.Step >= c.cfg.Steps
	// 成员信息恢复为已登记状态（worker 重连后即可继续，不自动算作在线）。
	now := c.clock.Now()
	for _, m := range st.Members {
		c.members[m.ID] = &memberState{ID: m.ID, JoinEra: st.Era, LastSeen: now}
	}
	return nil
}

// activeIDsLocked 返回按 ID 排序的活跃 worker 列表（canonical rank 顺序）。
func (c *Coordinator) activeIDsLocked() []string {
	ids := make([]string, 0, len(c.members))
	for id := range c.members {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func memberNames(ms []checkpoint.Member) []string {
	out := make([]string, 0, len(ms))
	for _, m := range ms {
		out = append(out, m.ID)
	}
	sort.Strings(out)
	return out
}
