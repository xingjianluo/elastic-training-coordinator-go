package collective

import (
	"errors"
	"fmt"
	"sort"

	"github.com/xingjianluo/elastic-training-coordinator-go/shard"
)

// ErrorKind 集合通信失败原因，调用方可据此区分拒绝理由。
type ErrorKind int

const (
	KindOK ErrorKind = iota
	KindEmptyGroup
	KindInvalidShardCount
	KindRankMismatch
	KindMisaligned
)

var (
	// ErrEmptyGroup 通信组里一个成员都没有。
	ErrEmptyGroup = errors.New("collective: empty communicator group")
	// ErrInvalidShardCount worldSize<=0 或 worldSize>向量长度。
	ErrInvalidShardCount = errors.New("collective: invalid shard/world count")
	// ErrRankMismatch rank 重复、越界或成员数与 worldSize 不一致。
	ErrRankMismatch = errors.New("collective: rank mismatch")
	// ErrMisaligned 各成员的区间/长度与统一分片规划对不齐。
	ErrMisaligned = errors.New("collective: shard plans misaligned")
)

// Input 描述一个 worker 对集合通信原语的一次贡献。
type Input struct {
	Rank      int
	WorldSize int
	VectorLen int
	// ShardRange 是该 rank 在本次通信中负责（reduce-scatter/allgather）的区间；
	// allreduce 时可留空。
	ShardRange shard.Interval
	// Data:
	//   reduce-scatter/allreduce: 该 worker 算出的完整局部向量（长度 VectorLen）；
	//   allgather: 该 worker 持有的分片（长度须等于 ShardRange.Len）。
	Data []float64
}

// Result 集合通信结果。
type Result struct {
	// Full 为全量结果（allreduce/allgather 时使用）。
	Full []float64
	// Chunk 为该 rank 负责区间对应的结果（reduce-scatter 时使用）。
	Chunk []float64
}

// validate 校验通信组并返回按 rank 升序排好的成员与统一分片方案。
// 校验规则：
//  1. 组非空；
//  2. 所有成员的 WorldSize/VectorLen 一致，worldSize 合法（1..n）；
//  3. 成员数 == worldSize，rank 不重复且在 [0,worldSize)；
//  4. 各 rank 声明的分片区间必须与统一规划（n 切 worldSize 片）逐片一致。
func validate(group []Input) ([]Input, *shard.Plan, error) {
	if len(group) == 0 {
		return nil, nil, ErrEmptyGroup
	}
	worldSize := group[0].WorldSize
	n := group[0].VectorLen
	if _, err := shard.NewPlan(n, worldSize); err != nil {
		return nil, nil, fmt.Errorf("%w: %v", ErrInvalidShardCount, err)
	}
	if len(group) != worldSize {
		return nil, nil, fmt.Errorf("%w: got %d contributors but worldSize=%d",
			ErrRankMismatch, len(group), worldSize)
	}
	seen := make(map[int]bool, worldSize)
	for _, in := range group {
		if in.WorldSize != worldSize || in.VectorLen != n {
			return nil, nil, fmt.Errorf("%w: member rank=%d declares worldSize=%d,n=%d, group expects %d,%d",
				ErrMisaligned, in.Rank, in.WorldSize, in.VectorLen, worldSize, n)
		}
		if in.Rank < 0 || in.Rank >= worldSize {
			return nil, nil, fmt.Errorf("%w: rank=%d out of range [0,%d)", ErrRankMismatch, in.Rank, worldSize)
		}
		if seen[in.Rank] {
			return nil, nil, fmt.Errorf("%w: duplicate rank=%d", ErrRankMismatch, in.Rank)
		}
		seen[in.Rank] = true
	}
	sorted := make([]Input, len(group))
	copy(sorted, group)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Rank < sorted[j].Rank })
	plan, _ := shard.NewPlan(n, worldSize)
	for _, in := range sorted {
		want := plan.IntervalOf(in.Rank)
		if in.ShardRange != want {
			return nil, nil, fmt.Errorf("%w: rank=%d declares %s, plan requires %s",
				ErrMisaligned, in.Rank, in.ShardRange, want)
		}
	}
	return sorted, plan, nil
}

// ReduceSum 以 canonical 顺序（rank 升序）对完整局部向量求和。
// 同一逻辑组无论请求到达顺序如何，结果都一致。
func ReduceSum(group []Input) ([]float64, error) {
	sorted, n, err := validate(group)
	if err != nil {
		return nil, err
	}
	_ = n
	out := make([]float64, sorted[0].VectorLen)
	for _, in := range sorted {
		if len(in.Data) != in.VectorLen {
			return nil, fmt.Errorf("%w: rank=%d data len=%d, want full vector len=%d",
				ErrMisaligned, in.Rank, len(in.Data), in.VectorLen)
		}
		for i, v := range in.Data {
			out[i] += v
		}
	}
	return out, nil
}

// ReduceScatter 对各 rank 的完整局部向量做 canonical 求和后按规划切分，
// 返回 in.Rank 负责区间上的片段。
func ReduceScatter(in Input, group []Input) (Result, error) {
	sum, err := ReduceSum(group)
	if err != nil {
		return Result{}, err
	}
	sorted, plan, err := validate(group)
	if err != nil {
		return Result{}, err
	}
	_ = sorted
	iv := plan.IntervalOf(in.Rank)
	return Result{Chunk: append([]float64(nil), sum[iv.Start:iv.End]...)}, nil
}

// AllGather 把各 rank 持有的分片按规划拼成完整向量。
func AllGather(in Input, group []Input) (Result, error) {
	sorted, plan, err := validate(group)
	if err != nil {
		return Result{}, err
	}
	out := make([]float64, plan.N)
	for _, m := range sorted {
		iv := plan.IntervalOf(m.Rank)
		if len(m.Data) != iv.Len() {
			return Result{}, fmt.Errorf("%w: rank=%d chunk len=%d, range %s needs %d",
				ErrMisaligned, m.Rank, len(m.Data), iv, iv.Len())
		}
		copy(out[iv.Start:iv.End], m.Data)
	}
	return Result{Full: out}, nil
}

// AllReduce 等价于 ReduceScatter+AllGather：对完整局部向量 canonical 求和，
// 每个 rank 都得到完整求和向量。
func AllReduce(in Input, group []Input) (Result, error) {
	sum, err := ReduceSum(group)
	if err != nil {
		return Result{}, err
	}
	return Result{Full: sum}, nil
}

// KindOf 返回集合通信错误的可区分原因。
func KindOf(err error) ErrorKind {
	switch {
	case err == nil:
		return KindOK
	case errors.Is(err, ErrEmptyGroup):
		return KindEmptyGroup
	case errors.Is(err, ErrInvalidShardCount):
		return KindInvalidShardCount
	case errors.Is(err, ErrRankMismatch):
		return KindRankMismatch
	case errors.Is(err, ErrMisaligned):
		return KindMisaligned
	default:
		return KindMisaligned
	}
}
