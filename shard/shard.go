package shard

import (
	"errors"
	"fmt"
)

// ErrorKind 区分分片相关失败原因，便于调用方按原因分别处理。
type ErrorKind int

const (
	KindOK ErrorKind = iota
	KindInvalidCount
	KindMisaligned
	KindRangeMismatch
	KindIndexOutOfRange
)

// ErrInvalidShardCount 在分片数非法（<=0 或超过向量长度）时返回。
var ErrInvalidShardCount = errors.New("shard: invalid shard count")

// ErrMisaligned 在各 worker 的分片方案彼此对不齐时返回。
var ErrMisaligned = errors.New("shard: shard plans are misaligned")

// ErrRangeMismatch 在给定区间与规划应有的区间不一致时返回。
var ErrRangeMismatch = errors.New("shard: shard range mismatch")

// ErrIndexOutOfRange 在 shard/rank 下标越界时返回。
var ErrIndexOutOfRange = errors.New("shard: index out of range")

// Interval 是参数向量上的半开区间 [Start,End)。
type Interval struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

func (i Interval) Len() int { return i.End - i.Start }

func (i Interval) Contains(p int) bool { return p >= i.Start && p < i.End }

// AdjacentTo 判断两段是否在 other 开始处正好相接（i 在前）。
func (i Interval) AdjacentTo(other Interval) bool { return i.End == other.Start }

func (i Interval) String() string { return fmt.Sprintf("[%d,%d)", i.Start, i.End) }

// Plan 描述把长度 n 的向量切成 shardCount 个连续非空分片后的方案。
// 余数 r=n%c 依次分给前 r 个分片（它们长度为 q+1），其余长度为 q。
type Plan struct {
	N          int `json:"n"`
	ShardCount int `json:"shard_count"`
}

// NewPlan 校验并创建分片方案。
func NewPlan(n, shardCount int) (*Plan, error) {
	if n <= 0 {
		return nil, fmt.Errorf("%w: vector length n=%d must be positive", ErrInvalidShardCount, n)
	}
	if shardCount <= 0 {
		return nil, fmt.Errorf("%w: shardCount=%d must be positive", ErrInvalidShardCount, shardCount)
	}
	if shardCount > n {
		return nil, fmt.Errorf("%w: shardCount=%d exceeds vector length n=%d (every shard must be non-empty)",
			ErrInvalidShardCount, shardCount, n)
	}
	return &Plan{N: n, ShardCount: shardCount}, nil
}

// shardStart 返回第 shardIndex 片的起点。
func (p *Plan) shardStart(shardIndex int) int {
	q, r := p.N/p.ShardCount, p.N%p.ShardCount
	if shardIndex < r {
		return shardIndex * (q + 1)
	}
	return r*(q+1) + (shardIndex-r)*q
}

// IntervalOf 返回第 shardIndex 片的半开区间。
func (p *Plan) IntervalOf(shardIndex int) Interval {
	start := p.shardStart(shardIndex)
	end := p.N
	if shardIndex+1 < p.ShardCount {
		end = p.shardStart(shardIndex + 1)
	}
	return Interval{Start: start, End: end}
}

// IntervalOfRank 按给定 worldSize 返回 rank 对应区间。
// worldSize 必须与 ShardCount 一致，否则视为对不齐。
func (p *Plan) IntervalOfRank(rank, worldSize int) (Interval, error) {
	if worldSize != p.ShardCount {
		return Interval{}, fmt.Errorf("%w: worldSize=%d but plan has shardCount=%d",
			ErrMisaligned, worldSize, p.ShardCount)
	}
	if rank < 0 || rank >= worldSize {
		return Interval{}, fmt.Errorf("%w: rank=%d worldSize=%d", ErrIndexOutOfRange, rank, worldSize)
	}
	return p.IntervalOf(rank), nil
}

// Equal 两个方案是否完全一致。
func (p *Plan) Equal(o *Plan) bool {
	if p == nil || o == nil {
		return p == o
	}
	return p.N == o.N && p.ShardCount == o.ShardCount
}

// ValidateRange 校验给定区间是否正好等于规划中 shardIndex 片。
func (p *Plan) ValidateRange(shardIndex int, got Interval) error {
	want := p.IntervalOf(shardIndex)
	if got != want {
		return fmt.Errorf("%w: shard %d wants %s, got %s", ErrRangeMismatch, shardIndex, want, got)
	}
	return nil
}

// Split 把向量按规划切成连续片段（共享底层数组的切片）。
func Split[T any](p *Plan, v []T) ([][]T, error) {
	if len(v) != p.N {
		return nil, fmt.Errorf("%w: vector len=%d but plan n=%d", ErrRangeMismatch, len(v), p.N)
	}
	out := make([][]T, p.ShardCount)
	for i := 0; i < p.ShardCount; i++ {
		iv := p.IntervalOf(i)
		out[i] = v[iv.Start:iv.End]
	}
	return out, nil
}

// Align 校验多个 worker 提交的规划必须全部一致（非 nil、N 与 ShardCount 相同）。
func Align(plans ...*Plan) error {
	if len(plans) == 0 {
		return fmt.Errorf("%w: no plans supplied", ErrMisaligned)
	}
	var base *Plan
	for i, p := range plans {
		if p == nil {
			return fmt.Errorf("%w: plan %d is nil", ErrMisaligned, i)
		}
		if base == nil {
			base = p
			continue
		}
		if !base.Equal(p) {
			return fmt.Errorf("%w: plan 0 = {n=%d,shards=%d}, plan %d = {n=%d,shards=%d}",
				ErrMisaligned, base.N, base.ShardCount, i, p.N, p.ShardCount)
		}
	}
	return nil
}

// KindOf 返回错误对应的可区分原因。
func KindOf(err error) ErrorKind {
	switch {
	case err == nil:
		return KindOK
	case errors.Is(err, ErrInvalidShardCount):
		return KindInvalidCount
	case errors.Is(err, ErrIndexOutOfRange):
		return KindIndexOutOfRange
	case errors.Is(err, ErrRangeMismatch):
		return KindRangeMismatch
	case errors.Is(err, ErrMisaligned):
		return KindMisaligned
	default:
		return KindMisaligned
	}
}
