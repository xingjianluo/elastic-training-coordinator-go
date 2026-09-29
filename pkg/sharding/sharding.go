package sharding

import (
	"errors"
	"fmt"
	"slices"
)

var (
	ErrInvalidShardCount = errors.New("sharding: invalid shard count")
	ErrInvalidDimension  = errors.New("sharding: invalid tensor dimension")
	ErrMisalignedShards  = errors.New("sharding: worker shards are not aligned")
	ErrUnknownWorker     = errors.New("sharding: unknown worker")
)

// Range is a half-open tensor interval [Start, End).
type Range struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

func (r Range) Len() int { return r.End - r.Start }

// Plan assigns one contiguous, balanced tensor shard to every worker. Worker
// IDs are sorted before ranges are calculated, so the order in which callers
// discover members cannot affect parameter placement.
type Plan struct {
	Dimension int      `json:"dimension"`
	Workers   []string `json:"workers"`
	Ranges    []Range  `json:"ranges"`
}

// PlanFor builds a deterministic plan. Boundaries are calculated with
// floor(i*dimension/n), so each worker owns [floor(i*d/n), floor((i+1)*d/n)).
// A shard count larger than the dimension is rejected instead of silently
// creating empty shards.
func PlanFor(dimension int, workers []string) (Plan, error) {
	if dimension <= 0 {
		return Plan{}, fmt.Errorf("%w: dimension=%d", ErrInvalidDimension, dimension)
	}
	n := len(workers)
	if n <= 0 {
		return Plan{}, fmt.Errorf("%w: worker count must be greater than zero", ErrInvalidShardCount)
	}
	if n > dimension {
		return Plan{}, fmt.Errorf("%w: shard count %d exceeds tensor dimension %d", ErrInvalidShardCount, n, dimension)
	}

	ids := append([]string(nil), workers...)
	slices.Sort(ids)
	if n >= 2 {
		for i := 1; i < n; i++ {
			if ids[i] == ids[i-1] {
				return Plan{}, fmt.Errorf("%w: duplicate worker %q", ErrInvalidShardCount, ids[i])
			}
		}
	}

	ranges := make([]Range, n)
	for i := range ids {
		start := dimension * i / n
		end := dimension * (i + 1) / n
		if end <= start {
			return Plan{}, fmt.Errorf("%w: empty shard for worker %q", ErrInvalidShardCount, ids[i])
		}
		ranges[i] = Range{Start: start, End: end}
	}
	return Plan{Dimension: dimension, Workers: ids, Ranges: ranges}, nil
}

// Validate verifies that a received plan still partitions the complete tensor
// exactly once and uses one non-empty range per worker.
func (p Plan) Validate() error {
	expected, err := PlanFor(p.Dimension, p.Workers)
	if err != nil {
		return err
	}
	if len(p.Ranges) != len(expected.Ranges) {
		return fmt.Errorf("%w: got %d ranges, want %d", ErrMisalignedShards, len(p.Ranges), len(expected.Ranges))
	}
	for i, got := range p.Ranges {
		if got != expected.Ranges[i] {
			return fmt.Errorf("%w: worker %q owns [%d,%d), expected [%d,%d)",
				ErrMisalignedShards, expected.Workers[i], got.Start, got.End,
				expected.Ranges[i].Start, expected.Ranges[i].End)
		}
	}
	if p.Dimension != expected.Dimension {
		return fmt.Errorf("%w: dimension mismatch", ErrMisalignedShards)
	}
	return nil
}

func (p Plan) ShardFor(worker string) (Range, error) {
	if err := p.Validate(); err != nil {
		return Range{}, err
	}
	i := slices.Index(p.Workers, worker)
	if i < 0 {
		return Range{}, fmt.Errorf("%w: %q", ErrUnknownWorker, worker)
	}
	return p.Ranges[i], nil
}

// ValidateAlignment is used by a collective operation to ensure every
// participant is using the same partition. A different generation's plan, for
// example a resharded membership, is rejected rather than concatenated.
func (p Plan) ValidateAlignment(other Plan) error {
	if p.Dimension != other.Dimension {
		return fmt.Errorf("%w: dimensions %d and %d", ErrMisalignedShards, p.Dimension, other.Dimension)
	}
	if len(p.Workers) != len(other.Workers) {
		return fmt.Errorf("%w: shard counts %d and %d", ErrMisalignedShards, len(p.Workers), len(other.Workers))
	}
	for i := range p.Workers {
		if p.Workers[i] != other.Workers[i] {
			return fmt.Errorf("%w: worker at position %d is %q and %q",
				ErrMisalignedShards, i, p.Workers[i], other.Workers[i])
		}
		want, err := PlanFor(p.Dimension, p.Workers)
		if err != nil {
			return err
		}
		if p.Ranges[i] != want.Ranges[i] || other.Ranges[i] != want.Ranges[i] {
			return fmt.Errorf("%w: worker %q has mismatching range", ErrMisalignedShards, p.Workers[i])
		}
	}
	return nil
}
