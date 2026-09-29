package collective

import (
	"errors"
	"fmt"
	"slices"

	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/sharding"
)

const (
	// OpSum is intentionally explicit. All distributed averages in this
	// project are an OpSum allreduce followed by division by the current
	// member count, which makes resizing a membership auditable.
	OpSum = "sum"
)

var (
	ErrInvalidShardCount = errors.New("collective: invalid shard count")
	ErrMisalignedShards  = errors.New("collective: shard ranges do not align")
	ErrEmptyReplica      = errors.New("collective: no replica supplied")
	ErrUnsupportedOp     = errors.New("collective: unsupported reduction operation")
)

func validateReplicas(replicas map[string][]float64) (int, error) {
	if len(replicas) == 0 {
		return 0, ErrEmptyReplica
	}
	ids := make([]string, 0, len(replicas))
	dimension := -1
	for id, values := range replicas {
		if len(values) == 0 {
			return 0, fmt.Errorf("%w: worker %q sent an empty vector", ErrMisalignedShards, id)
		}
		if dimension == -1 {
			dimension = len(values)
		} else if len(values) != dimension {
			return 0, fmt.Errorf("%w: worker %q sent dimension %d, expected %d",
				ErrMisalignedShards, id, len(values), dimension)
		}
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return dimension, nil
}

// AllReduce returns the element-wise sum of equally-sized replicas. It does not
// depend on map iteration order.
func AllReduce(replicas map[string][]float64, op string) ([]float64, error) {
	if op != OpSum {
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedOp, op)
	}
	dimension, err := validateReplicas(replicas)
	if err != nil {
		return nil, err
	}
	out := make([]float64, dimension)
	for _, values := range replicas {
		for i, v := range values {
			out[i] += v
		}
	}
	return out, nil
}

// AllGather reconstructs the complete vector from one owned range per worker.
// The canonical plan determines output placement; worker map order is ignored.
func AllGather(shards map[string][]float64, plan sharding.Plan) ([]float64, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if len(shards) != len(plan.Workers) {
		return nil, fmt.Errorf("%w: got %d shards, plan has %d workers",
			ErrMisalignedShards, len(shards), len(plan.Workers))
	}

	out := make([]float64, plan.Dimension)
	for i, worker := range plan.Workers {
		values, ok := shards[worker]
		if !ok {
			return nil, fmt.Errorf("%w: worker %q did not provide shard %v",
				ErrMisalignedShards, worker, plan.Ranges[i])
		}
		r := plan.Ranges[i]
		if len(values) != r.Len() {
			return nil, fmt.Errorf("%w: worker %q sent %d values for range [%d,%d), expected %d",
				ErrMisalignedShards, worker, len(values), r.Start, r.End, r.Len())
		}
		copy(out[r.Start:r.End], values)
	}
	return out, nil
}

// ReduceScatter performs an allreduce and returns only the range owned by each
// worker. Combining it with AllGather is equivalent to an allreduce.
func ReduceScatter(replicas map[string][]float64, plan sharding.Plan) (map[string][]float64, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	reduced, err := AllReduce(replicas, OpSum)
	if err != nil {
		return nil, err
	}
	if len(reduced) != plan.Dimension {
		return nil, fmt.Errorf("%w: vector dimension %d does not match plan dimension %d",
			ErrMisalignedShards, len(reduced), plan.Dimension)
	}
	if len(replicas) != len(plan.Workers) {
		return nil, fmt.Errorf("%w: got %d replicas, plan has %d workers",
			ErrMisalignedShards, len(replicas), len(plan.Workers))
	}
	out := make(map[string][]float64, len(plan.Workers))
	for i, worker := range plan.Workers {
		if _, ok := replicas[worker]; !ok {
			return nil, fmt.Errorf("%w: worker %q did not provide a replica", ErrMisalignedShards, worker)
		}
		r := plan.Ranges[i]
		out[worker] = append([]float64(nil), reduced[r.Start:r.End]...)
	}
	return out, nil
}
