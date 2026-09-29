package sharding

import (
	"errors"
	"slices"
	"testing"
)

func TestPlanOrderIndependentAndBalanced(t *testing.T) {
	dimension := 7
	orderA := []string{"w3", "w1", "w2"}
	orderB := []string{"w2", "w3", "w1"}
	planA, err := PlanFor(dimension, orderA)
	if err != nil {
		t.Fatalf("plan A: %v", err)
	}
	planB, err := PlanFor(dimension, orderB)
	if err != nil {
		t.Fatalf("plan B: %v", err)
	}
	t.Logf("members=%v ranges=%v decision=canonical-sorted-worker-ids", planA.Workers, planA.Ranges)
	if !slices.Equal(planA.Workers, planB.Workers) || !slices.Equal(planA.Ranges, planB.Ranges) {
		t.Fatalf("plans depend on input order: %+v != %+v", planA, planB)
	}
	if planA.Ranges[0] != (Range{0, 2}) || planA.Ranges[1] != (Range{2, 4}) || planA.Ranges[2] != (Range{4, 7}) {
		t.Fatalf("unexpected balanced ranges: %+v", planA.Ranges)
	}
	covered := 0
	for i, r := range planA.Ranges {
		if r.Len() <= 0 {
			t.Fatalf("empty shard for %s: %v", planA.Workers[i], r)
		}
		if i > 0 && planA.Ranges[i-1].End != r.Start {
			t.Fatalf("gap or overlap between shards: %+v", planA.Ranges)
		}
		covered += r.Len()
	}
	if covered != dimension {
		t.Fatalf("ranges cover %d elements, want %d", covered, dimension)
	}
}

func TestPlanRejectsInvalidShardCountAndMisalignment(t *testing.T) {
	tests := []struct {
		name      string
		dimension int
		workers   []string
		want      error
	}{
		{"zero dimension", 0, []string{"a"}, ErrInvalidDimension},
		{"no workers", 3, nil, ErrInvalidShardCount},
		{"more shards than dimension", 2, []string{"a", "b", "c"}, ErrInvalidShardCount},
		{"duplicate workers", 4, []string{"a", "a"}, ErrInvalidShardCount},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PlanFor(tt.dimension, tt.workers)
			t.Logf("members=%v decision=%q expected=%v", tt.workers, err, tt.want)
			if !errors.Is(err, tt.want) {
				t.Fatalf("got %v, want %v", err, tt.want)
			}
		})
	}

	plan, err := PlanFor(8, []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	other := plan
	other.Ranges[1].End++
	other.Dimension = 9
	err = plan.ValidateAlignment(other)
	t.Logf("members=%v valid=%v corrupt=%v decision=different-dimensions-must-not-align",
		plan.Workers, plan.Ranges, other.Ranges)
	if !errors.Is(err, ErrMisalignedShards) {
		t.Fatalf("got %v, want ErrMisalignedShards", err)
	}
}
