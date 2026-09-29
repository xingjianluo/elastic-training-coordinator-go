package collective

import (
	"errors"
	"math"
	"testing"

	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/sharding"
)

func TestAllReduceAndReduceScatterAllGatherOrderIndependent(t *testing.T) {
	replicasA := map[string][]float64{
		"w1": {1, 7, 2},
		"w2": {-2, 1, -3},
		"w3": {4, 0, 1},
	}
	replicasB := map[string][]float64{
		"w3": replicasA["w3"],
		"w1": replicasA["w1"],
		"w2": replicasA["w2"],
	}
	got, err := AllReduce(replicasA, OpSum)
	if err != nil {
		t.Fatal(err)
	}
	got2, err := AllReduce(replicasB, OpSum)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{3, 8, 0}
	t.Logf("members=[w1 w2 w3] decision=sum-elementwise-then-ignore-map-order got=%v want=%v", got, want)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 || math.Abs(got2[i]-want[i]) > 1e-12 {
			t.Fatalf("allreduce order dependent or wrong: %v %v", got, got2)
		}
	}

	plan, err := sharding.PlanFor(3, []string{"w1", "w2", "w3"})
	if err != nil {
		t.Fatal(err)
	}
	scattered, err := ReduceScatter(replicasA, plan)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("members=%v ranges=%v reduce-scatter=%v decision=owner-keeps-only-owned-range",
		plan.Workers, plan.Ranges, scattered)
	reassembled, err := AllGather(scattered, plan)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(reassembled[i]-want[i]) > 1e-12 {
			t.Fatalf("reduce-scatter+allgather=%v, want %v", reassembled, want)
		}
	}
}

func TestCollectiveRejectsMisalignedInput(t *testing.T) {
	plan, err := sharding.PlanFor(4, []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	missing := map[string][]float64{"a": {1, 2}}
	_, err = AllGather(missing, plan)
	t.Logf("members=%v ranges=%v supplied=%v decision=missing-shard-must-fail", plan.Workers, plan.Ranges, []string{"a"})
	if !errors.Is(err, ErrMisalignedShards) {
		t.Fatalf("missing shard got %v", err)
	}

	wrongLen := map[string][]float64{"a": {1, 2, 9}, "b": {3, 4}}
	_, err = AllGather(wrongLen, plan)
	t.Logf("members=%v ranges=%v lengths=a:3,b:2 decision=length-mismatch-must-fail", plan.Workers, plan.Ranges)
	if !errors.Is(err, ErrMisalignedShards) {
		t.Fatalf("wrong shard length got %v", err)
	}

	unequal := map[string][]float64{"a": {1, 2}, "b": {3}}
	_, err = AllReduce(unequal, OpSum)
	t.Logf("members=[a b] replica-lengths=2,1 decision=full-replica-misalignment-must-fail")
	if !errors.Is(err, ErrMisalignedShards) {
		t.Fatalf("unequal replicas got %v", err)
	}

	_, err = AllReduce(map[string][]float64{"a": {1}}, "max")
	if !errors.Is(err, ErrUnsupportedOp) {
		t.Fatalf("unsupported op got %v", err)
	}
}
