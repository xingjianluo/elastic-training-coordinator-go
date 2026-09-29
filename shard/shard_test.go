package shard

import (
	"errors"
	"testing"
)

func TestNewPlanRejectsInvalidCount(t *testing.T) {
	cases := []struct{ n, c int }{
		{0, 2}, {-1, 2}, {10, 0}, {10, -3}, {3, 4}, // 分片数不能超过长度（每片非空）
	}
	for _, tc := range cases {
		_, err := NewPlan(tc.n, tc.c)
		if !errors.Is(err, ErrInvalidShardCount) {
			t.Fatalf("NewPlan(%d,%d) err=%v, want ErrInvalidShardCount", tc.n, tc.c, err)
		}
		if KindOf(err) != KindInvalidCount {
			t.Fatalf("kind=%v, want KindInvalidCount", KindOf(err))
		}
	}
}

func TestPlanCoversVectorContiguously(t *testing.T) {
	p, err := NewPlan(10, 3)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("plan n=%d shards=%d intervals: %s %s %s",
		p.N, p.ShardCount, p.IntervalOf(0), p.IntervalOf(1), p.IntervalOf(2))
	pos := 0
	for i := 0; i < p.ShardCount; i++ {
		iv := p.IntervalOf(i)
		if iv.Start != pos {
			t.Fatalf("shard %d starts %d, want %d (gap/overlap)", i, iv.Start, pos)
		}
		if iv.End <= iv.Start {
			t.Fatalf("shard %d empty: %s", i, iv)
		}
		pos = iv.End
	}
	if pos != 10 {
		t.Fatalf("shards end at %d, want 10", pos)
	}
	// 余数分给前 r 片：10=3*3+1 -> 长度 4,3,3
	if l := p.IntervalOf(0).Len(); l != 4 {
		t.Fatalf("shard0 len=%d want 4", l)
	}
}

func TestSplitAndValidateRange(t *testing.T) {
	p, _ := NewPlan(7, 2)
	parts, err := Split(p, []int{0, 1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 || len(parts[0]) != 4 || len(parts[1]) != 3 {
		t.Fatalf("split parts=%v", parts)
	}
	if err := p.ValidateRange(1, p.IntervalOf(1)); err != nil {
		t.Fatal(err)
	}
	if err := p.ValidateRange(1, Interval{Start: 0, End: 3}); !errors.Is(err, ErrRangeMismatch) {
		t.Fatalf("got %v want ErrRangeMismatch", err)
	}
}

func TestAlignRejectsMismatch(t *testing.T) {
	a, _ := NewPlan(10, 2)
	b, _ := NewPlan(10, 3)
	c, _ := NewPlan(9, 2)
	if err := Align(a, a); err != nil {
		t.Fatalf("identical plans should align: %v", err)
	}
	if err := Align(a, b); !errors.Is(err, ErrMisaligned) {
		t.Fatalf("got %v want ErrMisaligned", err)
	}
	if err := Align(a, c); !errors.Is(err, ErrMisaligned) {
		t.Fatalf("got %v want ErrMisaligned", err)
	}
	if err := Align(a, nil); !errors.Is(err, ErrMisaligned) {
		t.Fatalf("got %v want ErrMisaligned", err)
	}
}

func TestIntervalOfRankRejectsWorldSizeMismatch(t *testing.T) {
	p, _ := NewPlan(10, 2)
	if _, err := p.IntervalOfRank(0, 3); !errors.Is(err, ErrMisaligned) {
		t.Fatalf("got %v want ErrMisaligned", err)
	}
	if _, err := p.IntervalOfRank(2, 2); !errors.Is(err, ErrIndexOutOfRange) {
		t.Fatalf("got %v want ErrIndexOutOfRange", err)
	}
}
