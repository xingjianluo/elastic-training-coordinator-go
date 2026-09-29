package collective

import (
	"errors"
	"testing"

	"github.com/xingjianluo/elastic-training-coordinator-go/shard"
)

func planRanges(n, world int) []shard.Interval {
	p, _ := shard.NewPlan(n, world)
	out := make([]shard.Interval, world)
	for r := 0; r < world; r++ {
		out[r] = p.IntervalOf(r)
	}
	return out
}

// makeGroup 构造 world 个成员，每个成员的完整局部向量由其 rank 决定。
func makeGroup(n, world int, vec func(rank int) []float64) []Input {
	ranges := planRanges(n, world)
	g := make([]Input, world)
	for r := 0; r < world; r++ {
		g[r] = Input{Rank: r, WorldSize: world, VectorLen: n, ShardRange: ranges[r], Data: vec(r)}
	}
	return g
}

func TestAllReduceMatchesCanonicalSum(t *testing.T) {
	const n = 5
	vecs := func(world int) [][]float64 {
		vs := make([][]float64, world)
		for r := range vs {
			v := make([]float64, n)
			for i := range v {
				v[i] = float64(r+1) + 0.01*float64(i)
			}
			vs[r] = v
		}
		return vs
	}
	for _, world := range []int{1, 2, 3, 5} {
		vs := vecs(world)
		group := makeGroup(n, world, func(rank int) []float64 { return vs[rank] })
		// 打乱到达顺序，验证 canonical rank 顺序与顺序无关。
		shuffled := append([]Input(nil), group...)
		if world > 1 {
			shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
		}
		res, err := AllReduce(group[0], shuffled)
		if err != nil {
			t.Fatalf("world=%d: %v", world, err)
		}
		// canonical: rank 0 开始逐个加
		want := make([]float64, n)
		for r := 0; r < world; r++ {
			for i := 0; i < n; i++ {
				want[i] += vs[r][i]
			}
		}
		for i := range want {
			if res.Full[i] != want[i] {
				t.Fatalf("world=%d idx=%d got=%v want=%v", world, i, res.Full, want)
			}
		}
		t.Logf("allreduce world=%d ranges=%v sum=%v", world, ranges2(planRanges(n, world)), want)
	}
}

func ranges2(rs []shard.Interval) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.String()
	}
	return out
}

func TestReduceScatterAndAllGatherRoundTrip(t *testing.T) {
	const n = 6
	world := 3
	group := makeGroup(n, world, func(rank int) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = float64((rank+1)*10 + i)
		}
		return v
	})
	// 先 reduce-scatter 拿到各片
	chunks := make([][]float64, world)
	for r := 0; r < world; r++ {
		res, err := ReduceScatter(group[r], group)
		if err != nil {
			t.Fatal(err)
		}
		chunks[r] = res.Chunk
		if len(res.Chunk) != planRanges(n, world)[r].Len() {
			t.Fatalf("rank %d chunk len wrong", r)
		}
	}
	// 再 allgather（每 rank 只持自己片）
	agGroup := make([]Input, world)
	ranges := planRanges(n, world)
	for r := 0; r < world; r++ {
		agGroup[r] = Input{Rank: r, WorldSize: world, VectorLen: n, ShardRange: ranges[r], Data: chunks[r]}
	}
	full, err := AllGather(agGroup[0], agGroup)
	if err != nil {
		t.Fatal(err)
	}
	direct, err := ReduceSum(group)
	if err != nil {
		t.Fatal(err)
	}
	for i := range direct {
		if full.Full[i] != direct[i] {
			t.Fatalf("idx %d rs+ag=%v != reduce %v", i, full.Full, direct)
		}
	}
	t.Logf("reduce-scatter ranges=%v; allgather reconstruct=%v", ranges2(ranges), full.Full)
}

func TestCollectiveRejectsBadInputs(t *testing.T) {
	// 空组
	if _, err := AllReduce(Input{}, nil); !errors.Is(err, ErrEmptyGroup) || KindOf(err) != KindEmptyGroup {
		t.Fatalf("empty group: %v", err)
	}
	// worldSize 非法
	bad := []Input{{Rank: 0, WorldSize: 0, VectorLen: 4, Data: []float64{1, 2, 3, 4}}}
	if _, err := AllReduce(bad[0], bad); !errors.Is(err, ErrInvalidShardCount) {
		t.Fatalf("got %v", err)
	}
	// 成员数与 worldSize 不一致
	g := makeGroup(6, 3, func(r int) []float64 { return make([]float64, 6) })
	if _, err := AllReduce(g[0], g[:2]); !errors.Is(err, ErrRankMismatch) {
		t.Fatalf("got %v", err)
	}
	// rank 重复
	dup := append([]Input(nil), g...)
	dup[1] = dup[0]
	if _, err := AllReduce(dup[0], dup); !errors.Is(err, ErrRankMismatch) {
		t.Fatalf("got %v", err)
	}
	// 区间对不齐：rank1 声明了错误区间
	mis := append([]Input(nil), g...)
	mis[1].ShardRange = shard.Interval{Start: 0, End: 2}
	if _, err := AllReduce(mis[1], mis); !errors.Is(err, ErrMisaligned) || KindOf(err) != KindMisaligned {
		t.Fatalf("got %v", err)
	}
	// 向量长度对不齐
	mis2 := append([]Input(nil), g...)
	mis2[1].Data = []float64{1, 2}
	if _, err := ReduceSum(mis2); !errors.Is(err, ErrMisaligned) {
		t.Fatalf("got %v", err)
	}
}
