package coordinator

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
)

// TestShardCountsMatchBaseline 覆盖 1/2/3 个分片（worker），且每步打乱 worker
// 动作顺序，验证无论切几片、顺序如何，结果都与单机基准一致。
func TestShardCountsMatchBaseline(t *testing.T) {
	for _, world := range []int{1, 2, 3} {
		world := world
		t.Run(worldName(world), func(t *testing.T) {
			c, err := NewNoResume(testConfig(t.TempDir(), 6))
			if err != nil {
				t.Fatal(err)
			}
			ctx := context.Background()
			ids := make([]string, world)
			for i := range ids {
				ids[i] = "w" + string(rune('a'+i))
			}
			for _, id := range ids {
				if r := c.Join(proto.JoinRequest{WorkerID: id}); !r.OK {
					t.Fatalf("join %s failed", id)
				}
			}
			t.Logf("world=%d members=%v (rank by sorted id)", world, c.Membership().Members)
			for step := 0; step < 6; step++ {
				order := append([]string(nil), ids...)
				if step%2 == 1 {
					order[0], order[len(order)-1] = order[len(order)-1], order[0]
				}
				runRound(t, ctx, c, order)
			}
			if !c.Done() {
				t.Fatal("expected done")
			}
			assertCloseToBaseline(t, c, 1e-9)
		})
	}
}

func worldName(w int) string {
	switch w {
	case 1:
		return "1shard"
	case 2:
		return "2shards"
	default:
		return "3shards"
	}
}

// TestMembershipChangeMidTraining 训练中途先 2 个 worker，第 3 步后加入新 worker，
// 再过一步一个 worker 退出；验证分片重分配后继续训练，结果仍等于单机基准，
// 既不丢梯度也不重复累加。
func TestMembershipChangeMidTraining(t *testing.T) {
	c, err := NewNoResume(testConfig(t.TempDir(), 6))
	if err != nil {
		t.Fatal(err)
	}
	clk := &fakeClock{t: time.Unix(1000, 0)}
	c.SetClock(clk)
	ctx := context.Background()

	c.Join(proto.JoinRequest{WorkerID: "wa"})
	c.Join(proto.JoinRequest{WorkerID: "wb"})
	t.Logf("step0-1 with 2 members: %v", c.Membership().Members)
	runRound(t, ctx, c, []string{"wb", "wa"}) // 故意逆序
	runRound(t, ctx, c, []string{"wa", "wb"})

	// 中途加入 wc -> era 推进，在途（还没有在途）无影响；之后按 3 片重分配。
	c.Join(proto.JoinRequest{WorkerID: "wc"})
	if c.Membership().Era < 2 {
		t.Fatalf("era should advance on join, got %d", c.Membership().Era)
	}
	t.Logf("wc joined mid-training; membership now: %v", c.Membership().Members)
	runRound(t, ctx, c, []string{"wc", "wa", "wb"})

	// wb 中途退出。
	c.Leave("wb")
	t.Logf("wb left; membership now: %v", c.Membership().Members)
	runRound(t, ctx, c, []string{"wa", "wc"})
	runRound(t, ctx, c, []string{"wc", "wa"})
	runRound(t, ctx, c, []string{"wa", "wc"})

	if c.Step() != 6 || !c.Done() {
		t.Fatalf("step=%d done=%v", c.Step(), c.Done())
	}
	assertCloseToBaseline(t, c, 1e-9)
}

// TestNodeDiesDuringCollective 集合通信进行到一半时节点死亡：
// 已收到的部分贡献必须随 era 一起作废（绝不部分提交），新成员补入后整步重做。
func TestNodeDiesDuringCollective(t *testing.T) {
	c, err := NewNoResume(testConfig(t.TempDir(), 4))
	if err != nil {
		t.Fatal(err)
	}
	clk := &fakeClock{t: time.Unix(2000, 0)}
	c.SetClock(clk)
	ctx := context.Background()
	c.Join(proto.JoinRequest{WorkerID: "wa"})
	c.Join(proto.JoinRequest{WorkerID: "wb"})
	c.Join(proto.JoinRequest{WorkerID: "wc"})

	// 领取 step0 分工
	ca := newCore(c, "wa")
	cb := newCore(c, "wb")
	cc := newCore(c, "wc")
	na, _ := ca.Next(ctx)
	nb, _ := cb.Next(ctx)
	_, _ = cc.Next(ctx)

	// wa、wb 提交 reduce-scatter 贡献，但 wc 在提交前崩溃。
	if err := ca.DoSubmitOnly(ctx, na.Assignment); err != nil {
		t.Fatal(err)
	}
	if err := cb.DoSubmitOnly(ctx, nb.Assignment); err != nil {
		t.Fatal(err)
	}
	stepBefore := c.Step()
	c.MarkDead("wc", "simulated crash during collective")

	// wa 若此时继续提交旧 era 的贡献，必须被拒绝（stale_era），不会部分计入。
	err = ca.DoSubmitOnly(ctx, na.Assignment)
	if !errors.Is(err, ErrStaleEra) {
		t.Fatalf("expected stale era rejection, got %v", err)
	}
	t.Logf("post-crash submit rejected as stale: %v", err)
	if c.Step() != stepBefore {
		t.Fatal("partial collective must not advance committed step")
	}

	// 新 worker wd 加入补位，重新以新成员组成跑完剩余步骤。
	c.Join(proto.JoinRequest{WorkerID: "wd"})
	t.Logf("new membership after crash: %v", c.Membership().Members)
	for step := c.Step(); step < 4; step++ {
		runRound(t, ctx, c, []string{"wa", "wb", "wd"})
	}
	if c.Step() != 4 || !c.Done() {
		t.Fatalf("step=%d", c.Step())
	}
	assertCloseToBaseline(t, c, 1e-9)
}

// TestReconnectAfterCompletionNoDoubleContribution worker 完成本步甚至全部训练
// 后重连并重发，验证幂等：不产生第二次贡献。
func TestReconnectAfterCompletionNoDoubleContribution(t *testing.T) {
	c, err := NewNoResume(testConfig(t.TempDir(), 2))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	c.Join(proto.JoinRequest{WorkerID: "wa"})
	c.Join(proto.JoinRequest{WorkerID: "wb"})
	runRound(t, ctx, c, []string{"wa", "wb"})

	// 用“全新 Core”模拟 wa 断线后重连：重复 join 必须幂等。
	jr := c.Join(proto.JoinRequest{WorkerID: "wa"})
	if !jr.OK {
		t.Fatal("rejoin failed")
	}
	// 重跑 step0（旧 era 的同一步）提交：此时 step 已推进，会被识别为 stale。
	ca := newCore(c, "wa")
	nr, _ := ca.Next(ctx)
	if nr.Assignment != nil && nr.Assignment.Step != 1 {
		t.Fatalf("reconnected worker should see step 1, got %v", nr.Assignment)
	}

	// 完成全部训练后再次 join/next：返回 Done，不产生任何贡献。
	runRound(t, ctx, c, []string{"wb", "wa"})
	if !c.Done() {
		t.Fatal("expected done")
	}
	c.Join(proto.JoinRequest{WorkerID: "wa"})
	nr2, err := ca.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !nr2.Done {
		t.Fatal("reconnect after completion must return Done")
	}
	p1 := c.Params()
	c.Join(proto.JoinRequest{WorkerID: "wa"}) // 再来一次
	if d := paramsDiff(p1, c.Params()); d != 0 {
		t.Fatalf("reconnect changed params: diff=%v", d)
	}
	t.Logf("reconnect after completion is idempotent; params unchanged: %v", p1)
}

// TestSlowVsDeadClassification 明确区分掉队（心跳新鲜但步骤超时）与死亡
// （心跳超时），慢节点不被摘除、步骤不丢；死节点被摘除并推进 era。
func TestSlowVsDeadClassification(t *testing.T) {
	c, err := NewNoResume(testConfig(t.TempDir(), 3))
	if err != nil {
		t.Fatal(err)
	}
	clk := &fakeClock{t: time.Unix(3000, 0)}
	c.SetClock(clk)
	ctx := context.Background()
	c.Join(proto.JoinRequest{WorkerID: "wa"})
	c.Join(proto.JoinRequest{WorkerID: "wb"})
	c.Join(proto.JoinRequest{WorkerID: "wc"})

	// 打开 step0，只有 wa 完成；此时推进时间超过 stepTimeout 但不超过心跳超时。
	ca := newCore(c, "wa")
	if _, err := ca.Next(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := newCore(c, "wb").Next(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := newCore(c, "wc").Next(ctx); err != nil {
		t.Fatal(err)
	}
	clk.Advance(80 * time.Millisecond) // > stepTimeout(60ms), < heartbeatTimeout(100ms)
	// 刷新 wa 的心跳，wb/wc 保持加入时心跳（80ms 仍未心跳超时）
	if err := ca.Heartbeat(ctx); err != nil {
		t.Fatal(err)
	}
	slow, dead := c.CheckTimeouts(clk.Now())
	if len(dead) != 0 {
		t.Fatalf("fresh heartbeats must not be classified dead: %v", dead)
	}
	if len(slow) < 2 {
		t.Fatalf("expected wb/wc classified slow(straggler), got %+v", slow)
	}
	t.Logf("stragglers correctly distinguished from dead: %+v", slow)
	set, _ := c.MembershipSet()
	if _, still := set["wb"]; !still {
		t.Fatal("slow worker must not be removed")
	}

	// 再推进超过心跳超时，且不发心跳 => wb/wc 死亡。
	clk.Advance(50 * time.Millisecond)
	slow2, dead2 := c.CheckTimeouts(clk.Now())
	if len(dead2) != 2 {
		t.Fatalf("expected 2 dead, got %v", dead2)
	}
	if len(slow2) != 0 {
		t.Fatalf("dead workers should not also be reported slow: %v", slow2)
	}
	t.Logf("dead workers after heartbeat timeout: %v", dead2)
	if m := c.Membership(); len(m.Members) != 1 {
		t.Fatalf("only wa should remain, got %v", m.Members)
	}
}
