package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
	"github.com/xingjianluo/elastic-training-coordinator-go/worker"
)

// fakeClock 是测试可控的时间源。
type fakeClock struct{ t time.Time }

func (f *fakeClock) Now() time.Time { return f.t }
func (f *fakeClock) Advance(d time.Duration) time.Time {
	f.t = f.t.Add(d)
	return f.t
}

func testConfig(dir string, steps int) Config {
	return Config{
		SocketPath:       "",
		CheckpointDir:    dir,
		NSamples:         32,
		NIn:              2,
		Seed:             20260930,
		Steps:            steps,
		Batch:            8,
		LR:               0.05,
		Momentum:         0.9,
		HeartbeatTimeout: 100 * time.Millisecond,
		StepTimeout:      60 * time.Millisecond,
		CheckpointEvery:  2,
	}
}

// localDispatch 把 worker 的消息直接路由到 coordinator 方法，返回可 JSON 序列化结果。
func localDispatch(c *Coordinator) func(ctx context.Context, t string, raw json.RawMessage) (any, error) {
	return func(ctx context.Context, msgType string, raw json.RawMessage) (any, error) {
		switch msgType {
		case proto.MsgJoin:
			var p proto.JoinRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err
			}
			return c.Join(p), nil
		case proto.MsgHeartbeat:
			var p proto.HeartbeatRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err
			}
			return proto.HeartbeatResponse{OK: true}, c.heartbeat(p.WorkerID)
		case proto.MsgNext:
			var p proto.NextRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err
			}
			return c.Next(p), nil
		case proto.MsgSubmit:
			var p proto.SubmitRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err
			}
			return c.Submit(p)
		case proto.MsgAllgather:
			var p proto.AllgatherRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err
			}
			return c.Allgather(p)
		case proto.MsgCheckpoint:
			var p proto.CheckpointRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err
			}
			return c.RequestCheckpoint(p.WorkerID)
		default:
			return nil, fmt.Errorf("unknown type %s", msgType)
		}
	}
}

func newCore(c *Coordinator, id string) *worker.Core {
	return &worker.Core{ID: id, Call: worker.LocalCall(localDispatch(c))}
}

// drainEvents 把判定日志打印到测试输出。
func drainEvents(t *testing.T, c *Coordinator) {
	t.Helper()
	for {
		select {
		case ev := <-c.Events():
			t.Logf("  [%s] %s", ev.Kind, ev.Detail)
		default:
			return
		}
	}
}

// runRound 驱动当前成员以给定顺序完整跑完当前在途的一步。
// 分两阶段调度（集合通信的真实语义）：先让所有成员提交 reduce-scatter 贡献，
// 再以给定顺序收集已归约分片、本地更新并 allgather。第二阶段的重复提交用于
// 验证协调者幂等去重。给定顺序可任意打乱，验证结果与 worker 顺序无关。
func runRound(t *testing.T, ctx context.Context, c *Coordinator, ids []string) {
	t.Helper()
	cores := map[string]*worker.Core{}
	assignments := map[string]*proto.StepAssignment{}
	for _, id := range ids {
		core := newCore(c, id)
		cores[id] = core
		nr, err := core.Next(ctx)
		if err != nil {
			t.Fatalf("%s next: %v", id, err)
		}
		if nr.Done {
			t.Fatalf("%s unexpectedly done mid-round", id)
		}
		if nr.Assignment == nil {
			t.Fatalf("%s got nil assignment", id)
		}
		assignments[id] = nr.Assignment
	}
	// 阶段一：所有成员提交 reduce-scatter 分片。
	for _, id := range ids {
		if _, err := cores[id].Submit(ctx, assignments[id]); err != nil {
			t.Fatalf("%s submit: %v", id, err)
		}
	}
	// 阶段二：按给定顺序取归约结果（重复提交必须被幂等去重）并 allgather。
	for _, id := range ids {
		sr, err := cores[id].SubmitRaw(ctx, submitReq(id, assignments[id]))
		if err != nil {
			t.Fatalf("%s re-submit: %v", id, err)
		}
		if !sr.Ready {
			t.Fatalf("%s expected reduced result ready", id)
		}
		if err := cores[id].Finish(ctx, assignments[id], sr); err != nil {
			t.Fatalf("%s finish: %v", id, err)
		}
	}
	drainEvents(t, c)
}

// submitReq 在测试侧重建一条提交请求（等价于 worker 内部计算）。
func submitReq(id string, a *proto.StepAssignment) proto.SubmitRequest {
	return worker.BuildSubmitRequest(id, a)
}

func assertCloseToBaseline(t *testing.T, c *Coordinator, tol float64) {
	t.Helper()
	base := c.Baseline()
	got := c.Params()
	vel := c.velCopy()
	if len(got) != len(base.Params) {
		t.Fatalf("param dim mismatch")
	}
	for i := range got {
		if abs(got[i]-base.Params[i]) > tol {
			t.Fatalf("param %d got=%.12f baseline=%.12f diff=%.3e", i, got[i], base.Params[i], got[i]-base.Params[i])
		}
		if abs(vel[i]-base.Velocity[i]) > tol {
			t.Fatalf("velocity %d got=%.12f baseline=%.12f", i, vel[i], base.Velocity[i])
		}
	}
	t.Logf("MATCH baseline within tol=%g: params=%.10f velocity=%.10f",
		tol, got, vel)
}

func (c *Coordinator) velCopy() []float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]float64(nil), c.vel...)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// MembershipSet 返回当前成员集合（测试用）。
func (c *Coordinator) MembershipSet() (map[string]struct{}, uint64) {
	m := c.Membership()
	out := map[string]struct{}{}
	for _, mem := range m.Members {
		out[mem.ID] = struct{}{}
	}
	return out, m.Era
}

func paramsDiff(a, b []float64) float64 {
	var d float64
	for i := range a {
		d += abs(a[i] - b[i])
	}
	return d
}

func expectErrCode(t *testing.T, err error, target error, name string) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("%s: err=%v want=%v", name, err, target)
	}
}
