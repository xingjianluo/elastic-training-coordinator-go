package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
	"github.com/xingjianluo/elastic-training-coordinator-go/transport"
)

// errCode 把协调者内部错误映射为稳定的可区分错误码字符串。
func errCode(err error) string {
	switch {
	case errors.Is(err, ErrUnknownWorker):
		return "unknown_worker"
	case errors.Is(err, ErrStaleEra):
		return "stale_era"
	case errors.Is(err, ErrDuplicateContrib):
		return "duplicate_contribution"
	case errors.Is(err, ErrStepNotOpen):
		return "step_not_open"
	case errors.Is(err, ErrShardMisaligned):
		return "shard_misaligned"
	case errors.Is(err, ErrTrainingComplete):
		return "training_complete"
	case errors.Is(err, ErrNoActiveMembers):
		return "no_active_members"
	case errors.Is(err, ErrShuttingDown):
		return "shutting_down"
	default:
		return "internal_error"
	}
}

// Serve 启动本地 unix-socket RPC 服务并阻塞，直到 ctx 取消、收到停止请求
// 或训练全部完成；返回前必然完成收尾（最终检查点 + 释放监听/连接）。
func (c *Coordinator) Serve(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	srv := transport.NewServer(c.cfg.SocketPath, c.handle)
	c.server = srv
	if err := srv.Start(ctx); err != nil {
		return err
	}
	// 周期性心跳超时检测（服务进程内使用真实时间）。
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		watchInterval := c.cfg.HeartbeatTimeout / 4
		if watchInterval <= 0 {
			watchInterval = 10 * time.Millisecond
		}
		t := time.NewTicker(watchInterval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case now := <-t.C:
				if _, dead := c.CheckTimeouts(now); len(dead) > 0 {
					c.log(EvFailure, "watchdog reaped dead workers: "+joinStr(dead))
				}
			}
		}
	}()
	// 周期性检查训练是否完成或已被请求停机，任一成立即优雅退出。
	donePoll := make(chan struct{})
	go func() {
		defer close(donePoll)
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				if c.Done() {
					c.log(EvDecision, "training complete; coordinator shutting down")
					cancel()
					return
				}
				if c.Stopping() {
					c.log(EvDecision, "stop requested; coordinator shutting down gracefully")
					cancel()
					return
				}
			}
		}
	}()
	<-ctx.Done()
	// 收尾：标记停机（拒绝尚在排队的新请求），在最新【完整提交】状态上落
	// 最终检查点并生成退出报告；在途步骤的部分贡献绝不写入。
	c.mu.Lock()
	c.stopping = true
	c.finalizeLocked()
	c.mu.Unlock()
	// 释放本地资源：监听 socket、连接与看门狗 goroutine 全部收干净再返回。
	srv.Wait()
	<-watchDone
	<-donePoll
	return nil
}

func joinStr(xs []string) string {
	out := ""
	for i, x := range xs {
		if i > 0 {
			out += ","
		}
		out += x
	}
	return out
}

func replyError(err error) *transport.Envelope {
	return &transport.Envelope{Error: err.Error(), Code: errCode(err)}
}

func (c *Coordinator) handle(ctx context.Context, req *transport.Envelope) *transport.Envelope {
	switch req.Type {
	case proto.MsgJoin:
		var p proto.JoinRequest
		if err := json.Unmarshal(req.Payload, &p); err != nil {
			return replyError(err)
		}
		resp := c.Join(p)
		b, _ := json.Marshal(resp)
		return &transport.Envelope{Type: proto.MsgJoin, Payload: b}

	case proto.MsgHeartbeat:
		var p proto.HeartbeatRequest
		if err := json.Unmarshal(req.Payload, &p); err != nil {
			return replyError(err)
		}
		if err := c.heartbeat(p.WorkerID); err != nil {
			return replyError(err)
		}
		b, _ := json.Marshal(proto.HeartbeatResponse{OK: true})
		return &transport.Envelope{Type: proto.MsgHeartbeat, Payload: b}

	case proto.MsgLeave:
		var p proto.LeaveRequest
		if err := json.Unmarshal(req.Payload, &p); err != nil {
			return replyError(err)
		}
		c.Leave(p.WorkerID)
		return &transport.Envelope{Type: proto.MsgLeave}

	case proto.MsgNext:
		var p proto.NextRequest
		if err := json.Unmarshal(req.Payload, &p); err != nil {
			return replyError(err)
		}
		resp := c.Next(p)
		b, _ := json.Marshal(resp)
		return &transport.Envelope{Type: proto.MsgNext, Payload: b}

	case proto.MsgSubmit:
		var p proto.SubmitRequest
		if err := json.Unmarshal(req.Payload, &p); err != nil {
			return replyError(err)
		}
		resp, err := c.Submit(p)
		if err != nil {
			return replyError(err)
		}
		b, _ := json.Marshal(resp)
		return &transport.Envelope{Type: proto.MsgSubmit, Payload: b}

	case proto.MsgAllgather:
		var p proto.AllgatherRequest
		if err := json.Unmarshal(req.Payload, &p); err != nil {
			return replyError(err)
		}
		resp, err := c.Allgather(p)
		if err != nil {
			return replyError(err)
		}
		b, _ := json.Marshal(resp)
		return &transport.Envelope{Type: proto.MsgAllgather, Payload: b}

	case proto.MsgCheckpoint:
		var p proto.CheckpointRequest
		if err := json.Unmarshal(req.Payload, &p); err != nil {
			return replyError(err)
		}
		resp, err := c.RequestCheckpoint(p.WorkerID)
		if err != nil {
			return replyError(err)
		}
		b, _ := json.Marshal(resp)
		return &transport.Envelope{Type: proto.MsgCheckpoint, Payload: b}

	case proto.MsgStop:
		var p proto.StopRequest
		if err := json.Unmarshal(req.Payload, &p); err != nil {
			return replyError(err)
		}
		resp := c.RequestStop(p.Requester)
		b, _ := json.Marshal(resp)
		return &transport.Envelope{Type: proto.MsgStop, Payload: b}

	default:
		return &transport.Envelope{Error: "unknown message type: " + req.Type, Code: "bad_request"}
	}
}
