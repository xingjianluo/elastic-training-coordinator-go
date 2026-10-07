package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
	"github.com/xingjianluo/elastic-training-coordinator-go/transport"
)

// defaultCompletionGrace 是跑满后、关 socket 前给落后 worker 的默认宽限。
const defaultCompletionGrace = 200 * time.Millisecond

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

// Serve 启动本地 unix-socket RPC 服务并阻塞，直到：
//   - 训练跑满目标步数（ExitCompleted），或
//   - 收到 stop 请求 / 终止信号 / 父 ctx 取消（ExitInterrupted）。
//
// 终结前一定已在最后完整完成的状态上落最终检查点；尚未整体完成的在途步骤
// 一律作废，绝不半提交。返回前会关闭 socket、释放本地资源。
func (c *Coordinator) Serve(ctx context.Context) error {
	srv := transport.NewServer(c.cfg.SocketPath, c.handle)
	c.mu.Lock()
	c.server = srv
	alreadyDone := c.done
	c.mu.Unlock()
	if err := srv.Start(ctx); err != nil {
		return err
	}
	serveCtx, cancel := context.WithCancel(ctx)
	defer cancel()
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
			case <-serveCtx.Done():
				return
			case now := <-t.C:
				if _, dead := c.CheckTimeouts(now); len(dead) > 0 {
					c.log(EvFailure, "watchdog reaped dead workers: "+joinStr(dead))
				}
			}
		}
	}()
	// 恢复到一个已经跑满的作业：等其最终固化完成即可，原因保持 completed。
	if alreadyDone {
		c.Shutdown(ExitCompleted, "resumed into an already-complete job", c.cfg.ShutdownTimeout)
		c.waitTerminalBounded()
		cancel()
		srv.Stop()
		<-watchDone
		return nil
	}
	var reason ExitReason
	var detail string
	select {
	case <-ctx.Done():
		// 终止信号 / 父上下文取消：立即走有界优雅停机（落最终检查点、停接新工作）。
		reason, detail = ExitInterrupted, "serve context canceled (signal or parent)"
		c.log(EvLifecycle, "serve context canceled: initiating graceful shutdown")
	case <-c.completeCh:
		// 跑满目标步数：最终检查点已在提交临界区同步落盘。继续保持服务，
		// 直到完成时在册的每个 worker 都通过 next 收到 Done；再以 CompletionGrace
		// 作为有界兜底，避免某个已消失的 worker 让进程无法结束。
		reason, detail = ExitCompleted, "reached target steps"
		grace := c.cfg.CompletionGrace
		if grace == 0 {
			grace = defaultCompletionGrace
		}
		c.log(EvLifecycle, "training complete: serving Done to members before releasing socket")
		if grace > 0 {
			t := time.NewTimer(grace)
			select {
			case <-ctx.Done():
				reason, detail = ExitInterrupted, "signal while waiting for completion ack"
			case <-c.terminalCh:
				reason, detail = ExitInterrupted, "stop while waiting for completion ack"
			case <-c.completionCh:
			case <-t.C:
				c.log(EvLifecycle, "completion grace elapsed; releasing socket even if some members did not fetch Done")
			}
			t.Stop()
		}
		c.log(EvLifecycle, "training complete: entering graceful shutdown")
	case <-c.terminalCh:
		// stop RPC（或进程内 Shutdown）已经完成状态固化。
		reason, detail = ExitInterrupted, "stop request over local IPC"
	}
	// 状态固化：最终检查点已落、在途半成品作废、新工作被拒、终结信号关闭。
	c.Shutdown(reason, detail, c.cfg.ShutdownTimeout)
	c.waitTerminalBounded()
	cancel()
	srv.Stop()
	<-watchDone
	return nil
}

// waitTerminalBounded 等待终结信号，但绝不超过停机上限+1s；
// 即便固化（如磁盘卡死）超时，Serve 也会继续释放传输资源并返回。
func (c *Coordinator) waitTerminalBounded() {
	bound := c.cfg.ShutdownTimeout
	if bound <= 0 {
		bound = defaultShutdownTimeout
	}
	select {
	case <-c.terminalCh:
	case <-time.After(bound + time.Second):
		c.log(EvFailure, "terminal signal not observed within shutdown bound; forcing transport release")
	}
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
		resp, err := c.RequestStop(p)
		if err != nil {
			return replyError(err)
		}
		b, _ := json.Marshal(resp)
		return &transport.Envelope{Type: proto.MsgStop, Payload: b}

	default:
		return &transport.Envelope{Error: "unknown message type: " + req.Type, Code: "bad_request"}
	}
}
