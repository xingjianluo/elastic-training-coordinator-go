package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
	"github.com/xingjianluo/elastic-training-coordinator-go/transport"
)

// Logger 简单日志接口（默认标准 logger）。
type Logger interface {
	Printf(format string, args ...any)
}

// Config worker 配置。
type Config struct {
	ID         string
	SocketPath string
	// PollInterval reduce 未就绪时的轮询间隔。
	PollInterval time.Duration
	// HeartbeatInterval 心跳间隔。
	HeartbeatInterval time.Duration
	// FinishDelay reduce 完成、allgather 之前的确定性停顿（停机演练用）。
	FinishDelay time.Duration
	Logger      Logger
}

// Worker 通过本地 IPC 与协调者协作的训练 worker。
type Worker struct {
	cfg    Config
	client *transport.Client
	logger Logger
	core   *Core
}

func New(cfg Config) *Worker {
	lg := cfg.Logger
	if lg == nil {
		lg = log.New(os.Stdout, "[worker "+cfg.ID+"] ", log.LstdFlags|log.Lmicroseconds)
	}
	if cfg.PollInterval == 0 {
		cfg.PollInterval = 5 * time.Millisecond
	}
	if cfg.HeartbeatInterval == 0 {
		cfg.HeartbeatInterval = 500 * time.Millisecond
	}
	cl := transport.NewClient(cfg.SocketPath)
	w := &Worker{cfg: cfg, client: cl, logger: lg}
	w.core = &Core{
		ID:   cfg.ID,
		Call: w.call,
		PollInterval: func(ctx context.Context) error {
			return sleepCtx(ctx, cfg.PollInterval)
		},
		FinishDelay: cfg.FinishDelay,
	}
	return w
}

// RPCError 携带协调者给出的稳定错误码。
type RPCError struct {
	Code string
	Msg  string
}

func (e *RPCError) Error() string { return fmt.Sprintf("%s: %s", e.Code, e.Msg) }

func (w *Worker) call(ctx context.Context, msgType string, payload any, out any) error {
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	resp, err := w.client.Call(ctx, &transport.Envelope{Type: msgType, Payload: b})
	if err != nil {
		return err
	}
	if resp.Error != "" {
		return &RPCError{Code: resp.Code, Msg: resp.Error}
	}
	if out != nil && len(resp.Payload) > 0 {
		return json.Unmarshal(resp.Payload, out)
	}
	return nil
}

// Core 暴露与传输无关的核心（测试可直接驱动）。
func (w *Worker) Core() *Core { return w.core }

// Run 不断领取步骤、计算梯度、参与集合通信，直到训练完成。
// reconnect=true 时：连接中断或遇到 stale_era 会退避重连并重新 join/next；
// 训练完成（包括完成后重连）由协调者幂等去重，绝不产生第二次贡献。
func (w *Worker) Run(ctx context.Context, reconnect bool) error {
	hbCtx, stopHB := context.WithCancel(ctx)
	defer stopHB()
	go w.heartbeatLoop(hbCtx)

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		done, err := w.joinThenRunSteps(ctx)
		if err != nil {
			if !reconnect || ctx.Err() != nil {
				return err
			}
			w.logger.Printf("connection/era error (%v); reconnecting after backoff", err)
			if sleepCtx(ctx, 20*time.Millisecond); ctx.Err() != nil {
				return ctx.Err()
			}
			continue
		}
		if done {
			w.logger.Printf("coordinator reports training DONE; exiting (a further reconnect is idempotent)")
			return nil
		}
	}
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// joinThenRunSteps 执行一次 join，然后尽可能多地完成步骤。
func (w *Worker) joinThenRunSteps(ctx context.Context) (bool, error) {
	jr, err := w.core.Join(ctx)
	if err != nil {
		return false, err
	}
	w.logger.Printf("joined; era=%d members=%v", jr.Membership.Era, memberIDs(jr.Membership.Members))
	for {
		nr, err := w.core.Next(ctx)
		if err != nil {
			if isStale(err) {
				w.logger.Printf("assignment invalidated by era change; rejoining to re-fetch")
				return false, err
			}
			return false, err
		}
		if nr.Done {
			return true, nil
		}
		if nr.Assignment == nil {
			if sleepCtx(ctx, w.cfg.PollInterval); ctx.Err() != nil {
				return false, ctx.Err()
			}
			continue
		}
		if err := w.core.DoStep(ctx, nr.Assignment); err != nil {
			if isCode(err, "training_complete") {
				// 其它 worker 已并发提交最后一步：训练确实完成，正常退出。
				return true, nil
			}
			if isStale(err) {
				w.logger.Printf("step %d aborted by coordinator (era change); will re-fetch assignment",
					nr.Assignment.Step)
				return false, err
			}
			return false, err
		}
	}
}

func isCode(err error, code string) bool {
	var rpcErr *RPCError
	if errors.As(err, &rpcErr) {
		return rpcErr.Code == code
	}
	return false
}

func isStale(err error) bool {
	var rpcErr *RPCError
	if errors.As(err, &rpcErr) {
		return rpcErr.Code == "stale_era"
	}
	return false
}

func memberIDs(ms []proto.Member) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.ID
	}
	return out
}

// heartbeatLoop 周期性发送心跳；任何错误都不致命（Run 会重连）。
func (w *Worker) heartbeatLoop(ctx context.Context) {
	t := time.NewTicker(w.cfg.HeartbeatInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			var hr proto.HeartbeatResponse
			_ = w.call(ctx, proto.MsgHeartbeat, proto.HeartbeatRequest{WorkerID: w.cfg.ID}, &hr)
		}
	}
}

// Close 关闭底层连接。
func (w *Worker) Close() error { return w.client.Close() }
