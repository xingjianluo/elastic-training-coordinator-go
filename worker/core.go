package worker

import (
	"context"
	"encoding/json"

	"github.com/xingjianluo/elastic-training-coordinator-go/nn"
	"github.com/xingjianluo/elastic-training-coordinator-go/proto"
)

// CallFunc 是一次本地 IPC 调用的抽象。生产实现走 transport，
// 进程内测试可以直接调用 coordinator 方法，从而在不引入网络不确定性的
// 情况下执行完全相同的计算与协议逻辑。
type CallFunc func(ctx context.Context, msgType string, payload any, out any) error

// Core 是与传输无关的 worker 计算/协议逻辑。
type Core struct {
	ID           string
	Call         CallFunc
	PollInterval durationNapper
}

type durationNapper func(ctx context.Context) error

// DecodeSamples 把展平的 [x..,y] 行解回 Sample。
func DecodeSamples(flat []float64, nIn int) []nn.Sample {
	row := nIn + 1
	if row == 0 || len(flat)%row != 0 {
		return nil
	}
	out := make([]nn.Sample, 0, len(flat)/row)
	for k := 0; k < len(flat); k += row {
		x := append([]float64(nil), flat[k:k+nIn]...)
		out = append(out, nn.Sample{X: x, Y: flat[k+nIn]})
	}
	return out
}

// Join 加入集群。
func (c *Core) Join(ctx context.Context) (proto.JoinResponse, error) {
	var jr proto.JoinResponse
	err := c.Call(ctx, proto.MsgJoin, proto.JoinRequest{WorkerID: c.ID}, &jr)
	return jr, err
}

// Next 领取分工。
func (c *Core) Next(ctx context.Context) (proto.NextResponse, error) {
	var nr proto.NextResponse
	err := c.Call(ctx, proto.MsgNext, proto.NextRequest{WorkerID: c.ID}, &nr)
	return nr, err
}

// Submit 计算局部分片梯度并只提交一次 reduce-scatter（测试可分阶段驱动）。
func (c *Core) Submit(ctx context.Context, a *proto.StepAssignment) (proto.SubmitResponse, error) {
	return c.SubmitRaw(ctx, submitRequestOf(c.ID, a))
}

// SubmitRaw 直接发送一条 reduce-scatter 提交（用于幂等重发）。
func (c *Core) SubmitRaw(ctx context.Context, sub proto.SubmitRequest) (proto.SubmitResponse, error) {
	var sr proto.SubmitResponse
	err := c.Call(ctx, proto.MsgSubmit, sub, &sr)
	return sr, err
}

// DoSubmitOnly 是 Submit 的便捷包装，供故障注入测试使用。
func (c *Core) DoSubmitOnly(ctx context.Context, a *proto.StepAssignment) error {
	_, err := c.Submit(ctx, a)
	return err
}

// Heartbeat 发送一次心跳。
func (c *Core) Heartbeat(ctx context.Context) error {
	var hr proto.HeartbeatResponse
	return c.Call(ctx, proto.MsgHeartbeat, proto.HeartbeatRequest{WorkerID: c.ID}, &hr)
}

// DoStep 执行一个完整训练步骤（提交 -> 轮询至 reduce 完成 -> 本地更新 -> allgather）。
func (c *Core) DoStep(ctx context.Context, a *proto.StepAssignment) error {
	sr, err := c.Submit(ctx, a)
	if err != nil {
		return err
	}
	sub := submitRequestOf(c.ID, a)
	for !sr.Ready {
		if c.PollInterval != nil {
			if err := c.PollInterval(ctx); err != nil {
				return err
			}
		}
		// 重发同一贡献：协调者必须幂等去重。
		sr, err = c.SubmitRaw(ctx, sub)
		if err != nil {
			return err
		}
	}
	return c.Finish(ctx, a, sr)
}

// Finish 拿到已就绪的 reduce-scatter 结果后，本地做 SGD+momentum 更新并 allgather。
func (c *Core) Finish(ctx context.Context, a *proto.StepAssignment, sr proto.SubmitResponse) error {
	iv := a.ParamRange
	paramChunk := append([]float64(nil), a.Params[iv.Start:iv.End]...)
	velChunk := append([]float64(nil), sr.VelocityChunk...)
	nn.StepSGDMomentum(paramChunk, sr.Chunk, velChunk, a.LR, a.Momentum)
	ag := proto.AllgatherRequest{
		WorkerID: c.ID, Era: a.Era, Step: a.Step, Rank: a.Rank,
		Range: iv, Chunk: paramChunk, VelocityChunk: velChunk,
	}
	var ar proto.AllgatherResponse
	return c.Call(ctx, proto.MsgAllgather, ag, &ar)
}

// BuildSubmitRequest 根据分工重建一条 reduce-scatter 提交请求（测试用）。
func BuildSubmitRequest(id string, a *proto.StepAssignment) proto.SubmitRequest {
	return submitRequestOf(id, a)
}

func submitRequestOf(id string, a *proto.StepAssignment) proto.SubmitRequest {
	batch := DecodeSamples(a.Samples, a.NIn)
	model := &nn.Model{Params: append([]float64(nil), a.Params...), NIn: a.NIn}
	norm := a.BatchSize
	if norm == 0 {
		norm = len(batch)
	}
	// reduce-scatter：提交完整局部梯度（按全 batch 归一化），协调者负责按片归约。
	localGrad := model.GradientN(batch, norm)
	return proto.SubmitRequest{
		WorkerID: id, Era: a.Era, Step: a.Step, Rank: a.Rank,
		VectorLen: len(localGrad), Range: a.ParamRange,
		Chunk: append([]float64(nil), localGrad...),
	}
}

// RunUntilDone 反复领取并完成步骤直到 Done。遇到 stale_era 等可恢复错误
// 由调用方决定是否重新 Join；这里返回错误类型码即可。
func (c *Core) RunUntilDone(ctx context.Context) error {
	for {
		nr, err := c.Next(ctx)
		if err != nil {
			return err
		}
		if nr.Done {
			return nil
		}
		if nr.Assignment == nil {
			if c.PollInterval != nil {
				if err := c.PollInterval(ctx); err != nil {
					return err
				}
			}
			continue
		}
		if err := c.DoStep(ctx, nr.Assignment); err != nil {
			return err
		}
	}
}

// LocalCall 基于 coordinator 直接方法调用构造一个 CallFunc（进程内测试用）。
func LocalCall(dispatch func(ctx context.Context, msgType string, raw json.RawMessage) (any, error)) CallFunc {
	return func(ctx context.Context, msgType string, payload any, out any) error {
		b, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		res, err := dispatch(ctx, msgType, b)
		if err != nil {
			return err
		}
		if out == nil || res == nil {
			return nil
		}
		rb, err := json.Marshal(res)
		if err != nil {
			return err
		}
		return json.Unmarshal(rb, out)
	}
}
