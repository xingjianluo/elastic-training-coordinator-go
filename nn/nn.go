package nn

// Sample 是一条确定性训练样本 (x, y)。
type Sample struct {
	X []float64
	Y float64
}

// Dataset 是一个带标签的本地合成数据集。
type Dataset struct {
	Samples []Sample
	NIn     int
}

// NewDataset 用固定种子生成可复现的线性回归数据集：
// y = sum(w_true_i * x_i) + b_true，标签带可复现的确定性噪声。
func NewDataset(nSamples, nIn int, seed int64) *Dataset {
	ds := &Dataset{
		Samples: make([]Sample, nSamples),
		NIn:     nIn,
	}
	// 轻量 LCG，保证任何环境下结果一致（不依赖随机库实现）。
	state := uint64(seed)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		// 映射到 [-1,1)
		return (float64(state>>40)/float64(1<<24))*2 - 1
	}
	wTrue := make([]float64, nIn)
	for i := range wTrue {
		wTrue[i] = 0.7 + 0.1*float64(i)
	}
	bTrue := 0.35
	for k := 0; k < nSamples; k++ {
		x := make([]float64, nIn)
		var y float64
		for i := 0; i < nIn; i++ {
			x[i] = next()
			y += wTrue[i] * x[i]
		}
		y += bTrue + 0.01*next()
		ds.Samples[k] = Sample{X: x, Y: y}
	}
	return ds
}

// Model 是 w*x+b 形式的线性模型，权重按 [w0..wk, b] 存放。
type Model struct {
	Params []float64
	NIn    int
}

// NewModel 创建全零初始参数的模型。
func NewModel(nIn int) *Model {
	return &Model{Params: make([]float64, nIn+1), NIn: nIn}
}

// Forward 计算前向输出 sum(w_i*x_i)+b。
func (m *Model) Forward(x []float64) float64 {
	var out float64
	for i := 0; i < m.NIn; i++ {
		out += m.Params[i] * x[i]
	}
	return out + m.Params[m.NIn]
}

// Loss 计算数据集上的 MSE。
func (m *Model) Loss(ds *Dataset) float64 {
	var sum float64
	for k := range ds.Samples {
		d := m.Forward(ds.Samples[k].X) - ds.Samples[k].Y
		sum += d * d
	}
	return sum / float64(len(ds.Samples))
}

// Gradient 计算 batch 上对全部参数的梯度向量（MSE 损失，按 batch 大小平均）。
// 顺序与单机顺序遍历完全一致，便于位级对齐。
func (m *Model) Gradient(batch []Sample) []float64 {
	return m.GradientN(batch, len(batch))
}

// GradientN 计算梯度但按给定的 norm 归一化。
// 分布式场景下每个 worker 只持有 batch 的一个分片，但必须按整个 batch 的
// 大小归一化，这样各分片 canonical 求和后才严格等于单机整 batch 梯度：
//
//	(2/norm) * Σ_sub  + ... = (2/norm) * Σ_all
func (m *Model) GradientN(batch []Sample, norm int) []float64 {
	grad := make([]float64, m.NIn+1)
	if norm <= 0 {
		norm = len(batch)
	}
	for k := range batch {
		pred := m.Forward(batch[k].X)
		d := 2 * (pred - batch[k].Y) / float64(norm)
		for i := 0; i < m.NIn; i++ {
			grad[i] += d * batch[k].X[i]
		}
		grad[m.NIn] += d
	}
	return grad
}

// StepSGDMomentum 以 SGD + momentum 原地更新参数：
//
//	v = momentum*v + grad
//	p = p - lr*v
func StepSGDMomentum(params, grad, velocity []float64, lr, momentum float64) {
	for i := range params {
		velocity[i] = momentum*velocity[i] + grad[i]
		params[i] -= lr * velocity[i]
	}
}

// BaselineConfig 单机基准配置。
type BaselineConfig struct {
	NSamples int
	NIn      int
	Seed     int64
	Steps    int
	Batch    int
	LR       float64
	Momentum float64
}

// BaselineResult 基准结果。
type BaselineResult struct {
	Params   []float64
	Velocity []float64
	Loss     float64
}

// RunBaseline 单机顺序训练：每步取数据集中连续的 Batch 个样本（步内循环），
// 作为所有分布式/弹性场景的对照基准。
func RunBaseline(cfg BaselineConfig) *BaselineResult {
	ds := NewDataset(cfg.NSamples, cfg.NIn, cfg.Seed)
	model := NewModel(cfg.NIn)
	velocity := make([]float64, cfg.NIn+1)
	for step := 0; step < cfg.Steps; step++ {
		off := (step * cfg.Batch) % cfg.NSamples
		batch := make([]Sample, 0, cfg.Batch)
		for j := 0; j < cfg.Batch; j++ {
			batch = append(batch, ds.Samples[(off+j)%cfg.NSamples])
		}
		grad := model.Gradient(batch)
		StepSGDMomentum(model.Params, grad, velocity, cfg.LR, cfg.Momentum)
	}
	return &BaselineResult{
		Params:   model.Params,
		Velocity: velocity,
		Loss:     model.Loss(ds),
	}
}
