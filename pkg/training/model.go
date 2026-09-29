package training

import "fmt"

// Model is the deterministic local reference model used by tests and the local
// demo. Its objective is sum((params-target)^2)/2, whose gradient is
// params-target. The optimizer is vanilla momentum.
type Model struct {
	Params   []float64 `json:"params"`
	Momentum []float64 `json:"momentum"`
	Target   []float64 `json:"target"`
	LR       float64   `json:"lr"`
	Mu       float64   `json:"mu"`
}

// NewModel returns reproducible initial parameters. Callers may overwrite
// Target only before the model participates in a checkpointed run.
func NewModel(dimension int, lr, mu float64) *Model {
	if dimension <= 0 {
		panic(fmt.Sprintf("training: dimension must be positive, got %d", dimension))
	}
	m := &Model{
		Params:   make([]float64, dimension),
		Momentum: make([]float64, dimension),
		Target:   make([]float64, dimension),
		LR:       lr,
		Mu:       mu,
	}
	for i := range m.Target {
		m.Target[i] = float64(i + 1)
	}
	return m
}

// Gradient returns the gradient of the deterministic quadratic objective.
func Gradient(params, target []float64) []float64 {
	if len(params) != len(target) {
		panic(fmt.Sprintf("training: gradient dimension mismatch: %d params and %d targets", len(params), len(target)))
	}
	out := make([]float64, len(params))
	for i := range params {
		out[i] = params[i] - target[i]
	}
	return out
}

// Apply consumes one already-averaged gradient:
//
//	v_t = mu*v_{t-1} + gradient_t
//	p_t = p_{t-1} - lr*v_t.
func (m *Model) Apply(gradient []float64) {
	if len(gradient) != len(m.Params) {
		panic(fmt.Sprintf("training: gradient dimension %d does not match parameters %d", len(gradient), len(m.Params)))
	}
	for i, g := range gradient {
		m.Momentum[i] = m.Mu*m.Momentum[i] + g
		m.Params[i] -= m.LR * m.Momentum[i]
	}
}

// Baseline runs the same optimizer without distribution and is the reference
// against which every sharding/membership test compares.
func Baseline(dimension, steps int, lr, mu float64) []float64 {
	m := NewModel(dimension, lr, mu)
	for range steps {
		g := Gradient(m.Params, m.Target)
		m.Apply(g)
	}
	return append([]float64(nil), m.Params...)
}
