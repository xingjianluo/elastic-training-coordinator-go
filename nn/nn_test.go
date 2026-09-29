package nn

import "testing"

func TestDatasetDeterministic(t *testing.T) {
	a := NewDataset(16, 3, 42)
	b := NewDataset(16, 3, 42)
	for i := range a.Samples {
		if a.Samples[i].Y != b.Samples[i].Y {
			t.Fatalf("sample %d non-deterministic", i)
		}
		for j := range a.Samples[i].X {
			if a.Samples[i].X[j] != b.Samples[i].X[j] {
				t.Fatalf("sample %d x%d non-deterministic", i, j)
			}
		}
	}
}

func TestBaselineDeterministic(t *testing.T) {
	cfg := BaselineConfig{NSamples: 32, NIn: 2, Seed: 1, Steps: 6, Batch: 8, LR: 0.05, Momentum: 0.9}
	r1 := RunBaseline(cfg)
	r2 := RunBaseline(cfg)
	for i := range r1.Params {
		if r1.Params[i] != r2.Params[i] || r1.Velocity[i] != r2.Velocity[i] {
			t.Fatalf("baseline not deterministic: %v vs %v", r1.Params, r2.Params)
		}
	}
	t.Logf("baseline params=%v loss=%.10f", r1.Params, r1.Loss)
	if r1.Loss >= 100 {
		t.Fatalf("training not reducing error, loss=%f", r1.Loss)
	}
}

func TestGradientShape(t *testing.T) {
	m := NewModel(2)
	ds := NewDataset(4, 2, 7)
	g := m.Gradient(ds.Samples)
	if len(g) != 3 {
		t.Fatalf("gradient dim=%d want 3 (w0,w1,b)", len(g))
	}
}
