package test

import (
	"github.com/elvin-mark/godl/data"
	"github.com/elvin-mark/godl/loss"
	"github.com/elvin-mark/godl/models"
	"github.com/elvin-mark/godl/optim"
)

func TestXOR() {
	inp := data.NewTensor(data.NewShape([]int{4, 2}))
	target := data.NewTensor(data.NewShape([]int{4, 1}))
	inp.SetData([]float64{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0})
	target.SetData([]float64{0.0, 1.0, 1.0, 0.0})

	s := models.NewMLP(2, 10, 2)

	loss := loss.NewCELoss()
	sgdOptim := optim.NewSGDOptimizer(s.GetWeights(), map[string]any{"lr": 0.001})

	o := s.Forward(inp)
	o.Print()

	for i := 0; i < 1000; i++ {
		sgdOptim.ZeroGrad()
		o := s.Forward(inp)
		l := loss.Criterion(o, target)
		l.Backward(nil)
		sgdOptim.Step()
	}
	o = s.Forward(inp)
	o.Softmax().Print()
}

func TestNN() {
	TestXOR()
}
