package test

import (
	"godl/data"
	"godl/loss"
	nn "godl/nn"
	"godl/optim"
)

func TestXOR() {
	inp := data.NewTensor(data.NewShape([]int{4, 2}))
	target := data.NewTensor(data.NewShape([]int{4, 2}))
	inp.SetData([]float64{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0})
	target.SetData([]float64{1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0})

	s := nn.NewSequential()
	s.AddModule(nn.NewLinear(2, 3, true))
	s.AddModule(nn.NewSigmoid())
	s.AddModule(nn.NewLinear(3, 2, true))
	s.AddModule(nn.NewSigmoid())

	loss := loss.NewMSELoss()
	sgdOptim := optim.NewSGDOptimizer(s.GetWeights(), map[string]any{"lr": 0.2})

	o := s.Forward(inp)
	o.Print()

	for i := 0; i < 500; i++ {
		sgdOptim.ZeroGrad()
		o := s.Forward(inp)
		l := loss.Criterion(o, target)
		l.Backward(nil)
		sgdOptim.Step()
	}
	o = s.Forward(inp)
	o.Print()

}

func TestNN() {
	TestXOR()
}
