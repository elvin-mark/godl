package test

import (
	"godl/data"
	"godl/loss"
	nn "godl/nn"
)

func TestXOR() {
	inp := data.NewTensor(data.NewShape([]int{4, 2}))
	target := data.NewTensor(data.NewShape([]int{4, 2}))
	inp.SetData([]float64{0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0})
	target.SetData([]float64{1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0})
	s := nn.NewSequential()
	s.AddModule(nn.NewLinear(2, 5, true))
	s.AddModule(nn.NewSigmoid())
	s.AddModule(nn.NewLinear(5, 2, true))

	loss := loss.NewMSELoss()

	o := s.Forward(inp)
	l := loss.Criterion(o, target)
	l.Backward(nil)

	for _, weight := range s.GetWeights() {
		weight.GetGrad().Print()
	}
}

func TestNN() {
	TestXOR()
}
