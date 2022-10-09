package test

import (
	"godl/data"
	nn "godl/nn"
)

func TestLayers() {
	s := nn.NewSequential()
	s.AddModule(nn.NewLinear(2, 5, true))
	s.AddModule(nn.NewSigmoid())
	s.AddModule(nn.NewLinear(5, 3, true))
	t := data.NewTensor(data.NewShape([]int{3, 2}))
	s.Forward(t).Print()
}
