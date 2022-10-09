package test

import (
	data "godl/data"
	nn "godl/nn"
)

func TestLayers() {
	l := nn.NewLinear(2, 3, true)
	t := data.NewTensor(data.NewShape([]int{5, 2}))
	l.Forward(t).Print()
}
