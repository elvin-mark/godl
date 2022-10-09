package nn

import (
	data "godl/data"
)

type leakyReLU struct {
	alpha float64
}

func NewLeakyReLU() (l Module) {
	return &leakyReLU{}
}

func (l *leakyReLU) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.LeakyReLU(l.alpha)
	return
}
