package nn

import (
	data "godl/data"
)

type sigmoid struct {
}

func NewSigmoid() (l Module) {
	return &sigmoid{}
}

func (l *sigmoid) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.Sigmoid()
	return
}
