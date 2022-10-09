package nn

import (
	data "godl/data"
)

type relu struct {
}

func NewReLU() (l Module) {
	return &relu{}
}

func (l *relu) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.ReLU()
	return
}
