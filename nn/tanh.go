package nn

import (
	data "godl/data"
)

type tanh struct {
}

func NewTanh() (l Module) {
	return &tanh{}
}

func (l *tanh) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.Sigmoid()
	return
}

func (l *tanh) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	return
}
