package nn

import (
	data "github.com/elvin-mark/godl/data"
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

func (l *tanh) Train() {

}

func (l *tanh) Eval() {

}
