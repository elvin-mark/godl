package nn

import (
	data "godl/data"
)

type gelu struct {
}

func NewGeLU() (l Module) {
	return &gelu{}
}

func (l *gelu) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.GeLU()
	return
}

func (l *gelu) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	return
}

func (l *gelu) Train() {

}

func (l *gelu) Eval() {

}
