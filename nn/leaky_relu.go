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

func (l *leakyReLU) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	return
}

func (l *leakyReLU) Train() {

}

func (l *leakyReLU) Eval() {

}
