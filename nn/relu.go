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

func (l *relu) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	return
}

func (l *relu) Train() {

}

func (l *relu) Eval() {

}
