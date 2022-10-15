package nn

import (
	data "godl/data"
)

type maxPool2d struct {
	kernelSize []int
}

func NewMaxPool2d(kernelSize []int) (l Module) {
	return &maxPool2d{
		kernelSize: kernelSize,
	}
}

func (l *maxPool2d) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.MaxPool2d(l.kernelSize)
	return
}

func (l *maxPool2d) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	return
}

func (l *maxPool2d) Train() {

}

func (l *maxPool2d) Eval() {

}
