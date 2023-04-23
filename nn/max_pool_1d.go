package nn

import (
	data "github.com/elvin-mark/godl/data"
)

type maxPool1d struct {
	kernelSize int
}

func NewMaxPool1d(kernelSize int) (l Module) {
	return &maxPool1d{
		kernelSize: kernelSize,
	}
}

func (l *maxPool1d) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.MaxPool1d(l.kernelSize)
	return
}

func (l *maxPool1d) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	return
}

func (l *maxPool1d) Train() {

}

func (l *maxPool1d) Eval() {

}
