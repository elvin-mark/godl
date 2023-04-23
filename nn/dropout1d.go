package nn

import (
	data "github.com/elvin-mark/godl/data"
)

type dropout1d struct {
	p        float64
	training bool
}

func NewDropout1d(p float64) (l Module) {
	return &dropout1d{
		p:        p,
		training: true,
	}
}

func (l *dropout1d) Forward(inp *data.Tensor) (out *data.Tensor) {
	if l.training {
		out = inp.Dropout1d(l.p)
	} else {
		out = inp
	}
	return
}

func (l *dropout1d) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	return
}

func (l *dropout1d) Train() {
	l.training = true
}

func (l *dropout1d) Eval() {
	l.training = false
}
