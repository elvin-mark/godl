package nn

import (
	data "github.com/elvin-mark/godl/data"
)

type dropout2d struct {
	p        float64
	training bool
}

func NewDropout2d(p float64) (l Module) {
	return &dropout2d{
		p:        p,
		training: true,
	}
}

func (l *dropout2d) Forward(inp *data.Tensor) (out *data.Tensor) {
	if l.training {
		out = inp.Dropout2d(l.p)
	} else {
		out = inp
	}
	return
}

func (l *dropout2d) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	return
}

func (l *dropout2d) Train() {
	l.training = true
}

func (l *dropout2d) Eval() {
	l.training = false
}
