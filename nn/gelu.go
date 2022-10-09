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
