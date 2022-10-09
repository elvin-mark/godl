package nn

import (
	data "godl/data"
)

type silu struct {
}

func NewSiLU() (l Module) {
	return &silu{}
}

func (l *silu) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.SiLU()
	return
}
