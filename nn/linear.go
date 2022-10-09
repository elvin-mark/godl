package nn

import (
	data "godl/data"
)

type linear struct {
	w *data.Tensor
	b *data.Tensor
}

func NewLinear(numInp int, numOut int, bias bool) (l Module) {
	w := data.NewTensor(data.NewShape([]int{numInp, numOut}))
	w.RandN(0, 1)
	var b *data.Tensor
	b = nil
	if bias {
		b = data.NewTensor(data.NewShape([]int{1, numOut}))
		b.RandN(0, 1)
	}
	return &linear{
		w: w,
		b: b,
	}
}

func (l *linear) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.Mm(l.w)
	if l.b != nil {
		out = out.Add(l.b)
	}
	return
}
