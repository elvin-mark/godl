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
	w.SetRequiresGrad(true)
	var b *data.Tensor
	b = nil
	if bias {
		b = data.NewTensor(data.NewShape([]int{1, numOut}))
		b.RandN(0, 1)
		b.SetRequiresGrad(true)
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

func (l *linear) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	weights = append(weights, l.w)
	weights = append(weights, l.b)
	return
}
