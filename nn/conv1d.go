package nn

import (
	data "godl/data"
)

type conv1d struct {
	w          *data.Tensor
	b          *data.Tensor
	kernelSize int
	stride     int
	padding    int
}

func NewConv1d(inChannels int, outChannels int, kernelSize int, stride int, padding int, bias bool) (l Module) {
	w := data.NewTensor(data.NewShape([]int{outChannels, inChannels, kernelSize}))
	w.RandN(0, 1)
	w.SetRequiresGrad(true)
	var b *data.Tensor
	b = nil
	if bias {
		b = data.NewTensor(data.NewShape([]int{1, outChannels, 1}))
		b.RandN(0, 1)
		b.SetRequiresGrad(true)
	}
	return &conv1d{
		w:          w,
		b:          b,
		kernelSize: kernelSize,
		stride:     stride,
		padding:    padding,
	}
}

func (l *conv1d) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp.Conv1d(l.w, l.stride, l.padding)
	if l.b != nil {
		out = out.Add(l.b)
	}
	return
}

func (l *conv1d) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	weights = append(weights, l.w)
	weights = append(weights, l.b)
	return
}

func (l *conv1d) Train() {

}

func (l *conv1d) Eval() {

}
