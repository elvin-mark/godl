package data

import "math"

type powBackward struct {
	t1     *Tensor
	exp    float64
	result *Tensor
}

func NewPowBackward(t1 *Tensor, exp float64, result *Tensor) Node {
	return &powBackward{
		t1:     t1,
		exp:    exp,
		result: result,
	}
}

func (ab *powBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		t1_loss.data[i] += val * math.Pow(ab.t1.data[i], ab.exp-1) * ab.exp
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *powBackward) IsLeaf() bool {
	return false
}
