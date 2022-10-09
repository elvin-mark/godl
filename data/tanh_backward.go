package data

type tanhBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewTanhBackward(t1 *Tensor, result *Tensor) Node {
	return &tanhBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *tanhBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		t1_loss.data[i] += val * (1 - ab.result.data[i]*ab.result.data[i])
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *tanhBackward) IsLeaf() bool {
	return false
}
