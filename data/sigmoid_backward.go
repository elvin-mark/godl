package data

type sigmoidBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewSigmoidBackward(t1 *Tensor, result *Tensor) Node {
	return &sigmoidBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *sigmoidBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		t1_loss.data[i] += val * ab.result.data[i] * (1 - ab.result.data[i])
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *sigmoidBackward) IsLeaf() bool {
	return false
}
