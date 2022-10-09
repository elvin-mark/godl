package data

type expBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewExpBackward(t1 *Tensor, result *Tensor) Node {
	return &expBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *expBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		t1_loss.data[i] += val * ab.result.data[i]
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *expBackward) IsLeaf() bool {
	return false
}
