package data

type logBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewLogBackward(t1 *Tensor, result *Tensor) Node {
	return &logBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *logBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		t1_loss.data[i] += val / ab.t1.data[i]
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *logBackward) IsLeaf() bool {
	return false
}
