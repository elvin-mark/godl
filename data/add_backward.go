package data

type addBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewAddBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Node {
	return &addBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *addBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	t2_loss := NewTensor(ab.t2.shape)

	ab.t1.node.Backward(t1_loss)
	ab.t2.node.Backward(t2_loss)
}

func (ab *addBackward) IsLeaf() bool {
	return false
}
