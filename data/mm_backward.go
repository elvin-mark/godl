package data

type mmBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewMmBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Node {
	return &mmBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *mmBackward) Backward(loss *Tensor) {
	t1_loss := loss.Mm(ab.t2.Transpose())
	t2_loss := ab.t1.Transpose().Mm(loss)

	ab.t1.node.Backward(t1_loss)
	ab.t2.node.Backward(t2_loss)
}

func (ab *mmBackward) IsLeaf() bool {
	return false
}
