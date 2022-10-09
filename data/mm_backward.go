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

}

func (ab *mmBackward) IsLeaf() bool {
	return false
}
