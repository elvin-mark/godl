package data

type divBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewDivBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Node {
	return &divBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *divBackward) Backward(loss *Tensor) {

}

func (ab *divBackward) IsLeaf() bool {
	return false
}
