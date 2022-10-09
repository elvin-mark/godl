package data

type mulBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewMulBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Backward {
	return &mulBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *mulBackward) Backward(loss *Tensor) {

}
