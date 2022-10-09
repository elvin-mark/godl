package data

type subBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewSubBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Backward {
	return &subBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *subBackward) Backward(loss *Tensor) {

}
