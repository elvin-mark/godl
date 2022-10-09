package data

type addBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewAddBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Backward {
	return &addBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *addBackward) Backward(loss *Tensor) {

}
