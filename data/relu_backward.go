package data

type reluBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewReLUBackward(t1 *Tensor, result *Tensor) Backward {
	return &reluBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *reluBackward) Backward(loss *Tensor) {

}
