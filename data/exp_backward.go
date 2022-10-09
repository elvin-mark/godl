package data

type expBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewExpBackward(t1 *Tensor, result *Tensor) Backward {
	return &expBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *expBackward) Backward(loss *Tensor) {

}
