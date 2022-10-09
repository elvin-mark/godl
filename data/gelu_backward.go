package data

type geluBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewGeLUBackward(t1 *Tensor, result *Tensor) Backward {
	return &geluBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *geluBackward) Backward(loss *Tensor) {

}
