package data

type sigmoidBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewSigmoidBackward(t1 *Tensor, result *Tensor) Backward {
	return &sigmoidBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *sigmoidBackward) Backward(loss *Tensor) {

}
