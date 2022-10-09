package data

type sigmoidBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewSigmoidBackward(t1 *Tensor, result *Tensor) Node {
	return &sigmoidBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *sigmoidBackward) Backward(loss *Tensor) {

}

func (ab *sigmoidBackward) IsLeaf() bool {
	return false
}
