package data

type softmaxBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewSoftmaxBackward(t1 *Tensor, result *Tensor) Node {
	return &softmaxBackward{
		t1:     t1,
		result: result,
	}
}

func (m *softmaxBackward) Backward(loss *Tensor) {

}

func (m *softmaxBackward) IsLeaf() bool {
	return false
}
