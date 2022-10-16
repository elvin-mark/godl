package data

type nllBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewNLLBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Node {
	return &nllBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (m *nllBackward) Backward(loss *Tensor) {

}

func (m *nllBackward) IsLeaf() bool {
	return false
}
