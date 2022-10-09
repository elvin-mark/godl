package data

type tanhBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewTanhBackward(t1 *Tensor, result *Tensor) Node {
	return &tanhBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *tanhBackward) Backward(loss *Tensor) {

}

func (ab *tanhBackward) IsLeaf() bool {
	return false
}
