package data

type geluBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewGeLUBackward(t1 *Tensor, result *Tensor) Node {
	return &geluBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *geluBackward) Backward(loss *Tensor) {

}

func (ab *geluBackward) IsLeaf() bool {
	return false
}
