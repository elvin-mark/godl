package data

type siluBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewSiLUBackward(t1 *Tensor, result *Tensor) Node {
	return &siluBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *siluBackward) Backward(loss *Tensor) {

}

func (ab *siluBackward) IsLeaf() bool {
	return false
}
