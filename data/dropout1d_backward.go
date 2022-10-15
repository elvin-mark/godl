package data

type dropout1dBackward struct {
	t1     *Tensor
	p      float64
	result *Tensor
}

func NewDropout1dBackward(t1 *Tensor, p float64, result *Tensor) Node {
	return &dropout1dBackward{
		t1:     t1,
		p:      p,
		result: result,
	}
}

func (ab *dropout1dBackward) Backward(loss *Tensor) {
	// t1_loss := NewTensor(ab.t1.shape)

}

func (ab *dropout1dBackward) IsLeaf() bool {
	return false
}
