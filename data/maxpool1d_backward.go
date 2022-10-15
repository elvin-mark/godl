package data

type maxPool1dBackward struct {
	t1     *Tensor
	kernel int
	result *Tensor
}

func NewMaxPool1dBackward(t1 *Tensor, kernel int, result *Tensor) Node {
	return &maxPool1dBackward{
		t1:     t1,
		kernel: kernel,
		result: result,
	}
}

func (ab *maxPool1dBackward) Backward(loss *Tensor) {
	// t1_loss := NewTensor(ab.t1.shape)

}

func (ab *maxPool1dBackward) IsLeaf() bool {
	return false
}
