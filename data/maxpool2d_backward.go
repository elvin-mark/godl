package data

type maxPool2dBackward struct {
	t1     *Tensor
	kernel []int
	result *Tensor
}

func NewMaxPool2dBackward(t1 *Tensor, kernel []int, result *Tensor) Node {
	return &maxPool2dBackward{
		t1:     t1,
		kernel: kernel,
		result: result,
	}
}

func (ab *maxPool2dBackward) Backward(loss *Tensor) {
	// t1_loss := NewTensor(ab.t1.shape)

}

func (ab *maxPool2dBackward) IsLeaf() bool {
	return false
}
