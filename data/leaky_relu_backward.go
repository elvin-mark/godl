package data

type leakyReLUBackward struct {
	t1     *Tensor
	alpha  float64
	result *Tensor
}

func NewLeakyReLUBackward(t1 *Tensor, alpha float64, result *Tensor) Node {
	return &leakyReLUBackward{
		t1:     t1,
		alpha:  alpha,
		result: result,
	}
}

func (ab *leakyReLUBackward) Backward(loss *Tensor) {

}

func (ab *leakyReLUBackward) IsLeaf() bool {
	return false
}
