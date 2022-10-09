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
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		if val > 0 {
			t1_loss.data[i] += val
		} else {
			t1_loss.data[i] += ab.alpha * val
		}
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *leakyReLUBackward) IsLeaf() bool {
	return false
}
