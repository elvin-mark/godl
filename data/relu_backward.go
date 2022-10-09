package data

type reluBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewReLUBackward(t1 *Tensor, result *Tensor) Node {
	return &reluBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *reluBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		if val > 0 {
			t1_loss.data[i] += val
		}
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *reluBackward) IsLeaf() bool {
	return false
}
