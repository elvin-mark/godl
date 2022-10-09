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
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		tmp := ab.result.data[i] / ab.t1.data[i]
		t1_loss.data[i] += val * (tmp + ab.t1.data[i]*tmp*(1-tmp))
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *siluBackward) IsLeaf() bool {
	return false
}
