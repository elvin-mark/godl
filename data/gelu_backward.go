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
	t1_loss := NewTensor(ab.t1.shape)
	for i, val := range loss.data {
		tmp := ab.result.data[i] / ab.t1.data[i]
		t1_loss.data[i] += val * (tmp + 1.702*ab.t1.data[i]*tmp*(1-tmp))
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *geluBackward) IsLeaf() bool {
	return false
}
