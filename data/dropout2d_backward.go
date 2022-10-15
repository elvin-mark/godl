package data

type dropout2dBackward struct {
	t1     *Tensor
	p      float64
	result *Tensor
}

func NewDropout2dBackward(t1 *Tensor, p float64, result *Tensor) Node {
	return &dropout2dBackward{
		t1:     t1,
		p:      p,
		result: result,
	}
}

func (ab *dropout2dBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	for i := 0; i < t1_loss.Size(); i++ {
		if ab.result.data[i] != 0 {
			t1_loss.data[i] = loss.data[i] / ab.p
		}
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *dropout2dBackward) IsLeaf() bool {
	return false
}
