package data

type meanBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewMeanBackward(t1 *Tensor, result *Tensor) Node {
	return &meanBackward{
		t1:     t1,
		result: result,
	}
}

func (m *meanBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(m.t1.shape)
	n := float64(m.t1.Size())
	if loss.Size() == 1 {
		for i := 0; i < t1_loss.Size(); i++ {
			t1_loss.data[i] = loss.Item() / n
		}
	}
	m.t1.node.Backward(t1_loss)
}

func (m *meanBackward) IsLeaf() bool {
	return false
}
