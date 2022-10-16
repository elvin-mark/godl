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
	} else {
		t1_loss.Ones()
		scale := 1.0 / float64(m.t1.Size()/loss.Size())
		new_shape := make([]int, loss.GetDim())
		for i := 0; i < loss.GetDim(); i++ {
			new_shape[i] = 1
		}
		tmp := NewTensor(NewShape(new_shape))
		tmp.data[0] = scale
		t1_loss.Mul(loss.Mul(tmp))
	}
	m.t1.node.Backward(t1_loss)
}

func (m *meanBackward) IsLeaf() bool {
	return false
}
