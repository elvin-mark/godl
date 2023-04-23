package data

type nllBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewNLLBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Node {
	return &nllBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (m *nllBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(m.t1.shape)
	batch_size := m.t1.shape.data[0]

	for i := 0; i < batch_size; i++ {
		for j := 0; j < m.t2.Size()/batch_size; j++ {
			t1_loss.data[i*t1_loss.stride.data[0]+int(m.t2.data[i*m.t2.stride.data[0]+j])*t1_loss.stride.data[1]+j] = -1.0 / float64(m.t2.Size())
		}
	}
	m.t1.node.Backward(t1_loss)
}

func (m *nllBackward) IsLeaf() bool {
	return false
}
