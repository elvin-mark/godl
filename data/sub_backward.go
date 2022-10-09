package data

type subBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewSubBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Node {
	return &subBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *subBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	t2_loss := NewTensor(ab.t2.shape)

	if t1_loss.shape.Equals(t2_loss.shape) {
		for i, val := range loss.data {
			t1_loss.data[i] += val
			t2_loss.data[i] += -val
		}
	} else {
		idxs := NewIndices(len(loss.stride.data))
		t1_loss.SetBroadcast()
		t2_loss.SetBroadcast()
		for _, val := range loss.data {
			i := t1_loss.stride.GetIndex(idxs)
			j := t2_loss.stride.GetIndex(idxs)
			t1_loss.data[i] += val
			t2_loss.data[j] += -val
			idxs.Increment(loss.shape)
		}
		t1_loss.UnsetBroadcast()
		t2_loss.UnsetBroadcast()
	}
	ab.t1.node.Backward(t1_loss)
	ab.t2.node.Backward(t2_loss)
}

func (ab *subBackward) IsLeaf() bool {
	return false
}
