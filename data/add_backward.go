package data

type addBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewAddBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Node {
	return &addBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *addBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	t2_loss := NewTensor(ab.t2.shape)

	if t1_loss.shape.Equals(t2_loss.shape) {
		for i, val := range loss.data {
			t1_loss.data[i] += val
			t2_loss.data[i] += val
		}
	} else {
		idxs := NewIndices(len(loss.stride.data))
		t1_loss.SetBroadcast()
		t2_loss.SetBroadcast()
		for _, val := range loss.data {
			t1_loss.data[t1_loss.stride.GetIndex(idxs)] += val
			t2_loss.data[t2_loss.stride.GetIndex(idxs)] += val
			idxs.Increment(loss.shape)
		}
		t1_loss.UnsetBroadcast()
		t2_loss.UnsetBroadcast()
	}
	ab.t1.node.Backward(t1_loss)
	ab.t2.node.Backward(t2_loss)
}

func (ab *addBackward) IsLeaf() bool {
	return false
}
