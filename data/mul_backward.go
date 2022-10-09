package data

type mulBackward struct {
	t1     *Tensor
	t2     *Tensor
	result *Tensor
}

func NewMulBackward(t1 *Tensor, t2 *Tensor, result *Tensor) Node {
	return &mulBackward{
		t1:     t1,
		t2:     t2,
		result: result,
	}
}

func (ab *mulBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	t2_loss := NewTensor(ab.t2.shape)

	if t1_loss.shape.Equals(t2_loss.shape) {
		for i, val := range loss.data {
			t1_loss.data[i] += val * ab.t2.data[i]
			t2_loss.data[i] += val * ab.t1.data[i]
		}
	} else {
		idxs := NewIndices(len(loss.stride.data))
		t1_loss.SetBroadcast()
		t2_loss.SetBroadcast()
		for _, val := range loss.data {
			i := t1_loss.stride.GetIndex(idxs)
			j := t2_loss.stride.GetIndex(idxs)
			t1_loss.data[i] += val * ab.t2.data[j]
			t2_loss.data[j] += val * ab.t1.data[i]
			idxs.Increment(loss.shape)
		}
		t1_loss.UnsetBroadcast()
		t2_loss.UnsetBroadcast()
	}
	ab.t1.node.Backward(t1_loss)
	ab.t2.node.Backward(t2_loss)
}

func (ab *mulBackward) IsLeaf() bool {
	return false
}
