package data

type conv1dBackward struct {
	t1      *Tensor
	t2      *Tensor
	stride  int
	padding int
	result  *Tensor
}

func NewConv1dBackward(t1 *Tensor, t2 *Tensor, stride int, padding int, result *Tensor) Node {
	return &conv1dBackward{
		t1:      t1,
		t2:      t2,
		stride:  stride,
		padding: padding,
		result:  result,
	}
}

func (cb *conv1dBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(cb.t1.shape)
	t2_loss := NewTensor(cb.t2.shape)
	N := cb.t1.shape.data[0]
	Cin := cb.t1.shape.data[1]
	L := cb.t1.shape.data[2]
	Cout := cb.t2.shape.data[0]
	K := cb.t2.shape.data[2]
	new_L := cb.result.shape.data[2]

	t1_idx := NewIndices(3)
	t2_idx := NewIndices(3)
	loss_idx := NewIndices(3)
	loss_t1_idx := NewIndices(3)
	loss_t2_idx := NewIndices(3)

	var raw_k int
	// Get the gradient of the loss with respect to the input
	for i := 0; i < N; i++ {
		loss_idx.data[0] = i
		loss_t1_idx.data[0] = i
		for j := 0; j < Cin; j++ {
			t2_idx.data[1] = j
			loss_t1_idx.data[1] = j
			for k := 0; k < L; k++ {
				sum := 0.
				loss_t1_idx.data[2] = k
				for p := 0; p < Cout; p++ {
					t2_idx.data[0] = p
					loss_idx.data[1] = p
					for q := 0; q < new_L; q++ {
						raw_k = k + cb.padding - q*cb.stride
						t2_idx.data[2] = raw_k
						loss_idx.data[2] = q
						if raw_k >= 0 && raw_k < K {
							sum += cb.t2.data[cb.t2.stride.GetIndex(t2_idx)] * loss.data[loss.stride.GetIndex(loss_idx)]
						}
					}
				}
				t1_loss.data[t1_loss.stride.GetIndex(loss_t1_idx)] = sum
			}
		}
	}
	// Get the gradient of the loss with respect to the weight
	for i := 0; i < Cout; i++ {
		loss_idx.data[1] = i
		loss_t2_idx.data[0] = i
		for j := 0; j < Cin; j++ {
			t1_idx.data[1] = j
			loss_t2_idx.data[1] = j
			for k := 0; k < K; k++ {
				sum := 0.
				loss_t2_idx.data[2] = k
				for p := 0; p < N; p++ {
					t1_idx.data[0] = p
					loss_idx.data[0] = p
					for q := 0; q < new_L; q++ {
						raw_k := q*cb.stride + k - cb.padding
						t1_idx.data[2] = raw_k
						loss_idx.data[2] = q
						if raw_k >= 0 && raw_k < L {
							sum += cb.t1.data[cb.t1.stride.GetIndex(t1_idx)] * loss.data[loss.stride.GetIndex(loss_idx)]
						}
					}
				}
				t2_loss.data[t2_loss.stride.GetIndex(loss_t2_idx)] = sum
			}
		}
	}

	cb.t1.node.Backward(t1_loss)
	cb.t2.node.Backward(t2_loss)
}

func (ab *conv1dBackward) IsLeaf() bool {
	return false
}
