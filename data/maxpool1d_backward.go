package data

type maxPool1dBackward struct {
	t1     *Tensor
	kernel int
	result *Tensor
}

func NewMaxPool1dBackward(t1 *Tensor, kernel int, result *Tensor) Node {
	return &maxPool1dBackward{
		t1:     t1,
		kernel: kernel,
		result: result,
	}
}

func (ab *maxPool1dBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)

	N := ab.t1.shape.data[0]
	C := ab.t1.shape.data[1]
	L := (ab.t1.shape.data[2] / ab.kernel) * ab.kernel

	result_idx := NewIndices(3)
	loss_t1_idx := NewIndices(3)
	t1_loss.Zeros()

	// TODO: Check and Change this
	for i := 0; i < N; i++ {
		result_idx.data[0] = i
		loss_t1_idx.data[0] = i
		for j := 0; j < C; j++ {
			result_idx.data[1] = j
			loss_t1_idx.data[1] = j
			for k := 0; k < L; k++ {
				result_idx.data[2] = k
				loss_t1_idx.data[2] = k / ab.kernel
				if ab.t1.data[ab.t1.stride.GetIndex(loss_t1_idx)] == ab.result.data[ab.result.stride.GetIndex(result_idx)] {
					t1_loss.data[t1_loss.stride.GetIndex(loss_t1_idx)] = loss.data[loss.stride.GetIndex(result_idx)]
				}

			}
		}
	}

	ab.t1.node.Backward(t1_loss)
}

func (ab *maxPool1dBackward) IsLeaf() bool {
	return false
}
