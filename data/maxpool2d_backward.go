package data

type maxPool2dBackward struct {
	t1     *Tensor
	kernel []int
	result *Tensor
}

func NewMaxPool2dBackward(t1 *Tensor, kernel []int, result *Tensor) Node {
	return &maxPool2dBackward{
		t1:     t1,
		kernel: kernel,
		result: result,
	}
}

func (ab *maxPool2dBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)

	N := ab.t1.shape.data[0]
	C := ab.t1.shape.data[1]
	H := (ab.t1.shape.data[2] / ab.kernel[0]) * ab.kernel[0]
	W := (ab.t1.shape.data[3] / ab.kernel[1]) * ab.kernel[1]

	t1_idx := NewIndices(4)
	result_idx := NewIndices(4)
	t1_loss.Zeros()
	for i := 0; i < N; i++ {
		t1_idx.data[0] = i
		result_idx.data[0] = i
		for j := 0; j < C; j++ {
			t1_idx.data[1] = j
			result_idx.data[1] = j
			for k := 0; k < H; k++ {
				t1_idx.data[2] = k
				result_idx.data[2] = k / ab.kernel[0]
				for l := 0; l < W; l++ {
					t1_idx.data[3] = l
					result_idx.data[3] = l / ab.kernel[1]
					if ab.t1.data[ab.t1.stride.GetIndex(t1_idx)] == ab.result.data[ab.result.stride.GetIndex(result_idx)] {
						t1_loss.data[t1_loss.stride.GetIndex(t1_idx)] = loss.data[loss.stride.GetIndex(result_idx)]
					}
				}
			}
		}
	}
	ab.t1.node.Backward(t1_loss)
}

func (ab *maxPool2dBackward) IsLeaf() bool {
	return false
}
