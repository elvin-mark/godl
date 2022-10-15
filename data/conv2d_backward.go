package data

type conv2dBackward struct {
	t1      *Tensor
	t2      *Tensor
	stride  []int
	padding []int
	result  *Tensor
}

func NewConv2dBackward(t1 *Tensor, t2 *Tensor, stride []int, padding []int, result *Tensor) Node {
	return &conv2dBackward{
		t1:      t1,
		t2:      t2,
		stride:  stride,
		padding: padding,
		result:  result,
	}
}

func (ab *conv2dBackward) Backward(loss *Tensor) {
	t1_loss := NewTensor(ab.t1.shape)
	t2_loss := NewTensor(ab.t2.shape)

	N := ab.t1.shape.data[0]
	Cin := ab.t1.shape.data[1]
	H := ab.t1.shape.data[2]
	W := ab.t1.shape.data[3]
	K1 := ab.t2.shape.data[2]
	K2 := ab.t2.shape.data[3]
	Cout := ab.t2.shape.data[0]
	new_H := ab.result.shape.data[2]
	new_W := ab.result.shape.data[3]
	var raw_h, raw_w int

	t1_idx := NewIndices(4)
	t2_idx := NewIndices(4)
	loss_idx := NewIndices(4)
	loss_t1_idx := NewIndices(4)
	loss_t2_idx := NewIndices(4)

	// Get the gradient of the loss with respect to the input
	for n := 0; n < N; n++ {
		loss_idx.data[0] = n
		loss_t1_idx.data[0] = n
		for c := 0; c < Cin; c++ {
			t2_idx.data[1] = c
			loss_t1_idx.data[1] = n
			for h := 0; h < H; h++ {
				loss_t1_idx.data[2] = h
				for w := 0; w < W; w++ {
					sum := 0.
					loss_t1_idx.data[3] = w
					for c1 := 0; c1 < Cout; c1++ {
						loss_idx.data[1] = c1
						t2_idx.data[0] = c1
						for h1 := 0; h1 < new_H; h1++ {
							loss_idx.data[2] = h1
							for w1 := 0; w1 < new_W; w1++ {
								raw_h = h - h1*ab.stride[0] + ab.padding[0]
								raw_w = w - w1*ab.stride[1] + ab.padding[1]
								loss_idx.data[3] = w1
								t2_idx.data[2] = raw_h
								t2_idx.data[3] = raw_w
								if raw_h >= 0 && raw_h < K1 && raw_w >= 0 && raw_w < K2 {
									sum += loss.data[loss.stride.GetIndex(loss_idx)] * ab.t2.data[ab.t2.stride.GetIndex(t2_idx)]
								}
							}
						}
					}
					t1_loss.data[t1_loss.stride.GetIndex(loss_t1_idx)] = sum
				}
			}
		}
	}

	// Get the gradient of the loss with respect to the weight
	for n := 0; n < Cout; n++ {
		loss_idx.data[1] = n
		loss_t2_idx.data[0] = n
		for c := 0; c < Cin; c++ {
			t1_idx.data[1] = c
			loss_t2_idx.data[1] = c
			for h := 0; h < K1; h++ {
				loss_t2_idx.data[2] = h
				for w := 0; w < K2; w++ {
					sum := 0.
					loss_t2_idx.data[3] = w
					for c1 := 0; c1 < N; c1++ {
						loss_idx.data[0] = c1
						t1_idx.data[0] = c1
						for h1 := 0; h1 < new_H; h1++ {
							loss_idx.data[2] = h1
							for w1 := 0; w1 < new_W; w1++ {
								raw_h = h1*ab.stride[0] + h - ab.padding[0]
								raw_w = w1*ab.stride[1] + w - ab.padding[1]
								loss_idx.data[3] = w1
								t1_idx.data[2] = raw_h
								t1_idx.data[3] = raw_w
								if raw_h >= 0 && raw_h < H && raw_w >= 0 && raw_w < W {
									sum += loss.data[loss.stride.GetIndex(loss_idx)] * ab.t1.data[ab.t1.stride.GetIndex(t1_idx)]
								}
							}
						}
					}
					t2_loss.data[t2_loss.stride.GetIndex(loss_t2_idx)] = sum
				}
			}
		}
	}

	ab.t1.node.Backward(t1_loss)
	ab.t2.node.Backward(t2_loss)
}

func (ab *conv2dBackward) IsLeaf() bool {
	return false
}
