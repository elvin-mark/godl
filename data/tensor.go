package data

import (
	"fmt"
	"math"
	"math/rand"
)

type Tensor struct {
	data         []float64
	shape        *Shape
	stride       *Stride
	grad         *Tensor
	node         Node
	requiresGrad bool
}

func NewTensor(shape *Shape) (t *Tensor) {
	stride := StrideFromShape(shape)
	data := make([]float64, shape.Size())
	t = &Tensor{
		data:         data,
		shape:        shape,
		stride:       stride,
		requiresGrad: false,
	}
	t.node = NewLeafNode(t)
	return
}

func (t *Tensor) Clone() (r *Tensor) {
	r = &Tensor{
		data:         t.data,
		stride:       t.stride,
		shape:        t.shape,
		requiresGrad: t.requiresGrad,
	}
	r.node = NewLeafNode(r)
	return
}

func (t *Tensor) String() string {

	return ""
}

func (t *Tensor) Print() {
	fmt.Println("Stride: ", t.stride.data)
	fmt.Println("Shape: ", t.shape.data)
	fmt.Println("RequiresGrad: ", t.requiresGrad)
	fmt.Println("Data: ", t.data)
}

func (t *Tensor) Size() int {
	return t.shape.Size()
}

func (t *Tensor) SetBroadcast() {
	for i, val := range t.shape.data {
		if val == 1 {
			t.stride.data[i] = 0
		}
	}
}

func (t *Tensor) UnsetBroadcast() {
	t.stride = StrideFromShape(t.shape)
}

func (t *Tensor) Zeros() {
	for i := 0; i < len(t.data); i++ {
		t.data[i] = 0
	}
}

func (t *Tensor) Ones() {
	for i := 0; i < len(t.data); i++ {
		t.data[i] = 1
	}
}

func (t *Tensor) Rand(minVal float64, maxVal float64) {
	for i := 0; i < len(t.data); i++ {
		t.data[i] = rand.Float64()*(maxVal-minVal) + minVal
	}
}

func (t *Tensor) RandN(mean float64, std float64) {
	for i := 0; i < len(t.data); i++ {
		t.data[i] = rand.NormFloat64()*std + mean
	}
}

func (t *Tensor) SetRequiresGrad(requiresGrad bool) {
	t.requiresGrad = requiresGrad
	if requiresGrad && t.grad == nil {
		t.grad = NewTensor(t.shape)
	}
}

func (t *Tensor) SetGrad(grad *Tensor) {
	t.grad = grad
}

func (t *Tensor) GetGrad() *Tensor {
	return t.grad
}

func (t *Tensor) ZeroGrad() {
	if t.grad != nil {
		t.grad.Zeros()
	}
}

func (t *Tensor) SetData(data []float64) {
	if len(data) != t.shape.Size() {
		panic("Length of data does not match tensor size")
	}
	t.data = data
}

func (t *Tensor) GetData() []float64 {
	return t.data
}

func (t *Tensor) GetDim() int {
	return len(t.shape.data)
}

func (t *Tensor) Item() float64 {
	if len(t.data) == 0 {
		panic("no data")
	}
	return t.data[0]
}

func (t *Tensor) Backward(loss *Tensor) {
	if t.requiresGrad {
		if loss == nil {
			loss = NewTensor(NewShape([]int{1}))
			loss.data[0] = 1.0
		}
		t.node.Backward(loss)
	}
}

func (t *Tensor) Transpose() (r *Tensor) {
	// TODO: Add Transpose Backward
	r = t.Clone()
	n := len(t.stride.data)
	newStrideData := make([]int, n)
	newShapeData := make([]int, n)
	for i := 0; i < n; i++ {
		newStrideData[i] = r.stride.data[n-i-1]
		newShapeData[i] = r.shape.data[n-i-1]
	}
	r.stride = NewStride(newStrideData)
	r.shape = NewShape(newShapeData)
	return
}

func (t *Tensor) Sum() (r *Tensor) {
	r = NewTensor(NewShape([]int{1}))
	for i := 0; i < t.Size(); i++ {
		r.data[0] += t.data[i]
	}
	return
}

func (t *Tensor) SumByAxis(axis []int) (r *Tensor) {
	// No gradients yet
	if len(axis) > t.GetDim() {
		panic("tensor index size mismatch")
	}

	new_shape := make([]int, t.GetDim())

	for i := 0; i < t.GetDim(); i++ {
		new_shape[i] = t.shape.data[i]
	}

	for i := 0; i < len(axis); i++ {
		new_shape[axis[i]] = 1
	}

	r = NewTensor(NewShape(new_shape))
	r.Zeros()
	indices := NewIndices(t.GetDim())
	r.SetBroadcast()
	for i := 0; i < t.Size(); i++ {
		r.data[r.stride.GetIndex(indices)] += t.data[i]
		indices.Increment(t.shape)
	}
	r.UnsetBroadcast()
	return
}

func (t *Tensor) Add(s *Tensor) (r *Tensor) {
	if t.shape.Equals(s.shape) {
		r = NewTensor(t.shape)
		for i, val := range t.data {
			r.data[i] = val + s.data[i]
		}
	} else if new_shape := t.shape.Like(s.shape); new_shape != nil {
		r = NewTensor(new_shape)
		idxs := NewIndices(len(r.stride.data))
		t.SetBroadcast()
		s.SetBroadcast()
		for i := 0; i < len(r.data); i++ {
			r.data[i] = t.data[t.stride.GetIndex(idxs)] + s.data[s.stride.GetIndex(idxs)]
			idxs.Increment(r.shape)
		}
		t.UnsetBroadcast()
		s.UnsetBroadcast()
	}
	if r == nil {
		return
	}
	if t.requiresGrad || s.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewAddBackward(t, s, r)
	}
	return
}

func (t *Tensor) Sub(s *Tensor) (r *Tensor) {
	if t.shape.Equals(s.shape) {
		r = NewTensor(t.shape)
		for i, val := range t.data {
			r.data[i] = val - s.data[i]
		}
	} else if new_shape := t.shape.Like(s.shape); new_shape != nil {
		r = NewTensor(new_shape)
		idxs := NewIndices(len(r.stride.data))
		t.SetBroadcast()
		s.SetBroadcast()
		for i := 0; i < len(r.data); i++ {
			r.data[i] = t.data[t.stride.GetIndex(idxs)] - s.data[s.stride.GetIndex(idxs)]
			idxs.Increment(r.shape)
		}
		t.UnsetBroadcast()
		s.UnsetBroadcast()
	}
	if r == nil {
		return
	}
	if t.requiresGrad || s.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewSubBackward(t, s, r)
	}
	return
}

func (t *Tensor) Mul(s *Tensor) (r *Tensor) {
	if t.shape.Equals(s.shape) {
		r = NewTensor(t.shape)
		for i, val := range t.data {
			r.data[i] = val * s.data[i]
		}
	} else if new_shape := t.shape.Like(s.shape); new_shape != nil {
		r = NewTensor(new_shape)
		idxs := NewIndices(len(r.stride.data))
		t.SetBroadcast()
		s.SetBroadcast()
		for i := 0; i < len(r.data); i++ {
			r.data[i] = t.data[t.stride.GetIndex(idxs)] * s.data[s.stride.GetIndex(idxs)]
			idxs.Increment(r.shape)
		}
		t.UnsetBroadcast()
		s.UnsetBroadcast()
	}
	if r == nil {
		return
	}
	if t.requiresGrad || s.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewMulBackward(t, s, r)
	}
	return
}

func (t *Tensor) Div(s *Tensor) (r *Tensor) {
	if t.shape.Equals(s.shape) {
		r = NewTensor(t.shape)
		for i, val := range t.data {
			r.data[i] = val / s.data[i]
		}
	} else if new_shape := t.shape.Like(s.shape); new_shape != nil {
		r = NewTensor(new_shape)
		idxs := NewIndices(len(r.stride.data))
		t.SetBroadcast()
		s.SetBroadcast()
		for i := 0; i < len(r.data); i++ {
			r.data[i] = t.data[t.stride.GetIndex(idxs)] / s.data[s.stride.GetIndex(idxs)]
			idxs.Increment(r.shape)
		}
		t.UnsetBroadcast()
		s.UnsetBroadcast()
	}
	if r == nil {
		return
	}
	if t.requiresGrad || s.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewDivBackward(t, s, r)
	}
	return
}

func (t *Tensor) Mm(s *Tensor) (r *Tensor) {
	if len(t.shape.data) != 2 || len(s.shape.data) != 2 {
		panic("MatrixMul: 2-dim tensors were expected")
	}
	if t.shape.data[1] != s.shape.data[0] {
		panic("MatrixMul: Mismatch dimensions")
	}
	r = NewTensor(NewShape([]int{t.shape.data[0], s.shape.data[1]}))
	for i := 0; i < r.shape.data[0]; i++ {
		for j := 0; j < r.shape.data[1]; j++ {
			tmp := 0.
			for k := 0; k < s.shape.data[0]; k++ {
				tmp += t.data[i*t.stride.data[0]+k*t.stride.data[1]] * s.data[k*s.stride.data[0]+j*s.stride.data[1]]
			}
			r.data[i*r.stride.data[0]+j*r.stride.data[1]] = tmp
		}
	}
	if r == nil {
		return
	}
	if t.requiresGrad || s.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewMmBackward(t, s, r)
	}
	return
}

func (t *Tensor) Pow(exp float64) (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Pow(val, exp)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewPowBackward(t, exp, r)
	}
	return
}

func (t *Tensor) Log() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Log(val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewLogBackward(t, r)
	}
	return
}

func (t *Tensor) Exp() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Exp(val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewExpBackward(t, r)
	}
	return
}

func (t *Tensor) Sigmoid() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = 1 / (1 + math.Exp(-val))
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewSigmoidBackward(t, r)
	}
	return
}

func (t *Tensor) Tanh() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Tanh(val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewTanhBackward(t, r)
	}
	return
}

func (t *Tensor) ReLU() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Max(0, val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewReLUBackward(t, r)
	}
	return
}

func (t *Tensor) LeakyReLU(alpha float64) (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Max(alpha*val, val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewLeakyReLUBackward(t, alpha, r)
	}
	return
}

func (t *Tensor) SiLU() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = val / (1 + math.Exp(-val))
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewSiLUBackward(t, r)
	}
	return
}

func (t *Tensor) GeLU() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = val / (1 + math.Exp(-1.702*val))
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewGeLUBackward(t, r)
	}
	return
}

// Calculate the mean
func (t *Tensor) Mean() (r *Tensor) {
	r = NewTensor(NewShape([]int{1}))
	sum := 0.
	for _, val := range t.data {
		sum += val
	}
	r.data[0] = sum / float64(t.shape.Size())
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewMeanBackward(t, r)
	}
	return
}

func (t *Tensor) MeanByAxis(axis []int) (r *Tensor) {
	s := 1.0
	r = t.SumByAxis(axis)
	for i := 0; i < len(axis); i++ {
		s *= float64(t.shape.data[axis[i]])
	}
	new_shape := make([]int, r.GetDim())
	for i := 0; i < r.GetDim(); i++ {
		new_shape[i] = 1
	}
	tmp := NewTensor(NewShape(new_shape))
	tmp.data[0] = s
	r = r.Div(tmp)

	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewMeanBackward(t, r)
	}
	return
}

func (t *Tensor) Conv1d(s *Tensor, stride int, padding int) (r *Tensor) {
	// shape of tensor: [N, Cin, L]
	// shape of kernel: [Cout, Cin, K]
	if t.GetDim() != 3 || s.GetDim() != 3 {
		panic("Conv1d: only support convolution on 3D tensor")
	}
	N := t.shape.data[0]
	Cin := t.shape.data[1]
	L := t.shape.data[2]
	Cout := s.shape.data[0]
	K := s.shape.data[2]

	var raw_k int

	if Cin != s.shape.data[1] {
		panic("Conv1d: convolution requires the number of channels of the first tensor to be equal to the number of channels of the second tensor")
	}
	new_L := (L-K+2*padding)/stride + 1
	new_shape := []int{N, Cout, new_L}
	r = NewTensor(NewShape(new_shape))
	t_idx := NewIndices(3)
	s_idx := NewIndices(3)
	r_idx := NewIndices(3)
	for i := 0; i < N; i++ {
		t_idx.data[0] = i
		r_idx.data[0] = i
		for j := 0; j < Cout; j++ {
			s_idx.data[0] = j
			r_idx.data[1] = j
			for k := 0; k < new_L; k++ {
				sum := 0.
				for p := 0; p < Cin; p++ {
					t_idx.data[1] = p
					s_idx.data[1] = p
					for q := 0; q < K; q++ {
						raw_k = k*stride + q - padding
						if raw_k >= 0 && raw_k < L {
							t_idx.data[2] = raw_k
							s_idx.data[2] = q
							sum += t.data[t.stride.GetIndex(t_idx)] * s.data[s.stride.GetIndex(s_idx)]
						}
					}
				}
				r_idx.data[2] = k
				r.data[r.stride.GetIndex(r_idx)] = sum
			}
		}
	}

	if t.requiresGrad || s.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewConv1dBackward(t, s, stride, padding, r)
	}

	return
}

func (t *Tensor) Conv2d(s *Tensor, stride []int, padding []int) (r *Tensor) {
	// shape of tensor: [N, Cin, H, W]
	// shape of kernel: [Cout, Cin, K1, K2]
	if t.GetDim() != 4 || s.GetDim() != 4 {
		panic("Conv2d: only support convolution on 4D tensor")
	}
	if t.shape.data[1] != s.shape.data[1] {
		panic("Conv2d: convolution requires the number of channels of the first tensor to be equal to the number of channels of the second tensor")
	}
	N := t.shape.data[0]
	Cin := t.shape.data[1]
	H := t.shape.data[2]
	W := t.shape.data[3]
	K1 := s.shape.data[2]
	K2 := s.shape.data[3]
	Cout := s.shape.data[0]
	new_H := (H-K1+2*padding[0])/stride[0] + 1
	new_W := (W-K2+2*padding[1])/stride[1] + 1
	new_shape := []int{N, Cout, new_H, new_W}
	var raw_h, raw_w int
	r = NewTensor(NewShape(new_shape))
	t_idx := NewIndices(4)
	s_idx := NewIndices(4)
	r_idx := NewIndices(4)
	for n := 0; n < N; n++ {
		t_idx.data[0] = n
		r_idx.data[0] = n
		for c := 0; c < Cout; c++ {
			s_idx.data[0] = c
			r_idx.data[1] = c
			for h := 0; h < new_H; h++ {
				for w := 0; w < new_W; w++ {
					sum := 0.
					for c1 := 0; c1 < Cin; c1++ {
						t_idx.data[1] = c1
						s_idx.data[1] = c1
						for k1 := 0; k1 < K1; k1++ {
							for k2 := 0; k2 < K2; k2++ {
								raw_h = h*stride[0] + k1 - padding[0]
								raw_w = w*stride[1] + k2 - padding[1]
								if raw_h >= 0 && raw_h < H && raw_w >= 0 && raw_w < W {
									t_idx.data[2] = raw_h
									t_idx.data[3] = raw_w

									s_idx.data[2] = k1
									s_idx.data[3] = k2

									sum += t.data[t.stride.GetIndex(t_idx)] * s.data[s.stride.GetIndex(s_idx)]
								}
							}
						}
					}
					r_idx.data[2] = h
					r_idx.data[3] = w
					r.data[r.stride.GetIndex(r_idx)] = sum
				}
			}
		}
	}

	if t.requiresGrad || s.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewConv2dBackward(t, s, stride, padding, r)
	}

	return
}

func (t *Tensor) MaxPool1d(kernel int) (r *Tensor) {
	// TODO: Check and Change this
	new_shape := make([]int, t.GetDim())
	new_shape[2] = t.shape.data[2] / kernel
	N := t.shape.data[0]
	Cin := t.shape.data[1]
	new_shape[0] = N
	new_shape[1] = Cin

	L := new_shape[2] * kernel

	r = NewTensor(NewShape(new_shape))
	t_idx := NewIndices(3)
	r_idx := NewIndices(3)
	for i := 0; i < N; i++ {
		t_idx.data[0] = i
		r_idx.data[0] = i
		for j := 0; j < Cin; j++ {
			t_idx.data[1] = j
			r_idx.data[1] = j
			for k := 0; k < L; k += kernel {
				t_idx.data[2] = k
				max := t.data[t.stride.GetIndex(t_idx)]
				for m := 0; m < kernel; m++ {
					t_idx.data[2] = k + m
					max = math.Max(max, t.data[t.stride.GetIndex(t_idx)])
				}
				r_idx.data[2] = k / kernel
				r.data[r.stride.GetIndex(r_idx)] = max
			}
		}
	}

	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewMaxPool1dBackward(t, kernel, r)
	}

	return
}

func (t *Tensor) MaxPool2d(kernel []int) (r *Tensor) {
	new_shape := make([]int, t.GetDim())
	new_shape[2] = t.shape.data[2] / kernel[0]
	new_shape[3] = t.shape.data[3] / kernel[1]
	N := t.shape.data[0]
	Cin := t.shape.data[1]
	new_shape[0] = N
	new_shape[1] = Cin

	H := new_shape[2] * kernel[0]
	W := new_shape[3] * kernel[1]

	r = NewTensor(NewShape(new_shape))
	t_idx := NewIndices(4)
	r_idx := NewIndices(4)
	for i := 0; i < N; i++ {
		t_idx.data[0] = i
		r_idx.data[0] = i
		for j := 0; j < Cin; j++ {
			t_idx.data[1] = j
			r_idx.data[1] = j
			for k := 0; k < H; k += kernel[0] {
				t_idx.data[2] = k
				r_idx.data[2] = k / kernel[0]
				for l := 0; l < W; l += kernel[1] {
					t_idx.data[3] = l
					r_idx.data[3] = l / kernel[1]
					max := t.data[t.stride.GetIndex(t_idx)]
					for m := 0; m < kernel[0]; m++ {
						t_idx.data[2] = k + m
						for n := 0; n < kernel[1]; n++ {
							t_idx.data[3] = l + n
							max = math.Max(max, t.data[t.stride.GetIndex(t_idx)])
						}
					}
					r.data[r.stride.GetIndex(r_idx)] = max
				}
			}
		}
	}

	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewMaxPool2dBackward(t, kernel, r)
	}

	return
}

func (t *Tensor) Dropout1d(p float64) (r *Tensor) {
	r = NewTensor(NewShape(t.shape.data))
	for i := 0; i < t.Size(); i++ {
		if rand.Float64() < p {
			r.data[i] = 0
		} else {
			r.data[i] = t.data[i] / p
		}
	}

	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewDropout1dBackward(t, p, r)
	}
	return
}

func (t *Tensor) Dropout2d(p float64) (r *Tensor) {
	r = NewTensor(NewShape(t.shape.data))
	for i := 0; i < t.Size(); i++ {
		if rand.Float64() < p {
			r.data[i] = 0
		} else {
			r.data[i] = t.data[i] / p
		}
	}

	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewDropout2dBackward(t, p, r)
	}
	return
}

func (t *Tensor) Softmax() (r *Tensor) {
	// input: [N, C , ...]
	// output: [N, C , ...]

	r = NewTensor(t.shape)
	num_batch := t.shape.data[0]
	num_classes := t.shape.data[1]

	// Loop over batches
	for i := 0; i < t.shape.data[0]; i++ {
		for j := 0; j < t.Size()/(num_batch*num_classes); j++ {
			sum := 0.
			for k := 0; k < num_classes; k++ {
				r.data[i*t.stride.data[0]+k*t.stride.data[1]+j] = math.Exp(t.data[i*t.stride.data[0]+k*t.stride.data[1]+j])
				sum += r.data[i*t.stride.data[0]+k*t.stride.data[1]+j]
			}
			for k := 0; k < num_classes; k++ {
				r.data[i*t.stride.data[0]+k*t.stride.data[1]+j] /= sum
			}
		}

	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewSoftmaxBackward(t, r)
	}
	return
}

func (t *Tensor) LogSoftmax() (r *Tensor) {
	r = NewTensor(t.shape)
	num_batch := t.shape.data[0]
	num_classes := t.shape.data[1]

	// Loop over batches
	for i := 0; i < t.shape.data[0]; i++ {
		for j := 0; j < t.Size()/(num_batch*num_classes); j++ {
			sum := 0.
			for k := 0; k < num_classes; k++ {
				r.data[i*t.stride.data[0]+k*t.stride.data[1]+j] = math.Exp(t.data[i*t.stride.data[0]+k*t.stride.data[1]+j])
				sum += r.data[i*t.stride.data[0]+k*t.stride.data[1]+j]
			}
			for k := 0; k < num_classes; k++ {
				r.data[i*t.stride.data[0]+k*t.stride.data[1]+j] = math.Log(r.data[i*t.stride.data[0]+k*t.stride.data[1]+j] / sum)
			}
		}
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewLogSoftmaxBackward(t, r)
	}
	return
}

func (t *Tensor) Nll(s *Tensor) (r *Tensor) {
	if t.GetDim() < 2 || t.GetDim() != s.GetDim() || t.shape.data[0] != s.shape.data[0] {
		panic("input and target must have the same shape")
	}

	batchSize := t.shape.data[0]
	targetStride := s.stride.data

	kl_divergence := 0.
	for i := 0; i < batchSize; i++ {
		for j := 0; j < s.Size()/batchSize; j++ {
			kl_divergence += -t.data[i*t.stride.data[0]+int(s.data[i*targetStride[0]+j])*t.stride.data[1]+j]
		}
	}

	new_shape := make([]int, t.GetDim())
	for i := 0; i < t.GetDim(); i++ {
		new_shape[i] = 1
	}
	r = NewTensor(NewShape(new_shape))
	r.data[0] = kl_divergence / float64(s.Size())
	if t.requiresGrad {
		r.SetRequiresGrad(true)
		r.node = NewNLLBackward(t, s, r)
	}
	return
}
