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

func (t *Tensor) String() string {

	return ""
}

func (t *Tensor) Print() {
	fmt.Println(t.data)
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

func (t *Tensor) Item() float64 {
	if len(t.data) == 0 {
		panic("no data")
	}
	return t.data[0]
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
	}
	r.node = NewAddBackward(t, s, r)
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
	}
	r.node = NewSubBackward(t, s, r)
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
	}
	r.node = NewMulBackward(t, s, r)
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
	}
	r.node = NewDivBackward(t, s, r)
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
	}
	r.node = NewMmBackward(t, s, r)
	return
}

func (t *Tensor) Pow(exp float64) (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Pow(val, exp)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewPowBackward(t, exp, r)
	return
}

func (t *Tensor) Log() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Log(val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewLogBackward(t, r)
	return
}

func (t *Tensor) Exp() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Exp(val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewExpBackward(t, r)
	return
}

func (t *Tensor) Sigmoid() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = 1 / (1 + math.Exp(-val))
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewSigmoidBackward(t, r)
	return
}

func (t *Tensor) Tanh() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Tanh(val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewTanhBackward(t, r)
	return
}

func (t *Tensor) ReLU() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Max(0, val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewReLUBackward(t, r)
	return
}

func (t *Tensor) LeakyReLU(alpha float64) (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = math.Max(alpha*val, val)
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewLeakyReLUBackward(t, alpha, r)
	return
}

func (t *Tensor) SiLU() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = val / (1 + math.Exp(-val))
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewSiLUBackward(t, r)
	return
}

func (t *Tensor) GeLU() (r *Tensor) {
	r = NewTensor(t.shape)
	for i, val := range t.data {
		r.data[i] = val / (1 + math.Exp(-1.702*val))
	}
	if t.requiresGrad {
		r.SetRequiresGrad(true)
	}
	r.node = NewGeLUBackward(t, r)
	return
}
