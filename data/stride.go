package data

import "fmt"

type Stride struct {
	data []int
}

func NewStride(data []int) (s *Stride) {
	return &Stride{
		data: data,
	}
}

func StrideFromShape(shape *Shape) (s *Stride) {
	n := len(shape.data)
	data := make([]int, n)

	i := n - 1
	acc := 1
	for i >= 0 {
		data[i] = acc
		acc *= shape.data[i]
		i -= 1
	}

	return &Stride{
		data: data,
	}
}

func (s *Stride) String() string {
	out := "[ "
	for _, v := range s.data {
		out += fmt.Sprintf("%d ", v)
	}
	out += "]"
	return out
}

func (s *Stride) GetIndex(i *Indices) int {
	ns := len(s.data)
	ni := len(i.data)
	if ns != ni {
		panic("Stride and Indices does not have same length")
	}
	idx := 0
	for j, val := range s.data {
		idx += val * i.data[j]
	}
	return idx
}
