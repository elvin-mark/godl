package data

import (
	"fmt"
)

type Indices struct {
	data []int
}

func NewIndices(n int) (i *Indices) {
	data := make([]int, n)
	for k := 0; k < n; k++ {
		data[k] = 0
	}
	return &Indices{
		data: data,
	}
}

func (i *Indices) String() string {
	out := "[ "
	for _, v := range i.data {
		out += fmt.Sprintf("%d ", v)
	}
	out += "]"
	return out
}
func (i *Indices) Increment(s *Shape) {
	ni := len(i.data)
	for j := ni - 1; j >= 0; j-- {
		i.data[j] = (i.data[j] + 1) % s.data[j]
		if i.data[j] != 0 {
			break
		}
	}
}
