package data

import "fmt"

type Shape struct {
	data []int
}

func NewShape(data []int) (s *Shape) {
	s = &Shape{
		data: data,
	}
	return
}

func (s *Shape) String() string {
	out := "[ "
	for _, v := range s.data {
		out += fmt.Sprintf("%d ", v)
	}
	out += "]"
	return out
}

func (s *Shape) Size() int {
	sz := 1
	for _, elem := range s.data {
		sz *= elem
	}
	return sz
}

func (s *Shape) Equals(r *Shape) bool {
	ns := len(s.data)
	nr := len(r.data)
	if ns != nr {
		return false
	}
	for i := 0; i < ns; i++ {
		if s.data[i] != r.data[i] {
			return false
		}
	}
	return true
}

func (s *Shape) Like(r *Shape) *Shape {
	ns := len(s.data)
	nr := len(r.data)
	if ns != nr {
		return nil
	}
	new_shape_ := make([]int, ns)
	for i := 0; i < ns; i++ {
		if s.data[i] != r.data[i] && s.data[i] != 1 && r.data[i] != 1 {
			return nil
		}
		if s.data[i] != 1 {
			new_shape_[i] = s.data[i]
		} else {
			new_shape_[i] = r.data[i]
		}
	}
	return NewShape(new_shape_)
}
