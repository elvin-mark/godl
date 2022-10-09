package test

import (
	data "godl/data"
)

func TestTensor() {
	s1 := data.NewShape([]int{1, 3})
	s2 := data.NewShape([]int{3, 1})
	t1 := data.NewTensor(s1)
	t2 := data.NewTensor(s2)
	t1.Rand(0, 2)
	t2.Rand(0, 4)
	t1.Print()
	t2.Print()
	t1.Mm(t2).Print()
}
