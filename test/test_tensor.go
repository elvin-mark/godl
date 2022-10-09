package test

import (
	data "godl/data"
)

func TestTensor() {
	s1 := data.NewShape([]int{1, 1})
	s2 := data.NewShape([]int{1, 1})
	t1 := data.NewTensor(s1)
	t2 := data.NewTensor(s2)
	t3 := data.NewTensor(s2)
	t1.Rand(0, 2)
	t2.Rand(0, 4)
	t3.Ones()
	t1.Print()
	t2.Print()
	t1.SetRequiresGrad(true)
	t2.SetRequiresGrad(true)
	r := t1.Add(t2)
	r.Backward(t3)
	r.Print()
	t1.GetGrad().Print()
	t2.GetGrad().Print()
}
