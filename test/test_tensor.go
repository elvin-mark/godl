package test

import (
	data "github.com/elvin-mark/godl/data"
)

func TestSum() {
	t1 := data.NewTensor(data.NewShape([]int{2, 3, 4}))
	// t1.Rand(0, 1)
	t1.SetData([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24})
	t1.Print()
	o1 := t1.Sum()
	o1.Print()
	o2 := t1.SumByAxis([]int{1, 2})
	o2.Print()
}

func TestConv1d() {
	t1 := data.NewTensor(data.NewShape([]int{1, 2, 3}))
	t2 := data.NewTensor(data.NewShape([]int{3, 2, 2}))
	t1.Rand(0, 1)
	t2.Rand(0, 1)
	s := t1.Conv1d(t2, 1, 0)
	s.Print()
}

func TestConv2d() {
	t1 := data.NewTensor(data.NewShape([]int{1, 3, 4, 4}))
	t2 := data.NewTensor(data.NewShape([]int{5, 3, 2, 2}))
	t1.Rand(0, 1)
	t2.Rand(0, 1)
	s := t1.Conv2d(t2, []int{1, 1}, []int{0, 0})
	s.Print()
}

func TestMaxPool1d() {
	t1 := data.NewTensor(data.NewShape([]int{1, 2, 4}))
	t1.Rand(0, 1)
	t1.Print()
	s := t1.MaxPool1d(2)
	s.Print()
}

func TestMaxPool2d() {
	t1 := data.NewTensor(data.NewShape([]int{1, 3, 4, 4}))
	t1.Rand(0, 1)
	t1.Print()
	s := t1.MaxPool2d([]int{2, 2})
	s.Print()
}

func TestDropout1d() {
	t1 := data.NewTensor(data.NewShape([]int{1, 3, 4}))
	t1.Rand(0, 1)
	t1.Print()
	s := t1.Dropout1d(0.3)
	s.Print()
}

func TestDropout2d() {
	t1 := data.NewTensor(data.NewShape([]int{1, 3, 4, 4}))
	t1.Rand(0, 1)
	t1.Print()
	s := t1.Dropout2d(0.3)
	s.Print()
}

func TestTensor() {
	TestSum()
	// TestConv1d()
	// TestConv2d()
	// TestMaxPool1d()
	// TestMaxPool2d()
	// TestDropout1d()
	// TestDropout2d()
}
