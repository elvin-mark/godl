package data

type Backward interface {
	Backward(loss *Tensor)
}
