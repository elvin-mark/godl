package data

type powBackward struct {
	t1     *Tensor
	exp    float64
	result *Tensor
}

func NewPowBackward(t1 *Tensor, exp float64, result *Tensor) Node {
	return &powBackward{
		t1:     t1,
		exp:    exp,
		result: result,
	}
}

func (ab *powBackward) Backward(loss *Tensor) {

}

func (ab *powBackward) IsLeaf() bool {
	return false
}
