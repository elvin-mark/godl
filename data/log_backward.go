package data

type logBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewLogBackward(t1 *Tensor, result *Tensor) Backward {
	return &logBackward{
		t1:     t1,
		result: result,
	}
}

func (ab *logBackward) Backward(loss *Tensor) {

}
