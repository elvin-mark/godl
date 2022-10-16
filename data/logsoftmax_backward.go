package data

type logSoftmaxBackward struct {
	t1     *Tensor
	result *Tensor
}

func NewLogSoftmaxBackward(t1 *Tensor, result *Tensor) Node {
	return &logSoftmaxBackward{
		t1:     t1,
		result: result,
	}
}

func (m *logSoftmaxBackward) Backward(loss *Tensor) {

}

func (m *logSoftmaxBackward) IsLeaf() bool {
	return false
}
