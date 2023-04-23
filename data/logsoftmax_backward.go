package data

import "math"

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
	t1_loss := NewTensor(m.t1.shape)
	num_batch := t1_loss.shape.data[0]
	num_classes := t1_loss.shape.data[1]

	for i := 0; i < m.result.shape.data[0]; i++ {
		for j := 0; j < m.result.Size()/(num_batch*num_classes); j++ {
			sum := 0.
			for k := 0; k < num_classes; k++ {
				t1_loss.data[i*m.t1.stride.data[0]+k*m.t1.stride.data[1]+j] = loss.data[i*m.t1.stride.data[0]+k*m.t1.stride.data[1]+j]
				sum += t1_loss.data[i*m.t1.stride.data[0]+k*m.t1.stride.data[1]+j]
			}
			for k := 0; k < num_classes; k++ {
				t1_loss.data[i*m.t1.stride.data[0]+k*m.t1.stride.data[1]+j] -= math.Exp(m.result.data[i*m.t1.stride.data[0]+k*m.t1.stride.data[1]+j]) * sum
			}
		}
	}
	m.t1.node.Backward(t1_loss)
}

func (m *logSoftmaxBackward) IsLeaf() bool {
	return false
}
