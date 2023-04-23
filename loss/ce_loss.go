package loss

import "github.com/elvin-mark/godl/data"

type ceLoss struct {
}

func NewCELoss() Loss {
	return &ceLoss{}
}

func (m *ceLoss) Criterion(inp *data.Tensor, target *data.Tensor) *data.Tensor {
	return inp.LogSoftmax().Nll(target)
}
