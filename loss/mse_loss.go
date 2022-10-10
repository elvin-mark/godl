package loss

import "godl/data"

type mseLoss struct {
}

func NewMSELoss() Loss {
	return &mseLoss{}
}

func (m *mseLoss) Criterion(inp *data.Tensor, target *data.Tensor) *data.Tensor {
	return inp.Sub(target).Pow(2.0).Mean()
}
