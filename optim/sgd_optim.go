package optim

import (
	"godl/data"
)

type sgdOptim struct {
	weights []*data.Tensor
	lr      []*data.Tensor
}

func NewSGDOptimizer(weights []*data.Tensor, hyperParameters map[string]any) Optimizer {
	lr := []*data.Tensor{}
	if val, ok := hyperParameters["lr"]; ok {
		for _, weight := range weights {
			shapeData := make([]int, weight.GetDim())
			for j := 0; j < weight.GetDim(); j++ {
				shapeData[j] = 1
			}
			lr_ := data.NewTensor(data.NewShape(shapeData))
			lr_.SetData([]float64{val.(float64)})
			lr = append(lr, lr_)
		}
	}

	return &sgdOptim{
		weights: weights,
		lr:      lr,
	}
}

func (s *sgdOptim) Step() {
	for i, weight := range s.weights {
		weight.SetData(weight.Sub(weight.GetGrad().Mul(s.lr[i])).GetData())
	}
}

func (s *sgdOptim) ZeroGrad() {
	for _, weight := range s.weights {
		weight.GetGrad().ZeroGrad()
	}
}
