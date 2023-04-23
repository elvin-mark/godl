package nn

import (
	"github.com/elvin-mark/godl/data"
)

type sequential struct {
	modules []Module
}

func NewSequential() (s Sequential) {
	return &sequential{
		modules: []Module{},
	}
}

func (s *sequential) AddModule(m Module) {
	s.modules = append(s.modules, m)
}

func (s *sequential) Forward(inp *data.Tensor) (out *data.Tensor) {
	out = inp
	for _, module := range s.modules {
		out = module.Forward(out)
	}
	return
}

func (s *sequential) GetWeights() (weights []*data.Tensor) {
	weights = []*data.Tensor{}
	for _, module := range s.modules {
		weights = append(weights, module.GetWeights()...)
	}
	return
}

func (s *sequential) Train() {
	for _, module := range s.modules {
		module.Train()
	}
}

func (s *sequential) Eval() {
	for _, module := range s.modules {
		module.Eval()
	}
}
