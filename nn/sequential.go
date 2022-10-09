package nn

import "godl/data"

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
	for _, layer := range s.modules {
		out = layer.Forward(out)
	}
	return
}
