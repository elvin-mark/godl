package models

import "github.com/elvin-mark/godl/nn"

func NewMLP(inp, hidden, out int) nn.Module {
	s := nn.NewSequential()
	s.AddModule(nn.NewLinear(inp, hidden, true))
	s.AddModule(nn.NewSigmoid())
	s.AddModule(nn.NewLinear(hidden, out, true))
	return s
}
