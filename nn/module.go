package nn

import "godl/data"

type Module interface {
	Forward(inp *data.Tensor) (out *data.Tensor)
	GetWeights() (weights []*data.Tensor)
	Train()
	Eval()
}

type Sequential interface {
	AddModule(m Module)
	Forward(inp *data.Tensor) (out *data.Tensor)
	GetWeights() (weights []*data.Tensor)
	Train()
	Eval()
}
