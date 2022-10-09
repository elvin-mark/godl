package nn

import "godl/data"

type Module interface {
	Forward(inp *data.Tensor) (out *data.Tensor)
}

type Sequential interface {
	AddModule(m Module)
	Forward(inp *data.Tensor) (out *data.Tensor)
}
