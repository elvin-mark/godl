package loss

import "godl/data"

type Loss interface {
	Criterion(inp *data.Tensor, target *data.Tensor) *data.Tensor
}
