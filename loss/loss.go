package loss

import "github.com/elvin-mark/godl/data"

type Loss interface {
	Criterion(inp *data.Tensor, target *data.Tensor) *data.Tensor
}
