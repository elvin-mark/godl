package data

type Node interface {
	IsLeaf() bool
	Backward(loss *Tensor)
}

type leafNode struct {
	t *Tensor
}

func NewLeafNode(t *Tensor) (l Node) {
	return &leafNode{
		t: t,
	}
}

func (b *leafNode) Backward(loss *Tensor) {
	if b.t.requiresGrad {
		b.t.grad.data = b.t.grad.Add(loss).data
	}
}

func (b *leafNode) IsLeaf() bool {
	return true
}
