package data

type conv1dBackward struct {
	t1      *Tensor
	t2      *Tensor
	stride  int
	padding int
	result  *Tensor
}

func NewConv1dBackward(t1 *Tensor, t2 *Tensor, stride int, padding int, result *Tensor) Node {
	return &conv1dBackward{
		t1:      t1,
		t2:      t2,
		stride:  stride,
		padding: padding,
		result:  result,
	}
}

func (ab *conv1dBackward) Backward(loss *Tensor) {

}

func (ab *conv1dBackward) IsLeaf() bool {
	return false
}
