package data

type conv2dBackward struct {
	t1      *Tensor
	t2      *Tensor
	stride  []int
	padding []int
	result  *Tensor
}

func NewConv2dBackward(t1 *Tensor, t2 *Tensor, stride []int, padding []int, result *Tensor) Node {
	return &conv2dBackward{
		t1:      t1,
		t2:      t2,
		stride:  stride,
		padding: padding,
		result:  result,
	}
}

func (ab *conv2dBackward) Backward(loss *Tensor) {

}

func (ab *conv2dBackward) IsLeaf() bool {
	return false
}
