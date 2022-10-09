package test

import (
	"fmt"
	"godl/data"
)

func TestStride() {
	sh := data.NewShape([]int{1, 2, 3, 4})
	st := data.StrideFromShape(sh)
	si := data.NewIndices(4)
	fmt.Println(sh, st)
	fmt.Println(si)
	for i := 0; i < 10; i++ {
		si.Increment(sh)
		fmt.Println(si, st.GetIndex(si))
	}
}
