package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/takara-ai/go-attention/attention"
)

func main() {
	fmt.Println("=== Quick Performance Test ===\n")

	// Test different vector sizes
	sizes := []int{64, 128, 256, 512, 1024, 2048, 4096}
	iterations := 10000

	for _, size := range sizes {
		v1 := makeRandomVector(size)
		v2 := makeRandomVector(size)

		fmt.Printf("Size %4d:\n", size)

		// Test canonical (best) implementation
		start := time.Now()
		for i := 0; i < iterations; i++ {
			attention.DotProduct(v1, v2)
		}
		bestTime := time.Since(start)

		fmt.Printf("  DotProduct: %8v\n", bestTime)
		fmt.Println()
	}
}

func makeRandomVector(size int) attention.Vector {
	v := make(attention.Vector, size)
	for i := range v {
		v[i] = rand.Float64()*2 - 1
	}
	return v
} 