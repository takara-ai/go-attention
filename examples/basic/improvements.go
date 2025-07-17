package main

import (
	"fmt"
	"log"
	"time"

	"github.com/takara-ai/go-attention/attention"
	"github.com/takara-ai/go-attention/transformer"
)

func main() {
	fmt.Println("=== Go-Attention Micro Improvements Demo ===\n")

	// 1. Demonstrate performance hints and better error messages
	demoPerformanceHints()

	// 2. Demonstrate new helper functions
	demoHelperFunctions()

	// 3. Demonstrate String() methods for debugging
	demoStringMethods()

	// 4. Demonstrate validation improvements
	demoValidationImprovements()

	// 5. Demonstrate benchmarks
	demoBenchmarks()
}

func demoPerformanceHints() {
	fmt.Println("1. Performance Hints and Better Error Messages")
	fmt.Println("=============================================")

	// Show the new DotProductUnsafe function
	v1 := attention.Vector{1.0, 2.0, 3.0, 4.0}
	v2 := attention.Vector{5.0, 6.0, 7.0, 8.0}

	// Canonical version (with bounds checking)
	result, err := attention.DotProduct(v1, v2)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("DotProduct: %v\n", result)

	// Unsafe version (faster, assumes equal lengths)
	resultUnsafe := attention.DotProductUnsafe(v1, v2)
	fmt.Printf("Unsafe DotProduct: %v\n", resultUnsafe)

	// Demonstrate better error messages
	fmt.Println("\nBetter error messages:")
	badV1 := attention.Vector{1.0, 2.0}
	badV2 := attention.Vector{1.0, 2.0, 3.0}
	_, err = attention.DotProduct(badV1, badV2)
	fmt.Printf("Error: %v\n", err)

	fmt.Println()
}

func demoHelperFunctions() {
	fmt.Println("2. Helper Functions")
	fmt.Println("===================")

	// Demonstrate validateMatrixDimensions
	matrices := []attention.Matrix{
		{{1.0, 2.0}, {3.0, 4.0}},
		{{5.0, 6.0}, {7.0, 8.0}},
		{{9.0, 10.0}, {11.0, 12.0}},
	}

	err := validateMatrixDimensions(matrices...)
	if err != nil {
		fmt.Printf("Validation error: %v\n", err)
	} else {
		fmt.Println("✓ All matrices have matching dimensions")
	}

	// Demonstrate mismatched dimensions
	badMatrices := []attention.Matrix{
		{{1.0, 2.0}, {3.0, 4.0}},
		{{5.0, 6.0}}, // Different number of rows
	}

	err = validateMatrixDimensions(badMatrices...)
	if err != nil {
		fmt.Printf("✓ Caught dimension mismatch: %v\n", err)
	}

	fmt.Println()
}

func demoStringMethods() {
	fmt.Println("3. String() Methods for Debugging")
	fmt.Println("=================================")

	// MultiHeadAttention
	mhaConfig := attention.MultiHeadConfig{
		NumHeads: 4,
		DModel:   64,
		DKey:     16,
		DValue:   16,
	}
	mha, err := attention.NewMultiHeadAttention(mhaConfig)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("MultiHeadAttention: %s\n", mha.String())

	// LayerNorm
	ln := transformer.NewLayerNorm(64, 1e-5)
	fmt.Printf("LayerNorm: %s\n", ln.String())

	// FeedForward
	ff := transformer.NewFeedForward(64, 256)
	fmt.Printf("FeedForward: %s\n", ff.String())

	// TransformerLayer
	tConfig := transformer.TransformerConfig{
		DModel:      64,
		NumHeads:    4,
		DHidden:     256,
		DropoutRate: 0.1,
	}
	tLayer, err := transformer.NewTransformerLayer(tConfig)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("TransformerLayer: %s\n", tLayer.String())

	fmt.Println()
}

func demoValidationImprovements() {
	fmt.Println("4. Validation Improvements")
	fmt.Println("=========================")

	// Test various invalid configurations
	testCases := []struct {
		name string
		test func() error
	}{
		{
			name: "Zero heads",
			test: func() error {
				_, err := attention.NewMultiHeadAttention(attention.MultiHeadConfig{
					NumHeads: 0,
					DModel:   64,
					DKey:     16,
					DValue:   16,
				})
				return err
			},
		},
		{
			name: "Zero model dimension",
			test: func() error {
				_, err := attention.NewMultiHeadAttention(attention.MultiHeadConfig{
					NumHeads: 4,
					DModel:   0,
					DKey:     16,
					DValue:   16,
				})
				return err
			},
		},
		{
			name: "Not divisible",
			test: func() error {
				_, err := attention.NewMultiHeadAttention(attention.MultiHeadConfig{
					NumHeads: 3,
					DModel:   64, // 64 not divisible by 3
					DKey:     16,
					DValue:   16,
				})
				return err
			},
		},
	}

	for _, tc := range testCases {
		err := tc.test()
		if err != nil {
			fmt.Printf("✓ %s: %v\n", tc.name, err)
		} else {
			fmt.Printf("✗ %s: Expected error but got none\n", tc.name)
		}
	}

	fmt.Println()
}

func demoBenchmarks() {
	fmt.Println("5. Performance Benchmarks")
	fmt.Println("=========================")

	// Create test data
	v1 := attention.Vector{1.0, 2.0, 3.0, 4.0, 5.0}
	v2 := attention.Vector{6.0, 7.0, 8.0, 9.0, 10.0}

	// Benchmark safe vs unsafe dot product
	fmt.Println("Benchmarking DotProduct vs DotProductUnsafe:")
	
	// Safe version
	iterations := 1000000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		attention.DotProduct(v1, v2)
	}
	safeDuration := time.Since(start)

	// Unsafe version
	start = time.Now()
	for i := 0; i < iterations; i++ {
		attention.DotProductUnsafe(v1, v2)
	}
	unsafeDuration := time.Since(start)

	fmt.Printf("Safe DotProduct: %v for %d iterations\n", safeDuration, iterations)
	fmt.Printf("Unsafe DotProduct: %v for %d iterations\n", unsafeDuration, iterations)
	fmt.Printf("Speedup: %.2fx\n", float64(safeDuration)/float64(unsafeDuration))

	fmt.Println()
}

// Helper function to demonstrate validateMatrixDimensions
func validateMatrixDimensions(matrices ...attention.Matrix) error {
	if len(matrices) == 0 {
		return nil
	}
	
	firstDim := len(matrices[0])
	for i, m := range matrices {
		if len(m) != firstDim {
			return fmt.Errorf("matrix %d has %d rows, expected %d", i, len(m), firstDim)
		}
	}
	return nil
} 