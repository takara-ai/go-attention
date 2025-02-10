package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/takara-ai/go-attention/attention"
	"github.com/takara-ai/go-attention/transformer"
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Create a transformer layer configuration
	config := transformer.TransformerConfig{
		DModel:      64,   // Model dimension
		NumHeads:    4,    // Number of attention heads
		DHidden:     256,  // Hidden dimension in feed-forward network
		DropoutRate: 0.1,  // Dropout rate (not used in this implementation)
	}

	// Create a transformer layer
	layer, err := transformer.NewTransformerLayer(config)
	if err != nil {
		log.Fatalf("Failed to create transformer layer: %v", err)
	}

	// Create sample input sequence
	// In this example, we'll create a sequence of 3 tokens, each with dimension DModel
	input := make(attention.Matrix, 3)
	for i := range input {
		input[i] = make(attention.Vector, config.DModel)
		// Initialize with random values
		for j := range input[i] {
			input[i][j] = rand.Float64()*2 - 1
		}
	}

	// Process the sequence through the transformer layer
	output, err := layer.Forward(input)
	if err != nil {
		log.Fatalf("Failed to process sequence: %v", err)
	}

	// Print results
	fmt.Println("Input sequence:")
	for i, vec := range input {
		fmt.Printf("Token %d: First 4 values: %v\n", i, vec[:4])
	}

	fmt.Println("\nTransformed sequence:")
	for i, vec := range output {
		fmt.Printf("Token %d: First 4 values: %v\n", i, vec[:4])
	}

	// Demonstrate multi-head attention separately
	fmt.Println("\nDemonstrating Multi-Head Attention:")
	
	// Create input vectors with correct dimensions
	batchSize := 2
	
	queries := make(attention.Matrix, batchSize)
	keys := make(attention.Matrix, batchSize)
	values := make(attention.Matrix, batchSize)
	
	// Initialize with some random values
	for b := 0; b < batchSize; b++ {
		// Create query sequence
		queries[b] = make(attention.Vector, config.DModel)
		for j := range queries[b] {
			queries[b][j] = rand.Float64()*2 - 1
		}
		
		// Create key sequence
		keys[b] = make(attention.Vector, config.DModel)
		for j := range keys[b] {
			keys[b][j] = rand.Float64()*2 - 1
		}
		
		// Create value sequence
		values[b] = make(attention.Vector, config.DModel)
		for j := range values[b] {
			values[b][j] = rand.Float64()*2 - 1
		}
	}

	// Create multi-head attention
	mha, err := attention.NewMultiHeadAttention(attention.MultiHeadConfig{
		NumHeads:    4,
		DModel:      config.DModel,
		DKey:        config.DModel / 4,
		DValue:      config.DModel / 4,
		DropoutRate: 0.1,
	})
	if err != nil {
		log.Fatalf("Failed to create multi-head attention: %v", err)
	}

	// Process through multi-head attention
	attended, err := mha.Forward(queries, keys, values)
	if err != nil {
		log.Fatalf("Failed to compute multi-head attention: %v", err)
	}

	fmt.Println("\nMulti-Head Attention outputs (first 4 values for each batch):")
	for b := range attended {
		fmt.Printf("Batch %d: %v\n", b, attended[b][:4])
	}
} 