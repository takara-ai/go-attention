// This script demonstrates all key APIs of the go-attention module
package main

import (
    "fmt"
    "log"
    "github.com/takara-ai/go-attention/attention"
    "github.com/takara-ai/go-attention/transformer"
)

func main() {
    // 1. Basic Dot-Product Attention
    fmt.Println("\n=== 1. Basic Dot-Product Attention ===")
    testDotProductAttention()

    // 2. Multi-Head Attention
    fmt.Println("\n=== 2. Multi-Head Attention ===")
    testMultiHeadAttention()

    // 3. Full Transformer Layer
    fmt.Println("\n=== 3. Full Transformer Layer ===")
    testTransformerLayer()
}

func testDotProductAttention() {
    // Create a simple query-key-value setup
    query := attention.Vector{1.0, 0.0, 1.0, 0.0}  // Looking for patterns similar to [1,0,1,0]
    keys := attention.Matrix{
        {1.0, 0.0, 1.0, 0.0},  // Similar to query
        {0.0, 1.0, 0.0, 1.0},  // Different from query
        {0.5, 0.5, 0.5, 0.5},  // Neutral pattern
    }
    values := attention.Matrix{
        {1.0, 2.0},  // Value for similar key
        {3.0, 4.0},  // Value for different key
        {5.0, 6.0},  // Value for neutral key
    }

    // Compute attention
    output, weights, err := attention.DotProductAttention(query, keys, values)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("Query:", query)
    fmt.Println("Attention Weights:", weights)
    fmt.Println("Output:", output)
}

func testMultiHeadAttention() {
    // Configure multi-head attention
    config := attention.MultiHeadConfig{
        NumHeads:    4,
        DModel:      64,
        DKey:        16,  // DModel / NumHeads
        DValue:      16,  // DModel / NumHeads
        DropoutRate: 0.1,
    }

    // Create multi-head attention module
    mha, err := attention.NewMultiHeadAttention(config)
    if err != nil {
        log.Fatal(err)
    }

    // Create sample input with dimensions:
    // - batch_size: number of sequences to process in parallel
    // - seq_len: number of tokens in each sequence
    // - d_model: dimension of each token's embedding
    batchSize, seqLen := 2, 3
    
    // Create input matrices with shape [batchSize × seqLen × DModel]
    queries := make(attention.Matrix, batchSize*seqLen)
    keys := make(attention.Matrix, batchSize*seqLen)
    values := make(attention.Matrix, batchSize*seqLen)
    
    // Initialize with a deterministic pattern
    for i := 0; i < batchSize*seqLen; i++ {
        queries[i] = make(attention.Vector, config.DModel)
        keys[i] = make(attention.Vector, config.DModel)
        values[i] = make(attention.Vector, config.DModel)
        for j := 0; j < config.DModel; j++ {
            val := float64(i*j) / float64(config.DModel)
            queries[i][j] = val
            keys[i][j] = val
            values[i][j] = val
        }
    }

    // Process through multi-head attention
    output, err := mha.Forward(queries, keys, values)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Input dimensions: [%d batches × %d tokens × %d features]\n", 
        batchSize, seqLen, config.DModel)
    fmt.Printf("Output shape: [%d×%d]\n", len(output), len(output[0]))
    fmt.Println("First few output values:", output[0][:4])
}

func testTransformerLayer() {
    // Configure transformer layer
    config := transformer.TransformerConfig{
        DModel:      64,
        NumHeads:    4,
        DHidden:     256,
        DropoutRate: 0.1,
    }

    // Create transformer layer
    layer, err := transformer.NewTransformerLayer(config)
    if err != nil {
        log.Fatal(err)
    }

    // Create sample sequence (seq_len=3, d_model=64)
    input := createRandomMatrix(3, config.DModel)

    // Process through transformer
    output, err := layer.Forward(input)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Input shape: [%d×%d]\n", len(input), len(input[0]))
    fmt.Printf("Output shape: [%d×%d]\n", len(output), len(output[0]))
    fmt.Println("First token before:", input[0][:4])
    fmt.Println("First token after:", output[0][:4])
}

// Helper function to create random matrices
func createRandomMatrix(rows, cols int) attention.Matrix {
    matrix := make(attention.Matrix, rows)
    for i := range matrix {
        matrix[i] = make(attention.Vector, cols)
        for j := range matrix[i] {
            matrix[i][j] = float64(i+j) / float64(cols) // Deterministic pattern for testing
        }
    }
    return matrix
} 