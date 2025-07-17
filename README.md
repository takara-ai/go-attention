# go-attention

<img src="https://takara.ai/images/logo-24/TakaraAi.svg" width="200" alt="Takara.ai Logo" />

From the Frontier Research Team at takara.ai we present the first pure Go implementation of attention mechanisms and transformer layers, designed for high performance and ease of use.

## Quick Start

Run our comprehensive examples:

```bash
# Get the module
go get github.com/takara-ai/go-attention

# Run the examples
go run api_examples.go
```

## API Documentation

### Core Types

```go
type Vector []float64           // Represents a 1D vector of float64 values
type Matrix []Vector           // Represents a 2D matrix of float64 values
```

### 1. Basic Dot-Product Attention

The simplest form of attention mechanism. Useful for basic sequence processing tasks.

```go
import "github.com/takara-ai/go-attention/attention"

// Create query-key-value setup
query := attention.Vector{1.0, 0.0, 1.0, 0.0}  // Pattern to search for
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

// Output will be a weighted combination of values based on query-key similarity
// Weights will show how much attention each key received
```

### 2. Multi-Head Attention

More sophisticated attention mechanism that can capture different types of relationships in parallel.

```go
import "github.com/takara-ai/go-attention/attention"

// Configure multi-head attention
config := attention.MultiHeadConfig{
    NumHeads:    4,        // Number of parallel attention heads
    DModel:      64,       // Size of input/output embeddings
    DKey:        16,       // Size per head (DModel/NumHeads)
    DValue:      16,       // Size per head (DModel/NumHeads)
    DropoutRate: 0.1,      // For regularization
}

// Create the attention module
mha, err := attention.NewMultiHeadAttention(config)
if err != nil {
    log.Fatal(err)
}

// Process sequences (batched input)
batchSize, seqLen := 2, 3  // Process 2 sequences, each with 3 tokens

// Create input matrices [batchSize × seqLen × DModel]
queries := make(attention.Matrix, batchSize*seqLen)
keys := make(attention.Matrix, batchSize*seqLen)
values := make(attention.Matrix, batchSize*seqLen)

// Initialize your matrices with actual data...

// Process through multi-head attention
output, err := mha.Forward(queries, keys, values)
if err != nil {
    log.Fatal(err)
}
```

### 3. Full Transformer Layer

Complete transformer layer with self-attention and feed-forward network.

```go
import (
    "github.com/takara-ai/go-attention/transformer"
    "github.com/takara-ai/go-attention/attention"
)

// Configure transformer layer
config := transformer.TransformerConfig{
    DModel:      64,       // Size of token embeddings
    NumHeads:    4,        // Number of attention heads
    DHidden:     256,      // Size of feed-forward hidden layer
    DropoutRate: 0.1,      // For regularization
}

// Create transformer layer
layer, err := transformer.NewTransformerLayer(config)
if err != nil {
    log.Fatal(err)
}

// Create input sequence [seq_len × d_model]
seqLen := 3
input := make(attention.Matrix, seqLen)
for i := range input {
    input[i] = make(attention.Vector, config.DModel)
    // Fill with your embedding data...
}

// Process through transformer
output, err := layer.Forward(input)
if err != nil {
    log.Fatal(err)
}
```

## Example Output

When running the examples, you'll see:

1. **Dot-Product Attention**:

   ```
   Query: [1 0 1 0]
   Attention Weights: [0.506 0.186 0.307]  // Shows focus on similar patterns
   Output: [2.601 3.601]                   // Weighted combination of values
   ```

2. **Multi-Head Attention**:

   ```
   Input dimensions: [2 batches × 3 tokens × 64 features]
   Output shape: [6×64]
   ```

3. **Transformer Layer**:
   ```
   Input shape: [3×64]
   Output shape: [3×64]
   ```

## Common Use Cases

1. **Text Processing**:

   - Sequence-to-sequence translation
   - Document summarization
   - Sentiment analysis

2. **Time Series**:

   - Financial forecasting
   - Sensor data analysis
   - Anomaly detection

3. **Structured Data**:
   - Graph node embedding
   - Feature interaction modeling
   - Recommendation systems

## Performance Considerations

- Matrix operations are optimized for CPU
- Memory allocations are minimized
- Batch processing for better throughput
- No external dependencies

For more detailed examples, see the `examples` directory in the repository.

## Why go-attention?

This module was created to provide a clean, efficient, and dependency-free implementation of attention mechanisms in Go. It's particularly useful for:

- **Edge Computing**: Zero external dependencies makes it perfect for edge devices where dependency management is crucial
- **Real-time Processing**: Pure Go implementation ensures predictable performance for real-time applications
- **Cloud-native Applications**: Efficient batched operations support high-throughput scaling in cloud environments
- **Embedded Systems**: Predictable resource usage and minimal memory allocations
- **Production Systems**: Comprehensive error handling and type safety for robust production deployments

## Features

- Efficient dot-product attention mechanism (upgraded with Scalable-Softmax (SSMax, s=1) for improved long-context performance)
- Multi-head attention support
- Full transformer layer implementation with:
  - Layer normalization
  - Position-wise feed-forward networks
  - Residual connections
- Batched operations for improved performance

## Roadmap

Future improvements may include:

- Positional encoding implementations
- Dropout support
- Additional transformer variants
- Pre-trained models
- Training utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

---

For research inquiries and press, please reach out to research@takara.ai

> 人類を変革する
