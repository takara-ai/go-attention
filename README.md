# go-attention

A pure Go implementation of attention mechanisms and transformer layers, designed for high performance and ease of use. This package provides implementations of scaled dot-product attention, multi-head attention, and transformer layers without any external dependencies.

## Features

- Pure Go implementation with no external dependencies
- Efficient dot-product attention mechanism
- Multi-head attention support
- Full transformer layer implementation with:
  - Layer normalization
  - Position-wise feed-forward networks
  - Residual connections
- Batched operations for improved performance
- Comprehensive error handling
- Example code for quick start

## Installation

```bash
go get github.com/takara-ai/go-attention
```

## Usage

### Simple Dot-Product Attention

```go
import "github.com/takara-ai/go-attention/attention"

query := attention.Vector{1.0, 0.0, 1.0, 0.0}
keys := attention.Matrix{
    {1.0, 0.0, 1.0, 0.0}, // Similar to query
    {0.0, 1.0, 0.0, 1.0}, // Different from query
    {0.5, 0.5, 0.5, 0.5}, // Neutral
}
values := attention.Matrix{
    {1.0, 2.0, 3.0, 4.0},
    {5.0, 6.0, 7.0, 8.0},
    {9.0, 10.0, 11.0, 12.0},
}

output, weights, err := attention.DotProductAttention(query, keys, values)
if err != nil {
    log.Fatal(err)
}
```

### Multi-Head Attention

```go
import "github.com/takara-ai/go-attention/attention"

config := attention.MultiHeadConfig{
    NumHeads:    4,
    DModel:      64,
    DKey:        16,  // DModel / NumHeads
    DValue:      16,  // DModel / NumHeads
    DropoutRate: 0.1,
}

mha, err := attention.NewMultiHeadAttention(config)
if err != nil {
    log.Fatal(err)
}

output, err := mha.Forward(queries, keys, values)
if err != nil {
    log.Fatal(err)
}
```

### Full Transformer Layer

```go
import (
    "github.com/takara-ai/go-attention/transformer"
)

config := transformer.TransformerConfig{
    DModel:      64,   // Model dimension
    NumHeads:    4,    // Number of attention heads
    DHidden:     256,  // Hidden dimension in feed-forward network
    DropoutRate: 0.1,
}

layer, err := transformer.NewTransformerLayer(config)
if err != nil {
    log.Fatal(err)
}

output, err := layer.Forward(input)
if err != nil {
    log.Fatal(err)
}
```

## Performance Considerations

- The implementation uses efficient matrix operations
- Memory allocations are minimized where possible
- Batch processing is supported for improved throughput
- No external dependencies means predictable performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details
