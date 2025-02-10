# go-attention

<img src="https://takara.ai/images/logo-24/TakaraAi.svg" width="200" alt="Takara.ai Logo" />

From the Frontier Research Team at takara.ai we present the first pure Go implementation of attention mechanisms and transformer layers, designed for high performance and ease of use.

---

## Quick Start

The fastest way to test the transformer layer is to run our example script:

```bash
# Get the module
go get github.com/takara-ai/go-attention

# Create example.go with the contents from our example file
# Then run:
go run example.go
```

This will demonstrate a transformer layer processing a sequence of random input tokens and show you the transformation results.

## Why go-attention?

This module was created to provide a clean, efficient, and dependency-free implementation of attention mechanisms in Go. It's particularly useful for:

- **Edge Computing**: Zero external dependencies makes it perfect for edge devices where dependency management is crucial
- **Real-time Processing**: Pure Go implementation ensures predictable performance for real-time applications
- **Cloud-native Applications**: Efficient batched operations support high-throughput scaling in cloud environments
- **Embedded Systems**: Predictable resource usage and minimal memory allocations
- **Production Systems**: Comprehensive error handling and type safety for robust production deployments

## Potential Applications

The module provides a foundation for building various attention-based systems:

- **Natural Language Processing**:

  - Text summarization
  - Machine translation
  - Question-answering systems
  - Sentiment analysis
  - Named entity recognition

- **Time Series Analysis**:

  - Financial market prediction
  - Weather forecasting
  - Anomaly detection
  - Traffic prediction
  - Energy consumption forecasting

- **Computer Vision** (with appropriate input encoding):

  - Image classification
  - Object detection
  - Image captioning
  - Visual attention models

- **Multimodal Systems**:
  - Text-to-image systems
  - Cross-modal retrieval
  - Multimodal fusion
  - Audio-visual processing

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
// This example demonstrates how to use the transformer layer from the go-attention module.
// It creates a simple transformer layer and processes a sequence of random input tokens.
package main

import (
    "fmt"
    "log"
    "math/rand"
    "github.com/takara-ai/go-attention/transformer"
    "github.com/takara-ai/go-attention/attention"
)

func main() {
    // Create transformer layer with configuration:
    // - DModel: 64 (size of each token's embedding vector)
    // - NumHeads: 4 (parallel attention heads for better feature capture)
    // - DHidden: 256 (size of feed-forward network's hidden layer)
    // - DropoutRate: 0.1 (regularization to prevent overfitting)
    layer, err := transformer.NewTransformerLayer(transformer.TransformerConfig{
        DModel: 64, NumHeads: 4, DHidden: 256, DropoutRate: 0.1,
    })
    if err != nil {
        log.Fatal(err)
    }

    // Create a sequence of 3 tokens, each represented by a 64-dimensional vector
    // This could represent word embeddings, image patches, or any other sequence data
    // Values are randomly initialized between -1 and 1 for this example
    input := make(attention.Matrix, 3)
    for i := range input {
        input[i] = make(attention.Vector, 64)
        for j := range input[i] {
            input[i][j] = rand.Float64()*2 - 1
        }
    }

    // Process the sequence through the transformer layer
    // The transformer will:
    // 1. Apply self-attention to capture relationships between tokens
    // 2. Process through a feed-forward network
    // 3. Apply layer normalization and residual connections
    if output, err := layer.Forward(input); err != nil {
        log.Fatal(err)
    } else {
        // Compare the first 4 dimensions of input and output
        // to see how the transformer has modified the sequence
        fmt.Printf("Input (first 4 dims): %v\nOutput (first 4 dims): %v\n",
            input[0][:4], output[0][:4])
    }
}
```

## Performance Considerations

- The implementation uses efficient matrix operations
- Memory allocations are minimized where possible
- Batch processing is supported for improved throughput
- No external dependencies means predictable performance

## Roadmap

Future improvements may include:

- Positional encoding implementations
- Dropout support
- CUDA acceleration support
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
