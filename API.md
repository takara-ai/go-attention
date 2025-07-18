# Go-Attention API Documentation

This document provides a complete reference for all exported APIs in the go-attention library.

## Table of Contents

- [Core Types](#core-types)
- [Attention Package](#attention-package)
- [Transformer Package](#transformer-package)
- [Performance Features](#performance-features)
- [Memory Management](#memory-management)
- [Parallel Operations](#parallel-operations)
- [Deprecated Functions](#deprecated-functions)

## Core Types

### Vector
```go
type Vector []float64
```
Represents a 1D vector of float64 values. Used throughout the library for representing embeddings, attention weights, and other numerical data.

### Matrix
```go
type Matrix []Vector
```
Represents a 2D matrix of float64 values. Used for batched operations, key-value pairs, and multi-dimensional data.

### AttentionWeights
```go
type AttentionWeights = Vector
```
Type alias for attention weights, providing semantic clarity in function signatures.

## Attention Package

### Basic Operations

#### DotProduct
```go
func DotProduct(v1, v2 Vector) (float64, error)
```
Computes the dot product of two vectors. This is the canonical, optimized implementation that automatically selects the best algorithm based on input size and hardware.

**Parameters:**
- `v1, v2 Vector`: Input vectors of equal length

**Returns:**
- `float64`: Dot product result
- `error`: Error if vectors have different lengths

**Performance:** O(d) where d = len(v1)

#### BestDotProduct
```go
func BestDotProduct(v1, v2 Vector) (float64, error)
```
Alias for `DotProduct` for API clarity. Provides the same functionality.

#### Softmax
```go
func Softmax(x Vector) Vector
```
Applies the softmax function to a vector for attention weight normalization.

**Parameters:**
- `x Vector`: Input vector

**Returns:**
- `Vector`: Softmax probabilities (sum to 1.0)

**Performance:** O(n) where n = len(x)

#### ScaleVector
```go
func ScaleVector(v Vector, scale float64) Vector
```
Multiplies a vector by a scalar value.

**Parameters:**
- `v Vector`: Input vector
- `scale float64`: Scaling factor

**Returns:**
- `Vector`: Scaled vector

#### AddVectors
```go
func AddVectors(v1, v2 Vector) (Vector, error)
```
Adds two vectors element-wise.

**Parameters:**
- `v1, v2 Vector`: Input vectors of equal length

**Returns:**
- `Vector`: Element-wise sum
- `error`: Error if vectors have different lengths

### Attention Mechanisms

#### DotProductAttention
```go
func DotProductAttention(query Vector, keys, values Matrix) (Vector, AttentionWeights, error)
```
Computes scaled dot-product attention. This is the canonical, optimized implementation.

**Parameters:**
- `query Vector`: Query vector [d_k]
- `keys Matrix`: Key matrix [n, d_k]
- `values Matrix`: Value matrix [n, d_v]

**Returns:**
- `Vector`: Attended output [d_v]
- `AttentionWeights`: Attention weights [n]
- `error`: Error for dimension mismatches

**Performance:** O(n*d_k + n*d_v) where n = len(keys)

#### BestDotProductAttention
```go
func BestDotProductAttention(query Vector, keys, values Matrix) (Vector, AttentionWeights, error)
```
Alias for `DotProductAttention` for API clarity.

### Multi-Head Attention

#### MultiHeadConfig
```go
type MultiHeadConfig struct {
    NumHeads    int     // Number of parallel attention heads
    DModel      int     // Size of input/output embeddings
    DKey        int     // Size per head (DModel/NumHeads)
    DValue      int     // Size per head (DModel/NumHeads)
    DropoutRate float64 // For regularization
}
```

#### NewMultiHeadAttention
```go
func NewMultiHeadAttention(config MultiHeadConfig) (*MultiHeadAttention, error)
```
Creates a new multi-head attention module.

**Parameters:**
- `config MultiHeadConfig`: Configuration parameters

**Returns:**
- `*MultiHeadAttention`: Configured attention module
- `error`: Error for invalid configuration

#### MultiHeadAttention.Forward
```go
func (mha *MultiHeadAttention) Forward(query, key, value Matrix) (Matrix, error)
```
Processes input through multi-head attention.

**Parameters:**
- `query, key, value Matrix`: Input matrices [batch_size*seq_len, d_model]

**Returns:**
- `Matrix`: Output matrix [batch_size*seq_len, d_model]
- `error`: Error for dimension mismatches

#### MultiHeadAttention.String
```go
func (mha *MultiHeadAttention) String() string
```
Returns a string representation of the multi-head attention configuration.

### Utility Functions

#### validateMatrixDimensions
```go
func validateMatrixDimensions(matrices ...Matrix) error
```
Validates that all matrices have matching dimensions.

**Parameters:**
- `matrices ...Matrix`: Variable number of matrices to validate

**Returns:**
- `error`: Error if dimensions don't match

## Transformer Package

### Layer Normalization

#### LayerNorm
```go
type LayerNorm struct {
    Dim   int           // Dimension size
    Eps   float64       // Epsilon for numerical stability
    Gamma attention.Vector // Scale parameter
    Beta  attention.Vector // Shift parameter
}
```

#### NewLayerNorm
```go
func NewLayerNorm(dim int, eps float64) *LayerNorm
```
Creates a new layer normalization module.

**Parameters:**
- `dim int`: Dimension size
- `eps float64`: Epsilon for numerical stability

**Returns:**
- `*LayerNorm`: Configured layer normalization module

#### LayerNorm.Forward
```go
func (ln *LayerNorm) Forward(input attention.Matrix) (attention.Matrix, error)
```
Applies layer normalization to input.

**Parameters:**
- `input attention.Matrix`: Input matrix [seq_len, dim]

**Returns:**
- `attention.Matrix`: Normalized output [seq_len, dim]
- `error`: Error for dimension mismatches

#### LayerNorm.String
```go
func (ln *LayerNorm) String() string
```
Returns a string representation of the layer normalization configuration.

### Feed-Forward Network

#### FeedForward
```go
type FeedForward struct {
    DModel  int                // Input/output dimension
    DHidden int                // Hidden layer dimension
    W1      attention.Matrix   // First weight matrix
    B1      attention.Vector   // First bias vector
    W2      attention.Matrix   // Second weight matrix
    B2      attention.Vector   // Second bias vector
}
```

#### NewFeedForward
```go
func NewFeedForward(dModel, dHidden int) *FeedForward
```
Creates a new feed-forward network.

**Parameters:**
- `dModel int`: Input/output dimension
- `dHidden int`: Hidden layer dimension

**Returns:**
- `*FeedForward`: Configured feed-forward network

#### FeedForward.Forward
```go
func (ff *FeedForward) Forward(input attention.Matrix) (attention.Matrix, error)
```
Processes input through the feed-forward network.

**Parameters:**
- `input attention.Matrix`: Input matrix [seq_len, d_model]

**Returns:**
- `attention.Matrix`: Output matrix [seq_len, d_model]
- `error`: Error for dimension mismatches

#### FeedForward.String
```go
func (ff *FeedForward) String() string
```
Returns a string representation of the feed-forward network configuration.

### Complete Transformer Layer

#### TransformerConfig
```go
type TransformerConfig struct {
    DModel      int     // Size of token embeddings
    NumHeads    int     // Number of attention heads
    DHidden     int     // Size of feed-forward hidden layer
    DropoutRate float64 // For regularization
}
```

#### NewTransformerLayer
```go
func NewTransformerLayer(config TransformerConfig) (*TransformerLayer, error)
```
Creates a complete transformer layer with self-attention and feed-forward network.

**Parameters:**
- `config TransformerConfig`: Configuration parameters

**Returns:**
- `*TransformerLayer`: Configured transformer layer
- `error`: Error for invalid configuration

#### TransformerLayer.Forward
```go
func (t *TransformerLayer) Forward(input attention.Matrix) (attention.Matrix, error)
```
Processes input through the complete transformer layer.

**Parameters:**
- `input attention.Matrix`: Input matrix [seq_len, d_model]

**Returns:**
- `attention.Matrix`: Output matrix [seq_len, d_model]
- `error`: Error for dimension mismatches

#### TransformerLayer.String
```go
func (t *TransformerLayer) String() string
```
Returns a string representation of the transformer layer configuration.

## Performance Features

### Performance Monitoring

#### PerformanceStats
```go
type PerformanceStats struct {
    OperationCount    int64         `json:"operation_count"`
    TotalTime         time.Duration `json:"total_time"`
    AverageTime       time.Duration `json:"average_time"`
    MinTime           time.Duration `json:"min_time"`
    MaxTime           time.Duration `json:"max_time"`
    LastOperationTime time.Time     `json:"last_operation_time"`
    MemoryAllocs      int64         `json:"memory_allocs"`
    MemoryBytes       int64         `json:"memory_bytes"`
}
```

#### PerformanceConfig
```go
type PerformanceConfig struct {
    EnableMonitoring bool
    EnableAutoTuning bool
    MinVectorSize    int // Minimum size for parallel operations
    MinMatrixSize    int // Minimum size for parallel matrix operations
    MaxWorkers       int // Maximum number of workers
}
```

#### DefaultPerformanceConfig
```go
func DefaultPerformanceConfig() PerformanceConfig
```
Returns reasonable default performance configuration.

#### SetPerformanceConfig
```go
func SetPerformanceConfig(config PerformanceConfig)
```
Updates the global performance configuration.

#### GetPerformanceStats
```go
func GetPerformanceStats(operation string) (*PerformanceStats, bool)
```
Returns performance statistics for a specific operation.

#### GetAllPerformanceStats
```go
func GetAllPerformanceStats() map[string]*PerformanceStats
```
Returns all performance statistics.

#### ResetPerformanceStats
```go
func ResetPerformanceStats()
```
Clears all performance statistics.

### Performance Wrappers

#### PerformanceWrappedDotProduct
```go
func PerformanceWrappedDotProduct(v1, v2 Vector) (float64, error)
```
Dot product with performance monitoring enabled.

#### PerformanceWrappedDotProductParallel
```go
func PerformanceWrappedDotProductParallel(v1, v2 Vector) (float64, error)
```
Parallel dot product with performance monitoring enabled.

## Memory Management

### Vector Pools

#### VectorPool
```go
type VectorPool struct {
    pool sync.Pool
    size int
}
```

#### NewVectorPool
```go
func NewVectorPool(size int) *VectorPool
```
Creates a new vector pool for vectors of the specified size.

#### VectorPool.Get
```go
func (vp *VectorPool) Get() Vector
```
Returns a vector from the pool.

#### VectorPool.Put
```go
func (vp *VectorPool) Put(v Vector)
```
Returns a vector to the pool.

### Matrix Pools

#### MatrixPool
```go
type MatrixPool struct {
    pool sync.Pool
    rows int
    cols int
}
```

#### NewMatrixPool
```go
func NewMatrixPool(rows, cols int) *MatrixPool
```
Creates a new matrix pool.

#### MatrixPool.Get
```go
func (mp *MatrixPool) Get() Matrix
```
Returns a matrix from the pool.

#### MatrixPool.Put
```go
func (mp *MatrixPool) Put(m Matrix)
```
Returns a matrix to the pool.

### Global Pool Access

#### GetVectorFromPool
```go
func GetVectorFromPool(size int) Vector
```
Returns a vector from an appropriate global pool based on size.

#### PutVectorToPool
```go
func PutVectorToPool(v Vector)
```
Returns a vector to the appropriate global pool.

## Parallel Operations

### Parallel Configuration

#### ParallelConfig
```go
type ParallelConfig struct {
    NumWorkers int // Number of worker goroutines (0 = auto-detect)
    ChunkSize  int // Size of work chunks
}
```

#### DefaultParallelConfig
```go
func DefaultParallelConfig() ParallelConfig
```
Returns a reasonable default parallel configuration.

### Parallel Matrix Operations

#### MatrixMultiplyParallel
```go
func MatrixMultiplyParallel(a, b Matrix, config ParallelConfig) (Matrix, error)
```
Performs parallel matrix multiplication.

**Parameters:**
- `a, b Matrix`: Input matrices
- `config ParallelConfig`: Parallel configuration

**Returns:**
- `Matrix`: Result matrix
- `error`: Error for dimension mismatches

#### SoftmaxParallel
```go
func SoftmaxParallel(x Vector, config ParallelConfig) Vector
```
Computes softmax in parallel.

**Parameters:**
- `x Vector`: Input vector
- `config ParallelConfig`: Parallel configuration

**Returns:**
- `Vector`: Softmax result

## Assembly Optimizations

### Optimized Operations

#### dotProductAVX2
```go
func dotProductAVX2(a, b unsafe.Pointer, n int) float64
```
AVX2-optimized dot product (placeholder implementation).

#### dotProductNEON
```go
func dotProductNEON(a, b unsafe.Pointer, n int) float64
```
NEON-optimized dot product (placeholder implementation).

#### MatrixMultiplyOptimized
```go
func MatrixMultiplyOptimized(a, b Matrix) (Matrix, error)
```
Optimized matrix multiplication with blocking and cache-friendly access patterns.

## Deprecated Functions

The following functions are deprecated and should not be used in new code. They are maintained for backward compatibility but will be removed in future versions.

### DotProductUnsafe
```go
func DotProductUnsafe(v1, v2 Vector) float64
```
**Deprecated:** Use `DotProduct` instead. This function assumes equal vector lengths without validation.

### DotProductPooled
```go
func DotProductPooled(v1, v2 Vector) (float64, error)
```
**Deprecated:** Use `DotProduct` instead. This function is now an alias for the canonical implementation.

### DotProductParallel
```go
func DotProductParallel(v1, v2 Vector, config interface{}) (float64, error)
```
**Deprecated:** Use `DotProduct` instead. This function is now an alias for the canonical implementation.

### DotProductOptimized
```go
func DotProductOptimized(v1, v2 Vector) (float64, error)
```
**Deprecated:** Use `DotProduct` instead. This function is now an alias for the canonical implementation.

## Constants

### DefaultEpsilon
```go
const DefaultEpsilon = 1e-5
```
Default epsilon value for numerical stability in softmax and normalization operations.

## Error Handling

All functions that can fail return an `error` value. Common error types include:

- **Dimension mismatches:** When input vectors/matrices have incompatible dimensions
- **Invalid configuration:** When configuration parameters are invalid (negative dimensions, etc.)
- **Empty inputs:** When required inputs are empty or nil

## Performance Characteristics

### Time Complexity
- **DotProduct:** O(d) where d = vector dimension
- **Softmax:** O(n) where n = vector length
- **DotProductAttention:** O(n*d_k + n*d_v) where n = number of keys
- **MultiHeadAttention:** O(seq_len² * d_model)
- **LayerNorm:** O(seq_len * dim)
- **FeedForward:** O(seq_len * d_model * d_hidden)

### Memory Usage
- **DotProduct:** O(1) additional memory
- **Softmax:** O(n) for exponential values
- **Attention:** O(n) for attention weights
- **MultiHead:** O(seq_len * d_model) for intermediate results
- **Transformer:** O(seq_len * d_model) for residual connections

## Best Practices

1. **Use canonical functions:** Always use `DotProduct` and `DotProductAttention` instead of deprecated variants
2. **Check errors:** Always handle error returns from functions
3. **Reuse memory pools:** Use `GetVectorFromPool` and `PutVectorToPool` for high-frequency operations
4. **Monitor performance:** Enable performance monitoring in production to track optimization opportunities
5. **Validate dimensions:** Use `validateMatrixDimensions` when working with multiple matrices
6. **Configure appropriately:** Set reasonable values for `ParallelConfig` and `PerformanceConfig` based on your use case

## Examples

See the `examples/` directory for comprehensive usage examples:

- `examples/basic/`: Basic usage patterns
- `examples/advanced/`: Advanced features like sentiment analysis and serverless deployment
- `examples/performance/`: Performance optimization examples
- `cmd/demo/`: Complete API demonstration 