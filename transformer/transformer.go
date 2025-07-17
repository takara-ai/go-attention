package transformer

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/takara-ai/go-attention/attention"
)

// LayerNorm implements layer normalization
type LayerNorm struct {
	Dim    int
	Eps    float64
	Gamma  attention.Vector // Scale parameter
	Beta   attention.Vector // Shift parameter
}

// NewLayerNorm creates a new layer normalization module
func NewLayerNorm(dim int, eps float64) *LayerNorm {
	if dim <= 0 {
		panic(fmt.Sprintf("invalid dimension: %d (must be positive)", dim))
	}
	if eps <= 0 {
		panic(fmt.Sprintf("invalid epsilon: %f (must be positive)", eps))
	}
	
	gamma := make(attention.Vector, dim)
	beta := make(attention.Vector, dim)
	
	// Initialize parameters
	for i := range gamma {
		gamma[i] = 1.0 // Initialize scale to 1
		beta[i] = 0.0  // Initialize shift to 0
	}
	
	return &LayerNorm{
		Dim:    dim,
		Eps:    eps,
		Gamma:  gamma,
		Beta:   beta,
	}
}

// Forward applies layer normalization
// Performance: O(seq_len * dim) where seq_len=len(input), dim=ln.Dim
func (ln *LayerNorm) Forward(input attention.Matrix) (attention.Matrix, error) {
	output := make(attention.Matrix, len(input))
	
	for i, vec := range input {
		if len(vec) != ln.Dim {
			return nil, fmt.Errorf("input dimension mismatch: expected %d, got %d", ln.Dim, len(vec))
		}
		
		// Compute mean
		mean := 0.0
		for _, v := range vec {
			mean += v
		}
		mean /= float64(ln.Dim)
		
		// Compute variance
		variance := 0.0
		for _, v := range vec {
			diff := v - mean
			variance += diff * diff
		}
		variance /= float64(ln.Dim)
		
		// Normalize
		normalized := make(attention.Vector, ln.Dim)
		stdDev := math.Sqrt(variance + ln.Eps)
		for j, v := range vec {
			normalized[j] = ln.Gamma[j]*((v-mean)/stdDev) + ln.Beta[j]
		}
		
		output[i] = normalized
	}
	
	return output, nil
}

// String returns a string representation of the LayerNorm
func (ln *LayerNorm) String() string {
	return fmt.Sprintf("LayerNorm(dim=%d, eps=%f)", ln.Dim, ln.Eps)
}

// FeedForward implements a position-wise feed-forward network
type FeedForward struct {
	DModel     int
	DHidden    int
	W1, W2     attention.Matrix
	B1, B2     attention.Vector
}

// NewFeedForward creates a new feed-forward network
func NewFeedForward(dModel, dHidden int) *FeedForward {
	if dModel <= 0 {
		panic(fmt.Sprintf("invalid d_model: %d (must be positive)", dModel))
	}
	if dHidden <= 0 {
		panic(fmt.Sprintf("invalid d_hidden: %d (must be positive)", dHidden))
	}
	
	ff := &FeedForward{
		DModel:  dModel,
		DHidden: dHidden,
		W1:      make(attention.Matrix, dModel),
		W2:      make(attention.Matrix, dHidden),
		B1:      make(attention.Vector, dHidden),
		B2:      make(attention.Vector, dModel),
	}

	// Initialize weights with Xavier initialization
	scale1 := math.Sqrt(2.0 / float64(dModel+dHidden))
	scale2 := math.Sqrt(2.0 / float64(dHidden+dModel))
	
	for i := range ff.W1 {
		ff.W1[i] = make(attention.Vector, dHidden)
		for j := range ff.W1[i] {
			ff.W1[i][j] = (rand.Float64() - 0.5) * scale1
		}
	}
	
	for i := range ff.W2 {
		ff.W2[i] = make(attention.Vector, dModel)
		for j := range ff.W2[i] {
			ff.W2[i][j] = (rand.Float64() - 0.5) * scale2
		}
	}
	
	return ff
}

// Forward applies the feed-forward network
// Performance: O(seq_len * d_model * d_hidden) where seq_len=len(input)
func (ff *FeedForward) Forward(input attention.Matrix) (attention.Matrix, error) {
	// First layer
	hidden := make(attention.Matrix, len(input))
	for i, vec := range input {
		projected, err := projectVector(vec, ff.W1)
		if err != nil {
			return nil, fmt.Errorf("projecting first layer: %w", err)
		}
		
		// Add bias and apply ReLU
		hidden[i] = make(attention.Vector, ff.DHidden)
		for j := range projected {
			val := projected[j] + ff.B1[j]
			if val > 0 {
				hidden[i][j] = val
			}
		}
	}
	
	// Second layer
	output := make(attention.Matrix, len(input))
	for i, vec := range hidden {
		projected, err := projectVector(vec, ff.W2)
		if err != nil {
			return nil, fmt.Errorf("projecting second layer: %w", err)
		}
		
		// Add bias
		output[i] = make(attention.Vector, ff.DModel)
		for j := range projected {
			output[i][j] = projected[j] + ff.B2[j]
		}
	}
	
	return output, nil
}

// String returns a string representation of the FeedForward
func (ff *FeedForward) String() string {
	return fmt.Sprintf("FeedForward(d_model=%d, d_hidden=%d)", ff.DModel, ff.DHidden)
}

// Helper function to project a vector through a weight matrix
func projectVector(input attention.Vector, weights attention.Matrix) (attention.Vector, error) {
	if len(weights) == 0 || len(weights[0]) == 0 {
		return nil, fmt.Errorf("empty weight matrix")
	}
	if len(input) != len(weights) {
		return nil, fmt.Errorf("input dimension (%d) does not match weights (%d)", len(input), len(weights))
	}

	output := make(attention.Vector, len(weights[0]))
	for i := range weights[0] {
		for j, w := range weights {
			output[i] += input[j] * w[i]
		}
	}
	return output, nil
}

// TransformerConfig holds configuration for a transformer layer
type TransformerConfig struct {
	DModel      int     // Model dimension
	NumHeads    int     // Number of attention heads
	DHidden     int     // Hidden dimension in feed-forward network
	DropoutRate float64 // Dropout rate
}

// TransformerLayer implements a single transformer layer
type TransformerLayer struct {
	Config      TransformerConfig
	SelfAttn    *attention.MultiHeadAttention
	FeedForward *FeedForward
	Norm1       *LayerNorm
	Norm2       *LayerNorm
}

// NewTransformerLayer creates a new transformer layer
func NewTransformerLayer(config TransformerConfig) (*TransformerLayer, error) {
	if config.DModel <= 0 {
		return nil, fmt.Errorf("invalid d_model: %d (must be positive)", config.DModel)
	}
	if config.NumHeads <= 0 {
		return nil, fmt.Errorf("invalid num_heads: %d (must be positive)", config.NumHeads)
	}
	if config.DHidden <= 0 {
		return nil, fmt.Errorf("invalid d_hidden: %d (must be positive)", config.DHidden)
	}
	if config.DModel%config.NumHeads != 0 {
		return nil, fmt.Errorf("d_model (%d) must be divisible by num_heads (%d)", config.DModel, config.NumHeads)
	}
	
	// Create multi-head attention
	attnConfig := attention.MultiHeadConfig{
		NumHeads:    config.NumHeads,
		DModel:      config.DModel,
		DKey:        config.DModel / config.NumHeads,
		DValue:      config.DModel / config.NumHeads,
		DropoutRate: config.DropoutRate,
	}
	
	selfAttn, err := attention.NewMultiHeadAttention(attnConfig)
	if err != nil {
		return nil, fmt.Errorf("creating self-attention: %w", err)
	}
	
	return &TransformerLayer{
		Config:      config,
		SelfAttn:    selfAttn,
		FeedForward: NewFeedForward(config.DModel, config.DHidden),
		Norm1:       NewLayerNorm(config.DModel, 1e-5),
		Norm2:       NewLayerNorm(config.DModel, 1e-5),
	}, nil
}

// Forward applies the transformer layer
// Performance: O(seq_len * d_model * d_hidden + seq_len^2 * d_model)
func (t *TransformerLayer) Forward(input attention.Matrix) (attention.Matrix, error) {
	// Self-attention sub-layer
	normalized1, err := t.Norm1.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("normalizing input: %w", err)
	}
	
	attended, err := t.SelfAttn.Forward(normalized1, normalized1, normalized1)
	if err != nil {
		return nil, fmt.Errorf("computing self-attention: %w", err)
	}
	
	// Add & Norm
	residual1 := make(attention.Matrix, len(input))
	for i := range input {
		residual1[i], err = attention.AddVectors(input[i], attended[i])
		if err != nil {
			return nil, fmt.Errorf("adding residual connection: %w", err)
		}
	}
	
	// Feed-forward sub-layer
	normalized2, err := t.Norm2.Forward(residual1)
	if err != nil {
		return nil, fmt.Errorf("normalizing first sub-layer output: %w", err)
	}
	
	ffOutput, err := t.FeedForward.Forward(normalized2)
	if err != nil {
		return nil, fmt.Errorf("computing feed-forward: %w", err)
	}
	
	// Add & Norm
	output := make(attention.Matrix, len(input))
	for i := range input {
		output[i], err = attention.AddVectors(residual1[i], ffOutput[i])
		if err != nil {
			return nil, fmt.Errorf("adding final residual connection: %w", err)
		}
	}
	
	return output, nil
}

// String returns a string representation of the TransformerLayer
func (t *TransformerLayer) String() string {
	return fmt.Sprintf("TransformerLayer(d_model=%d, heads=%d, d_hidden=%d)", 
		t.Config.DModel, t.Config.NumHeads, t.Config.DHidden)
} 