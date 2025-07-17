package transformer

import (
	"math"
	"testing"

	"github.com/takara-ai/go-attention/attention"
)

func TestNewLayerNorm(t *testing.T) {
	tests := []struct {
		name    string
		dim     int
		eps     float64
		wantPanic bool
	}{
		{
			name:    "valid parameters",
			dim:     64,
			eps:     1e-5,
			wantPanic: false,
		},
		{
			name:    "zero dimension",
			dim:     0,
			eps:     1e-5,
			wantPanic: true,
		},
		{
			name:    "negative dimension",
			dim:     -1,
			eps:     1e-5,
			wantPanic: true,
		},
		{
			name:    "zero epsilon",
			dim:     64,
			eps:     0.0,
			wantPanic: true,
		},
		{
			name:    "negative epsilon",
			dim:     64,
			eps:     -1e-5,
			wantPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if (r != nil) != tt.wantPanic {
					t.Errorf("NewLayerNorm() panic = %v, wantPanic %v", r, tt.wantPanic)
				}
			}()

			ln := NewLayerNorm(tt.dim, tt.eps)
			if !tt.wantPanic && ln == nil {
				t.Error("NewLayerNorm() returned nil for valid parameters")
			}
		})
	}
}

func TestLayerNormForward(t *testing.T) {
	dim := 4
	eps := 1e-5
	ln := NewLayerNorm(dim, eps)

	input := attention.Matrix{
		{1.0, 2.0, 3.0, 4.0},
		{0.0, 0.0, 0.0, 0.0},
		{-1.0, -2.0, -3.0, -4.0},
	}

	output, err := ln.Forward(input)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	// Check output dimensions
	if len(output) != len(input) {
		t.Errorf("output length = %d, want %d", len(output), len(input))
	}

	for i, vec := range output {
		if len(vec) != dim {
			t.Errorf("output[%d] dimension = %d, want %d", i, len(vec), dim)
		}
	}

	// Check that normalized values have mean close to 0 and std close to 1
	for i, vec := range output {
		mean := 0.0
		for _, v := range vec {
			mean += v
		}
		mean /= float64(len(vec))

		variance := 0.0
		for _, v := range vec {
			diff := v - mean
			variance += diff * diff
		}
		variance /= float64(len(vec))
		std := math.Sqrt(variance)

		if math.Abs(mean) > 1e-10 {
			t.Errorf("output[%d] mean = %v, want close to 0", i, mean)
		}
		// Skip std check for zero vectors (they will have std = 0)
		if i != 1 && math.Abs(std-1.0) > 1e-10 {
			t.Errorf("output[%d] std = %v, want close to 1", i, std)
		}
	}
}

func TestLayerNormForwardErrors(t *testing.T) {
	dim := 4
	eps := 1e-5
	ln := NewLayerNorm(dim, eps)

	tests := []struct {
		name    string
		input   attention.Matrix
		wantErr bool
	}{
		{
			name:    "dimension mismatch",
			input:   attention.Matrix{{1.0, 2.0, 3.0}}, // Wrong dimension
			wantErr: true,
		},
		{
			name:    "empty input",
			input:   attention.Matrix{},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ln.Forward(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Forward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestLayerNormString(t *testing.T) {
	dim := 64
	eps := 1e-5
	ln := NewLayerNorm(dim, eps)

	str := ln.String()
	expected := "LayerNorm(dim=64, eps=0.000010)"
	if str != expected {
		t.Errorf("String() = %v, want %v", str, expected)
	}
}

func TestNewFeedForward(t *testing.T) {
	tests := []struct {
		name    string
		dModel  int
		dHidden int
		wantPanic bool
	}{
		{
			name:    "valid parameters",
			dModel:  64,
			dHidden: 256,
			wantPanic: false,
		},
		{
			name:    "zero d_model",
			dModel:  0,
			dHidden: 256,
			wantPanic: true,
		},
		{
			name:    "negative d_model",
			dModel:  -1,
			dHidden: 256,
			wantPanic: true,
		},
		{
			name:    "zero d_hidden",
			dModel:  64,
			dHidden: 0,
			wantPanic: true,
		},
		{
			name:    "negative d_hidden",
			dModel:  64,
			dHidden: -1,
			wantPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if (r != nil) != tt.wantPanic {
					t.Errorf("NewFeedForward() panic = %v, wantPanic %v", r, tt.wantPanic)
				}
			}()

			ff := NewFeedForward(tt.dModel, tt.dHidden)
			if !tt.wantPanic && ff == nil {
				t.Error("NewFeedForward() returned nil for valid parameters")
			}
		})
	}
}

func TestFeedForwardForward(t *testing.T) {
	dModel := 4
	dHidden := 8
	ff := NewFeedForward(dModel, dHidden)

	input := attention.Matrix{
		{1.0, 2.0, 3.0, 4.0},
		{0.0, 0.0, 0.0, 0.0},
		{-1.0, -2.0, -3.0, -4.0},
	}

	output, err := ff.Forward(input)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	// Check output dimensions
	if len(output) != len(input) {
		t.Errorf("output length = %d, want %d", len(output), len(input))
	}

	for i, vec := range output {
		if len(vec) != dModel {
			t.Errorf("output[%d] dimension = %d, want %d", i, len(vec), dModel)
		}
	}
}

func TestFeedForwardString(t *testing.T) {
	dModel := 64
	dHidden := 256
	ff := NewFeedForward(dModel, dHidden)

	str := ff.String()
	expected := "FeedForward(d_model=64, d_hidden=256)"
	if str != expected {
		t.Errorf("String() = %v, want %v", str, expected)
	}
}

func TestNewTransformerLayer(t *testing.T) {
	tests := []struct {
		name    string
		config  TransformerConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: TransformerConfig{
				DModel:      64,
				NumHeads:    4,
				DHidden:     256,
				DropoutRate: 0.1,
			},
			wantErr: false,
		},
		{
			name: "zero d_model",
			config: TransformerConfig{
				DModel:      0,
				NumHeads:    4,
				DHidden:     256,
				DropoutRate: 0.1,
			},
			wantErr: true,
		},
		{
			name: "zero num_heads",
			config: TransformerConfig{
				DModel:      64,
				NumHeads:    0,
				DHidden:     256,
				DropoutRate: 0.1,
			},
			wantErr: true,
		},
		{
			name: "zero d_hidden",
			config: TransformerConfig{
				DModel:      64,
				NumHeads:    4,
				DHidden:     0,
				DropoutRate: 0.1,
			},
			wantErr: true,
		},
		{
			name: "not divisible",
			config: TransformerConfig{
				DModel:      64,
				NumHeads:    3, // 64 not divisible by 3
				DHidden:     256,
				DropoutRate: 0.1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := NewTransformerLayer(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewTransformerLayer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && layer == nil {
				t.Error("NewTransformerLayer() returned nil for valid config")
			}
		})
	}
}

func TestTransformerLayerForward(t *testing.T) {
	config := TransformerConfig{
		DModel:      8,
		NumHeads:    2,
		DHidden:     16,
		DropoutRate: 0.1,
	}

	layer, err := NewTransformerLayer(config)
	if err != nil {
		t.Fatalf("Failed to create TransformerLayer: %v", err)
	}

	// Create test input
	seqLen := 3
	input := make(attention.Matrix, seqLen)
	for i := range input {
		input[i] = make(attention.Vector, config.DModel)
		for j := range input[i] {
			input[i][j] = float64(i + j)
		}
	}

	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	// Check output dimensions
	if len(output) != seqLen {
		t.Errorf("output length = %d, want %d", len(output), seqLen)
	}

	for i, vec := range output {
		if len(vec) != config.DModel {
			t.Errorf("output[%d] dimension = %d, want %d", i, len(vec), config.DModel)
		}
	}
}

func TestTransformerLayerString(t *testing.T) {
	config := TransformerConfig{
		DModel:      64,
		NumHeads:    4,
		DHidden:     256,
		DropoutRate: 0.1,
	}

	layer, err := NewTransformerLayer(config)
	if err != nil {
		t.Fatalf("Failed to create TransformerLayer: %v", err)
	}

	str := layer.String()
	expected := "TransformerLayer(d_model=64, heads=4, d_hidden=256)"
	if str != expected {
		t.Errorf("String() = %v, want %v", str, expected)
	}
}

func BenchmarkLayerNormForward(b *testing.B) {
	dim := 64
	eps := 1e-5
	ln := NewLayerNorm(dim, eps)

	input := make(attention.Matrix, 10)
	for i := range input {
		input[i] = make(attention.Vector, dim)
		for j := range input[i] {
			input[i][j] = float64(i + j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ln.Forward(input)
	}
}

func BenchmarkFeedForwardForward(b *testing.B) {
	dModel := 64
	dHidden := 256
	ff := NewFeedForward(dModel, dHidden)

	input := make(attention.Matrix, 10)
	for i := range input {
		input[i] = make(attention.Vector, dModel)
		for j := range input[i] {
			input[i][j] = float64(i + j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ff.Forward(input)
	}
}

func BenchmarkTransformerLayerForward(b *testing.B) {
	config := TransformerConfig{
		DModel:      64,
		NumHeads:    4,
		DHidden:     256,
		DropoutRate: 0.1,
	}

	layer, err := NewTransformerLayer(config)
	if err != nil {
		b.Fatalf("Failed to create TransformerLayer: %v", err)
	}

	seqLen := 10
	input := make(attention.Matrix, seqLen)
	for i := range input {
		input[i] = make(attention.Vector, config.DModel)
		for j := range input[i] {
			input[i][j] = float64(i + j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
} 