package attention

import (
	"math"
	"testing"
)

func TestDotProduct(t *testing.T) {
	tests := []struct {
		name    string
		v1      Vector
		v2      Vector
		want    float64
		wantErr bool
	}{
		{
			name:    "basic dot product",
			v1:      Vector{1.0, 2.0, 3.0},
			v2:      Vector{4.0, 5.0, 6.0},
			want:    32.0, // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
			wantErr: false,
		},
		{
			name:    "zero vectors",
			v1:      Vector{0.0, 0.0, 0.0},
			v2:      Vector{1.0, 2.0, 3.0},
			want:    0.0,
			wantErr: false,
		},
		{
			name:    "dimension mismatch",
			v1:      Vector{1.0, 2.0},
			v2:      Vector{1.0, 2.0, 3.0},
			want:    0.0,
			wantErr: true,
		},
		{
			name:    "empty vectors",
			v1:      Vector{},
			v2:      Vector{},
			want:    0.0,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DotProduct(tt.v1, tt.v2)
			if (err != nil) != tt.wantErr {
				t.Errorf("DotProduct() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("DotProduct() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDotProductUnsafe(t *testing.T) {
	v1 := Vector{1.0, 2.0, 3.0}
	v2 := Vector{4.0, 5.0, 6.0}
	
	got := DotProductUnsafe(v1, v2)
	want := 32.0
	
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("DotProductUnsafe() = %v, want %v", got, want)
	}
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name string
		input Vector
		wantSum float64 // Should sum to 1.0
	}{
		{
			name: "basic softmax",
			input: Vector{1.0, 2.0, 3.0},
			wantSum: 1.0,
		},
		{
			name: "negative values",
			input: Vector{-1.0, 0.0, 1.0},
			wantSum: 1.0,
		},
		{
			name: "large values",
			input: Vector{100.0, 101.0, 102.0},
			wantSum: 1.0,
		},
		{
			name: "empty vector",
			input: Vector{},
			wantSum: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Softmax(tt.input)
			
			if len(tt.input) == 0 {
				if len(got) != 0 {
					t.Errorf("Softmax() returned non-empty result for empty input")
				}
				return
			}
			
			// Check that probabilities sum to 1.0
			sum := 0.0
			for _, v := range got {
				sum += v
			}
			
			if math.Abs(sum-tt.wantSum) > 1e-10 {
				t.Errorf("Softmax() probabilities sum to %v, want %v", sum, tt.wantSum)
			}
			
			// Check that all values are positive
			for i, v := range got {
				if v < 0 {
					t.Errorf("Softmax() result[%d] = %v, want positive", i, v)
				}
			}
		})
	}
}

func TestDotProductAttention(t *testing.T) {
	query := Vector{1.0, 0.0, 1.0, 0.0}
	keys := Matrix{
		{1.0, 0.0, 1.0, 0.0},  // Similar to query
		{0.0, 1.0, 0.0, 1.0},  // Different from query
		{0.5, 0.5, 0.5, 0.5},  // Neutral pattern
	}
	values := Matrix{
		{1.0, 2.0},  // Value for similar key
		{3.0, 4.0},  // Value for different key
		{5.0, 6.0},  // Value for neutral key
	}

	output, weights, err := DotProductAttention(query, keys, values)
	if err != nil {
		t.Fatalf("DotProductAttention() error = %v", err)
	}

	// Check output dimensions
	if len(output) != 2 {
		t.Errorf("output dimension = %d, want 2", len(output))
	}
	if len(weights) != 3 {
		t.Errorf("weights dimension = %d, want 3", len(weights))
	}

	// Check that weights sum to 1.0
	weightSum := 0.0
	for _, w := range weights {
		weightSum += w
	}
	if math.Abs(weightSum-1.0) > 1e-10 {
		t.Errorf("attention weights sum to %v, want 1.0", weightSum)
	}

	// Check that weights are positive
	for i, w := range weights {
		if w < 0 {
			t.Errorf("weight[%d] = %v, want positive", i, w)
		}
	}

	// The first key should have higher attention weight (more similar to query)
	if weights[0] <= weights[1] {
		t.Errorf("expected first key to have higher attention weight, got weights[0]=%v, weights[1]=%v", weights[0], weights[1])
	}
}

func TestDotProductAttentionErrors(t *testing.T) {
	tests := []struct {
		name    string
		query   Vector
		keys    Matrix
		values  Matrix
		wantErr bool
	}{
		{
			name:    "empty keys and values",
			query:   Vector{1.0, 2.0},
			keys:    Matrix{},
			values:  Matrix{},
			wantErr: true,
		},
		{
			name:    "dimension mismatch",
			query:   Vector{1.0, 2.0},
			keys:    Matrix{{1.0, 2.0}},
			values:  Matrix{{1.0}, {2.0}}, // Different number of values
			wantErr: true,
		},
		{
			name:    "key dimension mismatch",
			query:   Vector{1.0, 2.0},
			keys:    Matrix{{1.0, 2.0, 3.0}}, // Different dimension
			values:  Matrix{{1.0}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := DotProductAttention(tt.query, tt.keys, tt.values)
			if (err != nil) != tt.wantErr {
				t.Errorf("DotProductAttention() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateMatrixDimensions(t *testing.T) {
	tests := []struct {
		name    string
		matrices []Matrix
		wantErr bool
	}{
		{
			name:    "empty slice",
			matrices: []Matrix{},
			wantErr: false,
		},
		{
			name:    "single matrix",
			matrices: []Matrix{{{1.0, 2.0}, {3.0, 4.0}}},
			wantErr: false,
		},
		{
			name:    "matching dimensions",
			matrices: []Matrix{
				{{1.0, 2.0}, {3.0, 4.0}},
				{{5.0, 6.0}, {7.0, 8.0}},
			},
			wantErr: false,
		},
		{
			name:    "mismatched dimensions",
			matrices: []Matrix{
				{{1.0, 2.0}, {3.0, 4.0}},
				{{5.0, 6.0}}, // Different number of rows
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMatrixDimensions(tt.matrices...)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateMatrixDimensions() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func BenchmarkDotProduct(b *testing.B) {
	v1 := Vector{1.0, 2.0, 3.0, 4.0, 5.0}
	v2 := Vector{6.0, 7.0, 8.0, 9.0, 10.0}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProduct(v1, v2)
	}
}

func BenchmarkDotProductUnsafe(b *testing.B) {
	v1 := Vector{1.0, 2.0, 3.0, 4.0, 5.0}
	v2 := Vector{6.0, 7.0, 8.0, 9.0, 10.0}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProductUnsafe(v1, v2)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	input := Vector{1.0, 2.0, 3.0, 4.0, 5.0}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Softmax(input)
	}
}

func BenchmarkDotProductAttention(b *testing.B) {
	query := Vector{1.0, 0.0, 1.0, 0.0}
	keys := Matrix{
		{1.0, 0.0, 1.0, 0.0},
		{0.0, 1.0, 0.0, 1.0},
		{0.5, 0.5, 0.5, 0.5},
	}
	values := Matrix{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProductAttention(query, keys, values)
	}
} 