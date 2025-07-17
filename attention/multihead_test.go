package attention

import (
	"math"
	"testing"
)

func TestNewMultiHeadAttention(t *testing.T) {
	tests := []struct {
		name    string
		config  MultiHeadConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: MultiHeadConfig{
				NumHeads: 4,
				DModel:   64,
				DKey:     16,
				DValue:   16,
			},
			wantErr: false,
		},
		{
			name: "zero heads",
			config: MultiHeadConfig{
				NumHeads: 0,
				DModel:   64,
				DKey:     16,
				DValue:   16,
			},
			wantErr: true,
		},
		{
			name: "negative heads",
			config: MultiHeadConfig{
				NumHeads: -1,
				DModel:   64,
				DKey:     16,
				DValue:   16,
			},
			wantErr: true,
		},
		{
			name: "zero model dimension",
			config: MultiHeadConfig{
				NumHeads: 4,
				DModel:   0,
				DKey:     16,
				DValue:   16,
			},
			wantErr: true,
		},
		{
			name: "not divisible",
			config: MultiHeadConfig{
				NumHeads: 3,
				DModel:   64,
				DKey:     16,
				DValue:   16,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mha, err := NewMultiHeadAttention(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewMultiHeadAttention() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && mha == nil {
				t.Error("NewMultiHeadAttention() returned nil for valid config")
			}
		})
	}
}

func TestMultiHeadAttentionForward(t *testing.T) {
	config := MultiHeadConfig{
		NumHeads: 2,
		DModel:   8,
		DKey:     4,
		DValue:   4,
	}

	mha, err := NewMultiHeadAttention(config)
	if err != nil {
		t.Fatalf("Failed to create MultiHeadAttention: %v", err)
	}

	// Create test data
	batchSize := 2
	query := make(Matrix, batchSize)
	key := make(Matrix, batchSize)
	value := make(Matrix, batchSize)

	for i := 0; i < batchSize; i++ {
		query[i] = make(Vector, config.DModel)
		key[i] = make(Vector, config.DModel)
		value[i] = make(Vector, config.DModel)
		for j := 0; j < config.DModel; j++ {
			query[i][j] = float64(i + j)
			key[i][j] = float64(i + j)
			value[i][j] = float64(i + j)
		}
	}

	output, err := mha.Forward(query, key, value)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	// Check output dimensions
	if len(output) != batchSize {
		t.Errorf("output batch size = %d, want %d", len(output), batchSize)
	}
	for i, vec := range output {
		if len(vec) != config.DModel {
			t.Errorf("output[%d] dimension = %d, want %d", i, len(vec), config.DModel)
		}
	}
}

func TestMultiHeadAttentionForwardErrors(t *testing.T) {
	config := MultiHeadConfig{
		NumHeads: 2,
		DModel:   8,
		DKey:     4,
		DValue:   4,
	}

	mha, err := NewMultiHeadAttention(config)
	if err != nil {
		t.Fatalf("Failed to create MultiHeadAttention: %v", err)
	}

	// Create valid test data
	batchSize := 2
	query := make(Matrix, batchSize)
	key := make(Matrix, batchSize)
	value := make(Matrix, batchSize)

	for i := 0; i < batchSize; i++ {
		query[i] = make(Vector, config.DModel)
		key[i] = make(Vector, config.DModel)
		value[i] = make(Vector, config.DModel)
		for j := 0; j < config.DModel; j++ {
			query[i][j] = float64(i + j)
			key[i][j] = float64(i + j)
			value[i][j] = float64(i + j)
		}
	}

	tests := []struct {
		name    string
		query   Matrix
		key     Matrix
		value   Matrix
		wantErr bool
	}{
		{
			name:    "batch size mismatch - query",
			query:   make(Matrix, 1),
			key:     key,
			value:   value,
			wantErr: true,
		},
		{
			name:    "batch size mismatch - key",
			query:   query,
			key:     make(Matrix, 1),
			value:   value,
			wantErr: true,
		},
		{
			name:    "batch size mismatch - value",
			query:   query,
			key:     key,
			value:   make(Matrix, 1),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := mha.Forward(tt.query, tt.key, tt.value)
			if (err != nil) != tt.wantErr {
				t.Errorf("Forward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestMultiHeadAttentionString(t *testing.T) {
	config := MultiHeadConfig{
		NumHeads: 4,
		DModel:   64,
		DKey:     16,
		DValue:   16,
	}

	mha, err := NewMultiHeadAttention(config)
	if err != nil {
		t.Fatalf("Failed to create MultiHeadAttention: %v", err)
	}

	str := mha.String()
	expected := "MultiHeadAttention(heads=4, model_dim=64, key_dim=16, value_dim=16)"
	if str != expected {
		t.Errorf("String() = %v, want %v", str, expected)
	}
}

func TestRandomMatrix(t *testing.T) {
	rows, cols := 10, 5
	mat := randomMatrix(rows, cols)

	if len(mat) != rows {
		t.Errorf("randomMatrix() rows = %d, want %d", len(mat), rows)
	}

	for i, row := range mat {
		if len(row) != cols {
			t.Errorf("randomMatrix()[%d] cols = %d, want %d", i, len(row), cols)
		}
	}

	// Check that values are reasonable (not all zero, not all same)
	hasNonZero := false
	hasDifferent := false
	firstVal := mat[0][0]

	for _, row := range mat {
		for _, val := range row {
			if val != 0 {
				hasNonZero = true
			}
			if val != firstVal {
				hasDifferent = true
			}
		}
	}

	if !hasNonZero {
		t.Error("randomMatrix() returned all zeros")
	}
	if !hasDifferent {
		t.Error("randomMatrix() returned all same values")
	}
}

func TestProjectVector(t *testing.T) {
	// Test valid projection
	input := Vector{1.0, 2.0, 3.0}
	weights := Matrix{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}

	output, err := projectVector(input, weights)
	if err != nil {
		t.Fatalf("projectVector() error = %v", err)
	}

	expected := Vector{1.0, 2.0, 3.0}
	if len(output) != len(expected) {
		t.Errorf("projectVector() output length = %d, want %d", len(output), len(expected))
	}

	for i, val := range output {
		if math.Abs(val-expected[i]) > 1e-10 {
			t.Errorf("projectVector() output[%d] = %v, want %v", i, val, expected[i])
		}
	}

	// Test error cases
	tests := []struct {
		name    string
		input   Vector
		weights Matrix
		wantErr bool
	}{
		{
			name:    "empty weights",
			input:   Vector{1.0, 2.0},
			weights: Matrix{},
			wantErr: true,
		},
		{
			name:    "dimension mismatch",
			input:   Vector{1.0, 2.0},
			weights: Matrix{{1.0, 2.0, 3.0}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := projectVector(tt.input, tt.weights)
			if (err != nil) != tt.wantErr {
				t.Errorf("projectVector() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func BenchmarkMultiHeadAttentionForward(b *testing.B) {
	config := MultiHeadConfig{
		NumHeads: 4,
		DModel:   64,
		DKey:     16,
		DValue:   16,
	}

	mha, err := NewMultiHeadAttention(config)
	if err != nil {
		b.Fatalf("Failed to create MultiHeadAttention: %v", err)
	}

	batchSize := 2
	query := make(Matrix, batchSize)
	key := make(Matrix, batchSize)
	value := make(Matrix, batchSize)

	for i := 0; i < batchSize; i++ {
		query[i] = make(Vector, config.DModel)
		key[i] = make(Vector, config.DModel)
		value[i] = make(Vector, config.DModel)
		for j := 0; j < config.DModel; j++ {
			query[i][j] = float64(i + j)
			key[i][j] = float64(i + j)
			value[i][j] = float64(i + j)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mha.Forward(query, key, value)
	}
} 