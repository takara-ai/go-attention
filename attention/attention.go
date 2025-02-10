package attention

import (
	"fmt"
	"math"
)

// Vector represents a slice of float64 values
type Vector []float64

// Matrix represents a 2D slice of float64 values
type Matrix []Vector

// DotProduct computes the dot product of two vectors
func DotProduct(v1, v2 Vector) (float64, error) {
	if len(v1) != len(v2) {
		return 0, fmt.Errorf("vector dimensions mismatch: %d != %d", len(v1), len(v2))
	}
	
	sum := 0.0
	for i := range v1 {
		sum += v1[i] * v2[i]
	}
	return sum, nil
}

// Softmax applies the softmax function to a vector
func Softmax(x Vector) Vector {
	if len(x) == 0 {
		return Vector{}
	}

	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	exps := make(Vector, len(x))
	sumExp := 0.0
	for i, v := range x {
		exps[i] = math.Exp(v - maxVal)
		sumExp += exps[i]
	}

	// Normalize
	invSum := 1.0 / sumExp
	for i := range exps {
		exps[i] *= invSum
	}
	
	return exps
}

// ScaleVector multiplies a vector by a scalar
func ScaleVector(v Vector, scale float64) Vector {
	result := make(Vector, len(v))
	for i, val := range v {
		result[i] = val * scale
	}
	return result
}

// AddVectors adds two vectors element-wise
func AddVectors(v1, v2 Vector) (Vector, error) {
	if len(v1) != len(v2) {
		return nil, fmt.Errorf("vector dimensions mismatch: %d != %d", len(v1), len(v2))
	}

	result := make(Vector, len(v1))
	for i := range v1 {
		result[i] = v1[i] + v2[i]
	}
	return result, nil
}

// DotProductAttention computes scaled dot-product attention
// query: [d_k], keys: [n, d_k], values: [n, d_v]
// Returns: attended vector [d_v] and attention weights [n]
func DotProductAttention(query Vector, keys, values Matrix) (Vector, Vector, error) {
	if len(keys) != len(values) {
		return nil, nil, fmt.Errorf("number of keys (%d) must match number of values (%d)", len(keys), len(values))
	}
	if len(keys) == 0 {
		return nil, nil, fmt.Errorf("empty keys and values")
	}

	// Compute attention scores
	scores := make(Vector, len(keys))
	for i, key := range keys {
		score, err := DotProduct(query, key)
		if err != nil {
			return nil, nil, fmt.Errorf("computing attention score: %w", err)
		}
		// Scale by sqrt(d_k) for better gradient flow
		scores[i] = score / math.Sqrt(float64(len(key)))
	}

	// Apply softmax to get attention weights
	weights := Softmax(scores)

	// Compute weighted sum of values
	dim := len(values[0])
	attended := make(Vector, dim)
	for i, weight := range weights {
		for j := 0; j < dim; j++ {
			attended[j] += weight * values[i][j]
		}
	}

	return attended, weights, nil
} 