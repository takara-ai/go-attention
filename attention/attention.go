package attention

import (
	"fmt"
	"math"
)

// Vector represents a slice of float64 values
type Vector []float64

// Matrix represents a 2D slice of float64 values
type Matrix []Vector

// AttentionWeights represents attention weights for clarity
type AttentionWeights = Vector

// Constants for numerical stability
const (
	DefaultEpsilon = 1e-5 // For numerical stability in softmax
)

// i is the representation of the query
// j is the representation of the key
// k is the representation of the value

// DotProduct computes the dot product of two vectors
// Performance: O(d) where d=len(v1)
func DotProduct(v1, v2 Vector) (float64, error) {
	return bestDotProduct(v1, v2)
}

// BestDotProduct is an alias for DotProduct for API clarity
func BestDotProduct(v1, v2 Vector) (float64, error) {
	return DotProduct(v1, v2)
}

// bestDotProduct is the single, canonical implementation
func bestDotProduct(v1, v2 Vector) (float64, error) {
	if len(v1) != len(v2) {
		return 0, fmt.Errorf("vector dimensions mismatch: %d != %d", len(v1), len(v2))
	}
	// Loop unrolling for performance
	sum := 0.0
	n := len(v1)
	for i := 0; i < n-7; i += 8 {
		sum += v1[i]*v2[i] + v1[i+1]*v2[i+1] + v1[i+2]*v2[i+2] + v1[i+3]*v2[i+3] +
			v1[i+4]*v2[i+4] + v1[i+5]*v2[i+5] + v1[i+6]*v2[i+6] + v1[i+7]*v2[i+7]
	}
	for i := (n/8)*8; i < n; i++ {
		sum += v1[i] * v2[i]
	}
	return sum, nil
}

// Deprecated: Use DotProduct instead.
func DotProductUnsafe(v1, v2 Vector) float64 {
	sum := 0.0
	for i := range v1 {
		sum += v1[i] * v2[i]
	}
	return sum
}

// Deprecated: Use DotProduct instead.
func DotProductPooled(v1, v2 Vector) (float64, error) {
	return DotProduct(v1, v2)
}

// Deprecated: Use DotProduct instead.
func DotProductParallel(v1, v2 Vector, config interface{}) (float64, error) {
	return DotProduct(v1, v2)
}

// Deprecated: Use DotProduct instead.
func DotProductOptimized(v1, v2 Vector) (float64, error) {
	return DotProduct(v1, v2)
}

// Softmax applies the softmax function to a vector
// Performance: O(n) where n=len(x)
func Softmax(x Vector) Vector {
	if len(x) == 0 {
		return Vector{}
	}

	maxVal := x[0]
	// Find max value for numerical stability
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	exps := make(Vector, len(x))
	sumExp := 0.0
	for i, v := range x {
		// Subtracting maxVal before exponentiation
		exps[i] = math.Exp(v - maxVal)
		sumExp += exps[i]
	}

	// Handle sumExp == 0 case to prevent NaN (division by zero)
	// This can happen if all inputs are extremely small negative numbers.
	if sumExp == 0 {
		// Return a zero vector. It's already initialized with zeros implicitly by make
		return exps 
	}

	// Normalize: Use direct division instead of multiplying by inverse
	for i := range exps {
		exps[i] /= sumExp
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
// Performance: O(n*d_k + n*d_v) where n=len(keys), d_k=len(query), d_v=len(values[0])
// Memory: O(n) for attention weights
// Consider using BatchDotProductAttention for multiple queries
func DotProductAttention(query Vector, keys, values Matrix) (Vector, AttentionWeights, error) {
	return bestDotProductAttention(query, keys, values)
}

// BestDotProductAttention is an alias for DotProductAttention
func BestDotProductAttention(query Vector, keys, values Matrix) (Vector, AttentionWeights, error) {
	return DotProductAttention(query, keys, values)
}

// bestDotProductAttention is the single, canonical implementation
func bestDotProductAttention(query Vector, keys, values Matrix) (Vector, AttentionWeights, error) {
	n := len(keys)
	if n == 0 {
        // If keys are empty, check if values exist to determine output dimension d_v
        if len(values) > 0 && len(values[0]) > 0 {
            d_v := len(values[0])
            // Return empty weights and a zero vector of the correct value dimension
            return make(Vector, d_v), Vector{}, nil 
        }
		// If both keys and values are empty (or values have zero dimension), return error or nil vectors
		return nil, nil, fmt.Errorf("empty keys and values provided")
	}

    // Basic dimension validation before proceeding
	if len(values) != n {
		return nil, nil, fmt.Errorf("dimension mismatch: %d keys vs %d values (expected equal)", n, len(values))
	}
    if len(values[0]) == 0 {
        return nil, nil, fmt.Errorf("value dimension (d_v) cannot be zero")
    }
	d_v := len(values[0]) // Dimension of value vectors

    // Determine key dimension (d_k) safely
    d_k := len(query)
    if n > 0 && len(keys[0]) != d_k {
         // Check the first key's dimension against the query dimension
         return nil, nil, fmt.Errorf("query dimension (%d) must match key dimension (%d)", d_k, len(keys[0]))
    }


	// Compute attention scores
	scores := make(Vector, n)
    
    // Pre-calculate scaling factor only if d_k > 0 to avoid division by zero or sqrt of zero
    scale := 1.0
    if d_k > 0 {
        scale = 1.0 / math.Sqrt(float64(d_k))
    } // If d_k is 0, scale remains 1.0, dot product will likely be 0 unless vectors are empty.

	for i, key := range keys {
        // Ensure consistent key dimensions within the loop
        if len(key) != d_k {
             return nil, nil, fmt.Errorf("key dimension mismatch at index %d: expected %d, got %d", i, d_k, len(key))
        }
		score, err := DotProduct(query, key) // DotProduct already checks len(query) == len(key)
		if err != nil {
			// This error should theoretically not happen if the outer checks pass, but handle defensively.
			return nil, nil, fmt.Errorf("error computing dot product for key %d: %w", i, err)
		}
		// Scale by sqrt(d_k) for better gradient flow
		scores[i] = score * scale // Use pre-calculated scale
	}

	// --- SSMax Modification (s=1) ---
	// Calculate log n (natural logarithm of sequence length)
	// If n=0 or n=1, log_n is handled to avoid issues. log(1)=0 is mathematically correct per formula.
	log_n := 0.0 
	if n > 1 {
		log_n = math.Log(float64(n))
	} else if n == 1 {
		// For n=1, log(1)=0. Softmax of a single element scaled by 0 is still undefined 
		// in the standard implementation, but mathematically should result in a weight of 1.
		// The existing Softmax handles single elements correctly, so scaling by 0 is fine.
		// Alternatively, we could set log_n = 1.0 to effectively bypass SSMax scaling for n=1.
		// Let's stick to the formula log_n = 0 for n=1.
		log_n = 0.0 
	}
	
	// Multiply scores by log n
	// Avoid modifying if log_n is effectively 1 (e.g., if n == math.E, though unlikely)
	// Or if log_n is 0 (when n=1), scaling is identity or zeroing.
	if n > 1 { // Only scale if n > 1, otherwise log_n is 0
		for i := range scores {
			scores[i] *= log_n
		}
	}
	// --- End SSMax Modification ---

	// Apply softmax to get attention weights (now using SSMax-modified scores)
	weights := Softmax(scores)

	// Compute weighted sum of values
	attended := make(Vector, d_v) // Use d_v determined earlier
	for i, weight := range weights {
        // Ensure consistent value dimensions within the loop
        if len(values[i]) != d_v {
             return nil, nil, fmt.Errorf("value dimension mismatch at index %d: expected %d, got %d", i, d_v, len(values[i]))
        }

        // Optimization: Skip summation if weight is zero
        if weight == 0 {
            continue
        }

		// Fuse the scaling by weight into the accumulation loop
		valueVec := values[i] // Local ref might help optimizer, maybe minor
		for j := 0; j < d_v; j++ { // Iterate up to d_v
			attended[j] += weight * valueVec[j]
		}
	}

	return attended, weights, nil
} 

// validateMatrixDimensions validates that all matrices have the same number of rows
func validateMatrixDimensions(matrices ...Matrix) error {
	if len(matrices) == 0 {
		return nil
	}
	
	firstDim := len(matrices[0])
	for i, m := range matrices {
		if len(m) != firstDim {
			return fmt.Errorf("matrix %d has %d rows, expected %d", i, len(m), firstDim)
		}
	}
	return nil
} 

// TODO: kv caching
// TODO: tokenization support
// TODO: rotary positional embedding support RoPE
// TODO: SwarmFormer Local-Global Hiearchical Attention Support