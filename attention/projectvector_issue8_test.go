package attention

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// projectVectorColumnMajor matches the current implementation in multihead.go (issue #8).
func projectVectorColumnMajor(input Vector, weights Matrix) Vector {
	output := make(Vector, len(weights[0]))
	for i := range weights[0] {
		for j, w := range weights {
			output[i] += input[j] * w[i]
		}
	}
	return output
}

func projectVectorFixed(input Vector, weights Matrix) (Vector, error) {
	return projectVector(input, weights)
}

func makeProjectionFixture(dModel, dK int) (Vector, Matrix) {
	input := make(Vector, dModel)
	weights := make(Matrix, dModel)
	for i := range input {
		input[i] = rand.Float64()
		weights[i] = make(Vector, dK)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()
		}
	}
	return input, weights
}

func TestIssue8_ProjectVectorEquivalence(t *testing.T) {
	sizes := []struct{ dModel, dK int }{
		{64, 64},
		{512, 64},
		{768, 96},
	}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size.dModel, size.dK), func(t *testing.T) {
			input, weights := makeProjectionFixture(size.dModel, size.dK)
			col := projectVectorColumnMajor(input, weights)
			row, err := projectVectorFixed(input, weights)
			if err != nil {
				t.Fatal(err)
			}

			for i := range col {
				if math.Abs(col[i]-row[i]) > 1e-9 {
					t.Fatalf("mismatch at %d: column=%v row=%v", i, col[i], row[i])
				}
			}
		})
	}
}

func BenchmarkIssue8_ProjectVectorColumnMajor(b *testing.B) {
	benchmarkProjectVector(b, projectVectorColumnMajor, "column-major")
}

func BenchmarkIssue8_ProjectVectorRowMajor(b *testing.B) {
	benchmarkProjectVector(b, func(input Vector, weights Matrix) Vector {
		out, err := projectVectorFixed(input, weights)
		if err != nil {
			b.Fatal(err)
		}
		return out
	}, "row-major")
}

func benchmarkProjectVector(b *testing.B, fn func(Vector, Matrix) Vector, _ string) {
	sizes := []struct{ dModel, dK int }{
		{512, 64},
		{768, 96},
		{1024, 128},
		{2048, 256},
	}

	for _, size := range sizes {
		input, weights := makeProjectionFixture(size.dModel, size.dK)
		b.Run(fmt.Sprintf("%dx%d", size.dModel, size.dK), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fn(input, weights)
			}
		})
	}
}

func TestIssue8_ReproducePerformanceGap(t *testing.T) {
	// Typical transformer projection: d_model=768, d_k=96 (12 heads).
	const dModel, dK = 768, 96
	const iterations = 500

	input, weights := makeProjectionFixture(dModel, dK)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		projectVectorColumnMajor(input, weights)
	}
	columnTime := time.Since(start)

	start = time.Now()
	for i := 0; i < iterations; i++ {
		if _, err := projectVectorFixed(input, weights); err != nil {
			t.Fatal(err)
		}
	}
	rowTime := time.Since(start)

	speedup := float64(columnTime) / float64(rowTime)
	t.Logf("d_model=%d d_k=%d iterations=%d", dModel, dK, iterations)
	t.Logf("column-major (current): %v", columnTime)
	t.Logf("row-major (proposed):   %v", rowTime)
	t.Logf("speedup: %.2fx", speedup)

	if speedup < 1.5 {
		t.Fatalf("expected column-major to be noticeably slower; got only %.2fx speedup", speedup)
	}
}
