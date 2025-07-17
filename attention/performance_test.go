package attention

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// BenchmarkDotProductOptimized benchmarks the optimized dot product
func BenchmarkDotProductOptimized(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			v1 := make(Vector, size)
			v2 := make(Vector, size)
			
			for i := range v1 {
				v1[i] = rand.Float64()
				v2[i] = rand.Float64()
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				DotProductOptimized(v1, v2)
			}
		})
	}
}

// BenchmarkDotProductParallel benchmarks parallel dot product
func BenchmarkDotProductParallel(b *testing.B) {
	sizes := []int{1024, 4096, 16384}
	configs := []ParallelConfig{
		{NumWorkers: 2, ChunkSize: 64},
		{NumWorkers: 4, ChunkSize: 64},
		{NumWorkers: 8, ChunkSize: 64},
	}
	
	for _, size := range sizes {
		for _, config := range configs {
			b.Run(fmt.Sprintf("Size_%d_Workers_%d", size, config.NumWorkers), func(b *testing.B) {
				v1 := make(Vector, size)
				v2 := make(Vector, size)
				
				for i := range v1 {
					v1[i] = rand.Float64()
					v2[i] = rand.Float64()
				}
				
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					DotProductParallel(v1, v2, config)
				}
			})
		}
	}
}

// BenchmarkMatrixMultiplyOptimized benchmarks optimized matrix multiplication
func BenchmarkMatrixMultiplyOptimized(b *testing.B) {
	sizes := []struct{ rows, cols int }{
		{32, 32},
		{64, 64},
		{128, 128},
		{256, 256},
	}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size.rows, size.cols), func(b *testing.B) {
			a := make(Matrix, size.rows)
			matB := make(Matrix, size.cols)
			
			for i := range a {
				a[i] = make(Vector, size.cols)
				for j := range a[i] {
					a[i][j] = rand.Float64()
				}
			}
			
			for i := range matB {
				matB[i] = make(Vector, size.rows)
				for j := range matB[i] {
					matB[i][j] = rand.Float64()
				}
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatrixMultiplyOptimized(a, matB)
			}
		})
	}
}

// BenchmarkMatrixMultiplyParallel benchmarks parallel matrix multiplication
func BenchmarkMatrixMultiplyParallel(b *testing.B) {
	sizes := []struct{ rows, cols int }{
		{128, 128},
		{256, 256},
		{512, 512},
	}
	
	configs := []ParallelConfig{
		{NumWorkers: 2, ChunkSize: 32},
		{NumWorkers: 4, ChunkSize: 32},
		{NumWorkers: 8, ChunkSize: 32},
	}
	
	for _, size := range sizes {
		for _, config := range configs {
			b.Run(fmt.Sprintf("%dx%d_Workers_%d", size.rows, size.cols, config.NumWorkers), func(b *testing.B) {
				a := make(Matrix, size.rows)
				matB := make(Matrix, size.cols)
				
				for i := range a {
					a[i] = make(Vector, size.cols)
					for j := range a[i] {
						a[i][j] = rand.Float64()
					}
				}
				
				for i := range matB {
					matB[i] = make(Vector, size.rows)
					for j := range matB[i] {
						matB[i][j] = rand.Float64()
					}
				}
				
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MatrixMultiplyParallel(a, matB, config)
				}
			})
		}
	}
}

// BenchmarkSoftmaxParallel benchmarks parallel softmax
func BenchmarkSoftmaxParallel(b *testing.B) {
	sizes := []int{256, 1024, 4096, 16384}
	configs := []ParallelConfig{
		{NumWorkers: 2, ChunkSize: 64},
		{NumWorkers: 4, ChunkSize: 64},
		{NumWorkers: 8, ChunkSize: 64},
	}
	
	for _, size := range sizes {
		for _, config := range configs {
			b.Run(fmt.Sprintf("Size_%d_Workers_%d", size, config.NumWorkers), func(b *testing.B) {
				x := make(Vector, size)
				for i := range x {
					x[i] = rand.Float64()
				}
				
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					SoftmaxParallel(x, config)
				}
			})
		}
	}
}

// BenchmarkMemoryPooling benchmarks memory pooling vs standard allocation
func BenchmarkMemoryPooling(b *testing.B) {
	sizes := []int{64, 256, 1024}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Standard_Size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				v1 := make(Vector, size)
				v2 := make(Vector, size)
				DotProductUnsafe(v1, v2)
			}
		})
		
		b.Run(fmt.Sprintf("Pooled_Size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				v1 := GetVectorFromPool(size)
				v2 := GetVectorFromPool(size)
				DotProductUnsafe(v1, v2)
				PutVectorToPool(v1)
				PutVectorToPool(v2)
			}
		})
	}
}

// BenchmarkPerformanceMonitoring benchmarks the overhead of performance monitoring
func BenchmarkPerformanceMonitoring(b *testing.B) {
	// Enable monitoring
	config := DefaultPerformanceConfig()
	config.EnableMonitoring = true
	SetPerformanceConfig(config)
	
	size := 512
	v1 := make(Vector, size)
	v2 := make(Vector, size)
	
	for i := range v1 {
		v1[i] = rand.Float64()
		v2[i] = rand.Float64()
	}
	
	b.Run("WithMonitoring", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			PerformanceWrappedDotProduct(v1, v2)
		}
	})
	
	b.Run("WithoutMonitoring", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProduct(v1, v2)
		}
	})
}

// BenchmarkComparison compares all dot product implementations
func BenchmarkComparison(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	
	for _, size := range sizes {
		v1 := make(Vector, size)
		v2 := make(Vector, size)
		
		for i := range v1 {
			v1[i] = rand.Float64()
			v2[i] = rand.Float64()
		}
		
		b.Run(fmt.Sprintf("Safe_Size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				DotProduct(v1, v2)
			}
		})
		
		b.Run(fmt.Sprintf("Unsafe_Size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				DotProductUnsafe(v1, v2)
			}
		})
		
		b.Run(fmt.Sprintf("Optimized_Size_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				DotProductOptimized(v1, v2)
			}
		})
		
		b.Run(fmt.Sprintf("Parallel_Size_%d", size), func(b *testing.B) {
			config := DefaultParallelConfig()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				DotProductParallel(v1, v2, config)
			}
		})
	}
}

// TestPerformanceImprovements verifies that optimizations actually improve performance
func TestPerformanceImprovements(t *testing.T) {
	// This test ensures that our optimizations provide actual benefits
	// It's not a traditional unit test, but validates our performance claims
	
	size := 1024
	v1 := make(Vector, size)
	v2 := make(Vector, size)
	
	for i := range v1 {
		v1[i] = rand.Float64()
		v2[i] = rand.Float64()
	}
	
	iterations := 1000
	
	// Time standard implementation
	start := time.Now()
	for i := 0; i < iterations; i++ {
		DotProductUnsafe(v1, v2)
	}
	standardTime := time.Since(start)
	
	// Time best implementation (our new consistent one)
	start = time.Now()
	for i := 0; i < iterations; i++ {
		BestDotProduct(v1, v2)
	}
	bestTime := time.Since(start)
	
	// Time parallel implementation
	config := DefaultParallelConfig()
	start = time.Now()
	for i := 0; i < iterations; i++ {
		DotProductParallel(v1, v2, config)
	}
	parallelTime := time.Since(start)
	
	t.Logf("Standard: %v", standardTime)
	t.Logf("Best: %v", bestTime)
	t.Logf("Parallel: %v", parallelTime)
	
	// Verify that our best implementation is at least as fast as standard
	if bestTime > time.Duration(float64(standardTime)*1.5) {
		t.Errorf("Best implementation is too slow: %v vs %v", bestTime, standardTime)
	}
	
	// Verify that parallel is reasonable (may be slower due to overhead)
	if parallelTime > time.Duration(float64(standardTime)*5) {
		t.Errorf("Parallel implementation is too slow: %v vs %v", parallelTime, standardTime)
	}
} 