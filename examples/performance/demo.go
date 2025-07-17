package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"

	"github.com/takara-ai/go-attention/attention"
)

func main() {
	fmt.Println("=== Go-Attention Performance Demo ===\n")

	// 1. Demonstrate SIMD-optimized operations
	demoSIMDOptimizations()

	// 2. Demonstrate memory pooling
	demoMemoryPooling()

	// 3. Demonstrate parallel operations
	demoParallelOperations()

	// 4. Demonstrate performance monitoring
	demoPerformanceMonitoring()

	// 5. Demonstrate auto-tuning
	demoAutoTuning()

	fmt.Println("\n=== Performance Demo Complete ===")
}

func demoSIMDOptimizations() {
	fmt.Println("1. SIMD Optimizations")
	fmt.Println("=====================")

	// Create large vectors for testing
	size := 1024
	v1 := make(attention.Vector, size)
	v2 := make(attention.Vector, size)
	
	for i := range v1 {
		v1[i] = rand.Float64()
		v2[i] = rand.Float64()
	}

	// Test standard vs optimized dot product
	iterations := 10000
	
	// Standard version
	start := time.Now()
	for i := 0; i < iterations; i++ {
		attention.DotProductUnsafe(v1, v2)
	}
	standardDuration := time.Since(start)

	// Optimized version (with SIMD hints)
	start = time.Now()
	for i := 0; i < iterations; i++ {
		attention.DotProductOptimized(v1, v2)
	}
	optimizedDuration := time.Since(start)

	fmt.Printf("Standard DotProduct: %v for %d iterations\n", standardDuration, iterations)
	fmt.Printf("Optimized DotProduct: %v for %d iterations\n", optimizedDuration, iterations)
	fmt.Printf("Speedup: %.2fx\n", float64(standardDuration)/float64(optimizedDuration))
	fmt.Printf("Note: Assembly functions are stubs - real SIMD would be faster\n\n")
}

func demoMemoryPooling() {
	fmt.Println("2. Memory Pooling")
	fmt.Println("=================")

	// Test memory allocation patterns
	size := 256
	iterations := 10000

	// Standard allocation
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	startAllocs := m.Mallocs
	startBytes := m.TotalAlloc

	start := time.Now()
	for i := 0; i < iterations; i++ {
		v1 := make(attention.Vector, size)
		v2 := make(attention.Vector, size)
		attention.DotProductUnsafe(v1, v2)
	}
	standardDuration := time.Since(start)

	runtime.ReadMemStats(&m)
	standardAllocs := m.Mallocs - startAllocs
	standardBytes := m.TotalAlloc - startBytes

	// Pooled allocation
	runtime.ReadMemStats(&m)
	startAllocs = m.Mallocs
	startBytes = m.TotalAlloc

	start = time.Now()
	for i := 0; i < iterations; i++ {
		v1 := attention.GetVectorFromPool(size)
		v2 := attention.GetVectorFromPool(size)
		attention.DotProductUnsafe(v1, v2)
		attention.PutVectorToPool(v1)
		attention.PutVectorToPool(v2)
	}
	pooledDuration := time.Since(start)

	runtime.ReadMemStats(&m)
	pooledAllocs := m.Mallocs - startAllocs
	pooledBytes := m.TotalAlloc - startBytes

	fmt.Printf("Standard: %v, %d allocs, %d bytes\n", 
		standardDuration, standardAllocs, standardBytes)
	fmt.Printf("Pooled:   %v, %d allocs, %d bytes\n", 
		pooledDuration, pooledAllocs, pooledBytes)
	fmt.Printf("Alloc reduction: %.1fx\n", float64(standardAllocs)/float64(pooledAllocs))
	fmt.Printf("Time improvement: %.2fx\n", float64(standardDuration)/float64(pooledDuration))
	fmt.Println()
}

func demoParallelOperations() {
	fmt.Println("3. Parallel Operations")
	fmt.Println("=====================")

	// Test parallel vs serial operations
	size := 4096
	v1 := make(attention.Vector, size)
	v2 := make(attention.Vector, size)
	
	for i := range v1 {
		v1[i] = rand.Float64()
		v2[i] = rand.Float64()
	}

	iterations := 1000

	// Serial version
	start := time.Now()
	for i := 0; i < iterations; i++ {
		attention.DotProductUnsafe(v1, v2)
	}
	serialDuration := time.Since(start)

	// Parallel version
	config := attention.DefaultParallelConfig()
	config.NumWorkers = runtime.NumCPU()
	
	start = time.Now()
	for i := 0; i < iterations; i++ {
		attention.DotProductParallel(v1, v2, config)
	}
	parallelDuration := time.Since(start)

	fmt.Printf("CPU cores: %d\n", runtime.NumCPU())
	fmt.Printf("Serial:    %v for %d iterations\n", serialDuration, iterations)
	fmt.Printf("Parallel:  %v for %d iterations\n", parallelDuration, iterations)
	fmt.Printf("Speedup:   %.2fx\n", float64(serialDuration)/float64(parallelDuration))
	fmt.Printf("Efficiency: %.1f%%\n", 
		float64(serialDuration)/float64(parallelDuration)/float64(runtime.NumCPU())*100)
	fmt.Println()
}

func demoPerformanceMonitoring() {
	fmt.Println("4. Performance Monitoring")
	fmt.Println("=========================")

	// Enable performance monitoring
	config := attention.DefaultPerformanceConfig()
	config.EnableMonitoring = true
	attention.SetPerformanceConfig(config)

	// Run some operations
	size := 512
	v1 := make(attention.Vector, size)
	v2 := make(attention.Vector, size)
	
	for i := range v1 {
		v1[i] = rand.Float64()
		v2[i] = rand.Float64()
	}

	iterations := 1000
	for i := 0; i < iterations; i++ {
		attention.PerformanceWrappedDotProduct(v1, v2)
		attention.PerformanceWrappedDotProductParallel(v1, v2)
	}

	// Get performance statistics
	stats := attention.GetAllPerformanceStats()
	
	fmt.Println("Performance Statistics:")
	for operation, stat := range stats {
		fmt.Printf("  %s:\n", operation)
		fmt.Printf("    Count: %d\n", stat.OperationCount)
		fmt.Printf("    Avg:   %v\n", stat.AverageTime)
		fmt.Printf("    Min:   %v\n", stat.MinTime)
		fmt.Printf("    Max:   %v\n", stat.MaxTime)
		fmt.Printf("    Allocs: %d\n", stat.MemoryAllocs)
		fmt.Printf("    Bytes:  %d\n", stat.MemoryBytes)
		fmt.Println()
	}
}

func demoAutoTuning() {
	fmt.Println("5. Auto-Tuning")
	fmt.Println("==============")

	// Enable auto-tuning
	config := attention.DefaultPerformanceConfig()
	config.EnableAutoTuning = true
	attention.SetPerformanceConfig(config)

	// Run operations to gather data
	sizes := []int{64, 128, 256, 512, 1024, 2048}
	
	for _, size := range sizes {
		v1 := make(attention.Vector, size)
		v2 := make(attention.Vector, size)
		
		for i := range v1 {
			v1[i] = rand.Float64()
			v2[i] = rand.Float64()
		}

		iterations := 100
		for i := 0; i < iterations; i++ {
			attention.PerformanceWrappedDotProduct(v1, v2)
			attention.PerformanceWrappedDotProductParallel(v1, v2)
		}
	}

	// Run auto-tuning
	fmt.Println("Running auto-tuning analysis...")
	attention.AutoTuneConfig()

	// Show final configuration
	fmt.Println("Auto-tuned configuration:")
	fmt.Printf("  MinVectorSize: %d\n", config.MinVectorSize)
	fmt.Printf("  MinMatrixSize: %d\n", config.MinMatrixSize)
	fmt.Printf("  MaxWorkers:    %d\n", config.MaxWorkers)
	fmt.Println()
}

// Helper function to create random matrix
func randomMatrix(rows, cols int) attention.Matrix {
	matrix := make(attention.Matrix, rows)
	for i := range matrix {
		matrix[i] = make(attention.Vector, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64()
		}
	}
	return matrix
} 