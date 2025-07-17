package attention

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// ParallelConfig holds configuration for parallel operations
type ParallelConfig struct {
	NumWorkers int // Number of worker goroutines (0 = auto-detect)
	ChunkSize  int // Size of work chunks
}

// DefaultParallelConfig returns a reasonable default configuration
func DefaultParallelConfig() ParallelConfig {
	return ParallelConfig{
		NumWorkers: runtime.NumCPU(),
		ChunkSize:  64, // Good balance for most operations
	}
}

// MatrixMultiplyParallel performs parallel matrix multiplication
func MatrixMultiplyParallel(a, b Matrix, config ParallelConfig) (Matrix, error) {
	if len(a) == 0 || len(b) == 0 {
		return nil, fmt.Errorf("empty matrix")
	}
	
	if len(a[0]) != len(b) {
		return nil, fmt.Errorf("matrix dimensions incompatible: %dx%d * %dx%d", 
			len(a), len(a[0]), len(b), len(b[0]))
	}
	
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	
	rows := len(a)
	cols := len(b[0])
	
	result := make(Matrix, rows)
	for i := range result {
		result[i] = make(Vector, cols)
	}
	
	// Parallelize by rows
	chunkSize := (rows + config.NumWorkers - 1) / config.NumWorkers
	
	var wg sync.WaitGroup
	for i := 0; i < config.NumWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * chunkSize
			end := start + chunkSize
			if end > rows {
				end = rows
			}
			if start >= rows {
				return
			}
			
			// Compute matrix multiplication for assigned rows
			for row := start; row < end; row++ {
				for col := 0; col < cols; col++ {
					sum := 0.0
					for k := 0; k < len(b); k++ {
						sum += a[row][k] * b[k][col]
					}
					result[row][col] = sum
				}
			}
		}(i)
	}
	
	wg.Wait()
	return result, nil
}

// SoftmaxParallel computes softmax in parallel
func SoftmaxParallel(x Vector, config ParallelConfig) Vector {
	if len(x) == 0 {
		return Vector{}
	}
	
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	
	// Find max value (parallel)
	maxVal := x[0]
	if len(x) > config.ChunkSize {
		maxChunks := make([]float64, config.NumWorkers)
		chunkSize := (len(x) + config.NumWorkers - 1) / config.NumWorkers
		
		var wg sync.WaitGroup
		for i := 0; i < config.NumWorkers; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				
				start := workerID * chunkSize
				end := start + chunkSize
				if end > len(x) {
					end = len(x)
				}
				if start >= len(x) {
					return
				}
				
				localMax := x[start]
				for j := start + 1; j < end; j++ {
					if x[j] > localMax {
						localMax = x[j]
					}
				}
				maxChunks[workerID] = localMax
			}(i)
		}
		wg.Wait()
		
		for _, chunkMax := range maxChunks {
			if chunkMax > maxVal {
				maxVal = chunkMax
			}
		}
	} else {
		for _, v := range x[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
	}
	
	// Compute exponentials and sum (parallel)
	exps := make(Vector, len(x))
	sumExp := 0.0
	
	if len(x) > config.ChunkSize {
		sumChunks := make([]float64, config.NumWorkers)
		chunkSize := (len(x) + config.NumWorkers - 1) / config.NumWorkers
		
		var wg sync.WaitGroup
		for i := 0; i < config.NumWorkers; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				
				start := workerID * chunkSize
				end := start + chunkSize
				if end > len(x) {
					end = len(x)
				}
				if start >= len(x) {
					return
				}
				
				localSum := 0.0
				for j := start; j < end; j++ {
					exps[j] = math.Exp(x[j] - maxVal)
					localSum += exps[j]
				}
				sumChunks[workerID] = localSum
			}(i)
		}
		wg.Wait()
		
		for _, chunkSum := range sumChunks {
			sumExp += chunkSum
		}
	} else {
		for i, v := range x {
			exps[i] = math.Exp(v - maxVal)
			sumExp += exps[i]
		}
	}
	
	// Normalize (parallel)
	if sumExp == 0 {
		return exps
	}
	
	if len(x) > config.ChunkSize {
		chunkSize := (len(x) + config.NumWorkers - 1) / config.NumWorkers
		var wg sync.WaitGroup
		for i := 0; i < config.NumWorkers; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				
				start := workerID * chunkSize
				end := start + chunkSize
				if end > len(x) {
					end = len(x)
				}
				if start >= len(x) {
					return
				}
				
				for j := start; j < end; j++ {
					exps[j] /= sumExp
				}
			}(i)
		}
		wg.Wait()
	} else {
		for i := range exps {
			exps[i] /= sumExp
		}
	}
	
	return exps
} 