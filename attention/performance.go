package attention

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// PerformanceStats tracks performance metrics
type PerformanceStats struct {
	OperationCount    int64         `json:"operation_count"`
	TotalTime         time.Duration `json:"total_time"`
	AverageTime       time.Duration `json:"average_time"`
	MinTime           time.Duration `json:"min_time"`
	MaxTime           time.Duration `json:"max_time"`
	LastOperationTime time.Time     `json:"last_operation_time"`
	MemoryAllocs      int64         `json:"memory_allocs"`
	MemoryBytes       int64         `json:"memory_bytes"`
}

// PerformanceMonitor tracks performance across operations
type PerformanceMonitor struct {
	stats map[string]*PerformanceStats
	mu    sync.RWMutex
}

// NewPerformanceMonitor creates a new performance monitor
func NewPerformanceMonitor() *PerformanceMonitor {
	return &PerformanceMonitor{
		stats: make(map[string]*PerformanceStats),
	}
}

// StartOperation begins timing an operation
func (pm *PerformanceMonitor) StartOperation(operation string) *OperationTimer {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return &OperationTimer{
		monitor:    pm,
		operation:  operation,
		startTime:  time.Now(),
		startAllocs: m.Mallocs,
		startBytes:  m.TotalAlloc,
	}
}

// OperationTimer tracks a single operation
type OperationTimer struct {
	monitor     *PerformanceMonitor
	operation   string
	startTime   time.Time
	startAllocs uint64
	startBytes  uint64
}

// End finishes timing the operation and records stats
func (ot *OperationTimer) End() {
	duration := time.Since(ot.startTime)
	
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	allocs := int64(m.Mallocs - ot.startAllocs)
	bytes := int64(m.TotalAlloc - ot.startBytes)
	
	ot.monitor.recordStats(ot.operation, duration, allocs, bytes)
}

// recordStats updates performance statistics
func (pm *PerformanceMonitor) recordStats(operation string, duration time.Duration, allocs, bytes int64) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	stats, exists := pm.stats[operation]
	if !exists {
		stats = &PerformanceStats{}
		pm.stats[operation] = stats
	}
	
	count := atomic.AddInt64(&stats.OperationCount, 1)
	atomic.AddInt64(&stats.MemoryAllocs, allocs)
	atomic.AddInt64(&stats.MemoryBytes, bytes)
	
	// Update timing stats
	stats.TotalTime += duration
	stats.AverageTime = stats.TotalTime / time.Duration(count)
	stats.LastOperationTime = time.Now()
	
	if duration < stats.MinTime || stats.MinTime == 0 {
		stats.MinTime = duration
	}
	if duration > stats.MaxTime {
		stats.MaxTime = duration
	}
}

// GetStats returns performance statistics for an operation
func (pm *PerformanceMonitor) GetStats(operation string) (*PerformanceStats, bool) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	
	stats, exists := pm.stats[operation]
	return stats, exists
}

// GetAllStats returns all performance statistics
func (pm *PerformanceMonitor) GetAllStats() map[string]*PerformanceStats {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	
	result := make(map[string]*PerformanceStats)
	for op, stats := range pm.stats {
		result[op] = stats
	}
	return result
}

// Reset clears all performance statistics
func (pm *PerformanceMonitor) Reset() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	pm.stats = make(map[string]*PerformanceStats)
}

// PerformanceConfig holds auto-tuning configuration
type PerformanceConfig struct {
	EnableMonitoring bool
	EnableAutoTuning bool
	MinVectorSize    int // Minimum size for parallel operations
	MinMatrixSize    int // Minimum size for parallel matrix operations
	MaxWorkers       int // Maximum number of workers
}

// DefaultPerformanceConfig returns reasonable defaults
func DefaultPerformanceConfig() PerformanceConfig {
	return PerformanceConfig{
		EnableMonitoring: true,
		EnableAutoTuning: true,
		MinVectorSize:    1024, // Only parallelize vectors >= 1024 elements
		MinMatrixSize:    64,   // Only parallelize matrices >= 64x64
		MaxWorkers:       runtime.NumCPU(),
	}
}

// Global performance monitor instance
var globalMonitor = NewPerformanceMonitor()
var globalConfig = DefaultPerformanceConfig()

// SetPerformanceConfig updates the global performance configuration
func SetPerformanceConfig(config PerformanceConfig) {
	globalConfig = config
}

// GetPerformanceStats returns performance statistics
func GetPerformanceStats(operation string) (*PerformanceStats, bool) {
	return globalMonitor.GetStats(operation)
}

// GetAllPerformanceStats returns all performance statistics
func GetAllPerformanceStats() map[string]*PerformanceStats {
	return globalMonitor.GetAllStats()
}

// ResetPerformanceStats clears all performance statistics
func ResetPerformanceStats() {
	globalMonitor.Reset()
}

// AutoTuneConfig automatically tunes performance parameters based on observed performance
func AutoTuneConfig() {
	if !globalConfig.EnableAutoTuning {
		return
	}
	
	// Analyze performance data and adjust parameters
	stats := globalMonitor.GetAllStats()
	
	// Example auto-tuning logic:
	// - If parallel operations are slower than serial, reduce parallelization
	// - If memory allocations are high, increase pool sizes
	// - If operations are fast, increase thresholds for parallelization
	
	for operation, stat := range stats {
		if stat.OperationCount < 10 {
			continue // Need more data
		}
		
		// Example: If average time is very low, reduce parallelization overhead
		if stat.AverageTime < 100*time.Microsecond {
			// Increase minimum sizes for parallel operations
			if globalConfig.MinVectorSize > 256 {
				globalConfig.MinVectorSize /= 2
			}
			if globalConfig.MinMatrixSize > 32 {
				globalConfig.MinMatrixSize /= 2
			}
		}
		
		// Example: If memory allocations are high, consider using pools more aggressively
		if stat.MemoryAllocs > stat.OperationCount*10 {
			// Could increase pool sizes or use pools more aggressively
		}
		
		fmt.Printf("Auto-tuning %s: avg=%v, allocs=%d, bytes=%d\n", 
			operation, stat.AverageTime, stat.MemoryAllocs, stat.MemoryBytes)
	}
}

// PerformanceWrappedDotProduct is a performance-monitored version of DotProduct
func PerformanceWrappedDotProduct(v1, v2 Vector) (float64, error) {
	if !globalConfig.EnableMonitoring {
		return DotProduct(v1, v2)
	}
	
	timer := globalMonitor.StartOperation("DotProduct")
	defer timer.End()
	
	return DotProduct(v1, v2)
}

// PerformanceWrappedDotProductParallel is a performance-monitored version with auto-tuning
func PerformanceWrappedDotProductParallel(v1, v2 Vector) (float64, error) {
	if !globalConfig.EnableMonitoring {
		// Auto-tune based on vector size
		if len(v1) >= globalConfig.MinVectorSize {
			config := DefaultParallelConfig()
			config.NumWorkers = globalConfig.MaxWorkers
			return DotProductParallel(v1, v2, config)
		}
		return DotProduct(v1, v2)
	}
	
	timer := globalMonitor.StartOperation("DotProductParallel")
	defer timer.End()
	
	// Auto-tune based on vector size
	if len(v1) >= globalConfig.MinVectorSize {
		config := DefaultParallelConfig()
		config.NumWorkers = globalConfig.MaxWorkers
		return DotProductParallel(v1, v2, config)
	}
	return DotProduct(v1, v2)
} 