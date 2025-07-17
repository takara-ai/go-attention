package attention

import (
	"sync"
)

// VectorPool provides a pool of reusable vectors to reduce allocations
type VectorPool struct {
	pool sync.Pool
	size int
}

// NewVectorPool creates a new vector pool for vectors of the specified size
func NewVectorPool(size int) *VectorPool {
	return &VectorPool{
		size: size,
		pool: sync.Pool{
			New: func() interface{} {
				return make(Vector, size)
			},
		},
	}
}

// Get returns a vector from the pool
func (vp *VectorPool) Get() Vector {
	return vp.pool.Get().(Vector)
}

// Put returns a vector to the pool
func (vp *VectorPool) Put(v Vector) {
	// Reset the vector to zero
	for i := range v {
		v[i] = 0
	}
	vp.pool.Put(v)
}

// MatrixPool provides a pool of reusable matrices
type MatrixPool struct {
	pool     sync.Pool
	rows     int
	cols     int
}

// NewMatrixPool creates a new matrix pool
func NewMatrixPool(rows, cols int) *MatrixPool {
	return &MatrixPool{
		rows: rows,
		cols: cols,
		pool: sync.Pool{
			New: func() interface{} {
				matrix := make(Matrix, rows)
				for i := range matrix {
					matrix[i] = make(Vector, cols)
				}
				return matrix
			},
		},
	}
}

// Get returns a matrix from the pool
func (mp *MatrixPool) Get() Matrix {
	return mp.pool.Get().(Matrix)
}

// Put returns a matrix to the pool
func (mp *MatrixPool) Put(m Matrix) {
	// Reset the matrix to zero
	for i := range m {
		for j := range m[i] {
			m[i][j] = 0
		}
	}
	mp.pool.Put(m)
}

// Global pools for common sizes (can be tuned based on usage patterns)
var (
	smallVectorPool  = NewVectorPool(64)   // Common embedding size
	mediumVectorPool = NewVectorPool(256)  // Medium attention size
	largeVectorPool  = NewVectorPool(1024) // Large attention size
	
	smallMatrixPool  = NewMatrixPool(32, 64)   // Small batch size
	mediumMatrixPool = NewMatrixPool(64, 256)  // Medium batch size
)

// GetVectorFromPool returns a vector from an appropriate global pool
func GetVectorFromPool(size int) Vector {
	switch {
	case size <= 64:
		return smallVectorPool.Get()
	case size <= 256:
		return mediumVectorPool.Get()
	case size <= 1024:
		return largeVectorPool.Get()
	default:
		return make(Vector, size)
	}
}

// PutVectorToPool returns a vector to the appropriate global pool
func PutVectorToPool(v Vector) {
	size := len(v)
	switch {
	case size <= 64:
		smallVectorPool.Put(v)
	case size <= 256:
		mediumVectorPool.Put(v)
	case size <= 1024:
		largeVectorPool.Put(v)
	// For larger vectors, let GC handle them
	}
} 