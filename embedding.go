package golsh

import "math/rand"

type random interface {
	draw() float64
}

type gauss struct {
}

func (g *gauss) draw() float64 {
	return rand.NormFloat64()
}

type embedding struct {
	normals [][]float64
}

func newEmbedding(d int, size int, r random) embedding {
	normals := make([][]float64, d)
	for i := 0; i < d; i++ {
		normals[i] = normal(size, r)
	}
	return embedding{normals}
}

func normal(size int, r random) []float64 {
	result := make([]float64, size)
	for i := 0; i < size; i++ {
		result[i] = r.draw()
	}
	return result
}

// returns an embedding of size d
func (e *embedding) embed(vector []float64) uint64 {
	var result uint64
	for i, normal := range e.normals {
		if dot(vector, normal) > 0 {
			result |= (1 << uint64(i))
		}
	}
	return result
}

func dot(x, y []float64) float64 {
	var sum float64
	for i, v := range x {
		sum += y[i] * v
	}
	return sum
}
