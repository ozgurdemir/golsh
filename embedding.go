package golsh

import "math/rand"

type embedding struct {
	normals []vector
}

func newEmbedding(d int, size int) embedding {
	normals := make([]vector, d, d)
	for i := 0; i < d; i++ {
		normals[i] = normal(size)
	}
	return embedding{normals}
}

func normal(size int) vector {
	result := make([]feature, size, size)
	for i := 0; i < size; i++ {
		result[i] = gauss()
	}
	return result
}

func gauss() feature {
	return feature(rand.NormFloat64())
}

// returns an embedding of size d
func (e *embedding) embed(vector vector) string {
	result := make([]bool, len(e.normals), len(e.normals))
	for i, normal := range e.normals {
		result[i] = dimension(vector, normal)
	}
	return bitToString(result)
}

func bitToString(bit []bool) string {
	return "test"
}

func dimension(vecA vector, vecB vector) bool {
	dot := dot(vecA, vecB)
	if dot > 0 {
		return true
	}
	return false
}
