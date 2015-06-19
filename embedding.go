package golsh

import (
	"bytes"
	"math/rand"
)

type random interface {
	draw() feature
}

type gauss struct{}

func (g *gauss) draw() feature {
	return feature(rand.NormFloat64())
}

type embedding struct {
	normals []vector
}

func newEmbedding(d int, size int, r random) embedding {
	normals := make([]vector, d, d)
	for i := 0; i < d; i++ {
		normals[i] = normal(size, r)
	}
	return embedding{normals}
}

func normal(size int, r random) vector {
	result := make([]feature, size, size)
	for i := 0; i < size; i++ {
		result[i] = r.draw()
	}
	return result
}

// returns an embedding of size d
func (e *embedding) embed(vector vector) string {
	result := make([]bool, len(e.normals), len(e.normals))
	for i, normal := range e.normals {
		result[i] = dimension(vector, normal)
	}
	return bitToString(result)
}

func dimension(vecA vector, vecB vector) bool {
	dot := dot(vecA, vecB)
	if dot > 0 {
		return true
	}
	return false
}

func bitToString(bits []bool) string {
	var buffer bytes.Buffer

	for _, bit := range bits {
		if bit {
			buffer.WriteString("1")
		} else {
			buffer.WriteString("0")
		}
	}
	return buffer.String()
}
