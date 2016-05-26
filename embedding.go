package golsh

import (
	"bytes"
	"fmt"
	"math/rand"
)

type random interface {
	draw() float32
}

type gauss struct{}

func (g *gauss) draw() float32 {
	return float32(rand.NormFloat64())
}

type embedding struct {
	normals [][]float32
}

func newEmbedding(d int, size int, r random) embedding {
	normals := make([][]float32, d, d)
	for i := 0; i < d; i++ {
		normals[i] = normal(size, r)
	}
	return embedding{normals}
}

func normal(size int, r random) []float32 {
	result := make([]float32, size, size)
	for i := 0; i < size; i++ {
		result[i] = r.draw()
	}
	return result
}

// returns an embedding of size d
func (e *embedding) embed(id int, vector []float32) string {
	result := make([]bool, len(e.normals), len(e.normals))
	for i, normal := range e.normals {
		result[i] = dimension(vector, normal)
	}
	return fmt.Sprintf("%d-%s", id, bitToString(result))
}

func dimension(vecA []float32, vecB []float32) bool {
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
