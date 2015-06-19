package golsh

import (
	"fmt"
	"math"
)

// Vector defines a mathematical vector
type Vector []float64

// cosine similarity between two vectors.
func cosine(v1, v2 Vector) (float64, error) {
	if len(v1) != len(v2) {
		return 0, fmt.Errorf("vectors do not have the same dimensions")
	}

	var dot, sum1, sum2 float64
	for i := range v1 {
		dot += v1[i] * v2[i]
		sum1 += v1[i] * v1[i]
		sum2 += v2[i] * v2[i]
	}

	similarity := dot / float64(math.Sqrt(float64(sum1*sum2)))
	if math.IsNaN(float64(similarity)) {
		return 0, fmt.Errorf("NaN %v and %v", v1, v2)
	}

	return similarity, nil
}

func dot(vecA Vector, vecB Vector) float64 {
	var dot float64
	for i := range vecA {
		dot += vecA[i] * vecB[i]
	}
	return dot
}
