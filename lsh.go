package golsh

import ()

// Lsh creates a new Lsh object
type Lsh struct {
	vectors    map[int]vector
	embeddings []embedding
	hash       map[string][]int
}

// NewLsh created a new Lsh object
func NewLsh(vectors map[int]vector, numEmbeddings int, d int, size int) Lsh {

	// create global embeddings
	embeddings := make([]embedding, d, d)
	for i := 0; i < numEmbeddings; i++ {
		embeddings[i] = newEmbedding(d, size)
	}

	// embed input vectors
	hash := make(map[string][]int)
	for id, vector := range vectors {
		for _, embedding := range embeddings {
			h := embedding.embed(vector)
			hash[h] = append(hash[h], id)
		}
	}

	return Lsh{vectors, embeddings, hash}
}

func (l *Lsh) vector(id int) vector {
	return l.vectors[id]
}

func (l *Lsh) ann(vector vector, k int) []int {
	candidates := l.candidates(vector)
	return l.knn(vector, candidates, k)
}

func (l *Lsh) candidates(vec vector) []int {
	// embed input vector
	candidates := make([]int, 100)
	for _, embedding := range l.embeddings {
		h := embedding.embed(vec)
		candidates = append(candidates, l.hash[h]...)
	}

	return candidates
}

type hit struct {
	id     int
	vector vector
	score  float64
}

// hit scorer

func (l *Lsh) knn(vector vector, candidates []int, k int) []int {
	return candidates
}
