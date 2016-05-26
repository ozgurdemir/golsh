package golsh

import (
	"fmt"
	"sort"
)

// Lsh creates a new Lsh object
type Lsh struct {
	vectors    *map[int][]float32
	embeddings []embedding
	hash       map[string][]int
}

// NewLsh created a new Lsh object
func NewLsh(vectors *map[int][]float32, numEmbeddings int, d int) Lsh {
	return newLsh(vectors, numEmbeddings, d, &gauss{})
}

func newLsh(vectors *map[int][]float32, numEmbeddings int, d int, r random) Lsh {
	size := getSize(vectors)

	// create global embeddings
	embeddings := make([]embedding, numEmbeddings, numEmbeddings)
	for i := 0; i < numEmbeddings; i++ {
		embeddings[i] = newEmbedding(d, size, r)
	}

	// embed input vectors
	hash := make(map[string][]int)
	for id, vector := range *vectors {
		for embedID, embedding := range embeddings {
			h := embedding.embed(embedID, vector)
			hash[h] = append(hash[h], id)
		}
	}

	return Lsh{vectors, embeddings, hash}
}

func getSize(vectors *map[int][]float32) int {
	for _, vector := range *vectors {
		return len(vector)
	}
	return 0
}

// Vector fetches the vector for a given id
func (l *Lsh) Vector(id int) ([]float32, bool) {
	vector, ok := (*l.vectors)[id]
	return vector, ok
}

// Ann finds approximate nearest neughbour using LSH cosine
func (l *Lsh) Ann(vector []float32, k int, threshold float32) ([]Hit, int, error) {
	candidates := l.candidates(vector)
	hits, err := l.knn(vector, deduplicate(candidates), k)
	hits = minCosine(hits, threshold)
	return hits, len(candidates), err
}

func (l *Lsh) candidates(vec []float32) []int {
	candidates := make([]int, 0, 100)
	for embedID, embedding := range l.embeddings {
		h := embedding.embed(embedID, vec)
		candidates = append(candidates, l.hash[h]...)
	}

	return candidates
}

func deduplicate(ids []int) []int {
	hash := make(map[int]bool)
	for _, id := range ids {
		hash[id] = true
	}

	result := make([]int, 0, len(hash))
	for id := range hash {
		result = append(result, id)
	}

	return result
}

func minCosine(hits []Hit, threshold float32) []Hit {
	result := make([]Hit, 0, len(hits))
	for _, hit := range hits {
		if hit.Cosine >= threshold {
			result = append(result, hit)
		}
	}
	return result
}

// Hit result for an NN
type Hit struct {
	ID     int
	Vector *[]float32
	Cosine float32
}

// ByScore sorts hits descending by score
type ByScore []Hit

func (a ByScore) Len() int           { return len(a) }
func (a ByScore) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByScore) Less(i, j int) bool { return a[i].Cosine > a[j].Cosine }

func (l *Lsh) knn(vector []float32, candidates []int, k int) ([]Hit, error) {
	hits := make([]Hit, len(candidates), len(candidates))
	for i, id := range candidates {
		vec := (*l.vectors)[id]
		cosine, err := cosine(vector, vec)
		if err != nil {
			return []Hit{}, fmt.Errorf("error computing knn %q", err)
		}

		hits[i] = Hit{id, &vec, cosine}
	}

	sortHits(&hits)

	if len(hits) > k {
		hits = hits[0:k]
	}

	return hits, nil
}

func sortHits(hits *[]Hit) {
	sort.Sort(ByScore(*hits))
}
