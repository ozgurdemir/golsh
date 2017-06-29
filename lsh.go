package golsh

import (
	"fmt"
	"sort"
)

// Lsh creates a new Lsh object
type Lsh struct {
	vectors    [][]float64
	embeddings []embedding
	hash       map[uint64][]int
}

// NewLsh created a new Lsh object
func NewLsh(vectors [][]float64, numEmbeddings int, d int) Lsh {
	return newLsh(vectors, numEmbeddings, d, &gauss{})
}

func newLsh(vectors [][]float64, numEmbeddings int, d int, r random) Lsh {
	size := getSize(vectors)

	// create global embeddings
	embeddings := make([]embedding, numEmbeddings, numEmbeddings)
	for i := 0; i < numEmbeddings; i++ {
		embeddings[i] = newEmbedding(d, size, r)
	}

	// embed input vectors
	hash := make(map[uint64][]int)
	for id, vector := range vectors {
		for _, embedding := range embeddings {
			h := embedding.embed(vector)
			hash[h] = append(hash[h], id)
		}
	}

	return Lsh{vectors, embeddings, hash}
}

func getSize(vectors [][]float64) int {
	for _, vector := range vectors {
		return len(vector)
	}
	return 0
}

// Vector fetches the vector for a given id
func (l *Lsh) Vector(id int) []float64 {
	return l.vectors[id]
}

// Ann finds approximate nearest neughbour using LSH cosine
func (l *Lsh) Ann(vector []float64, k int, threshold float64) ([]Hit, int, error) {
	candidates := l.candidates(vector)
	hits, err := l.knn(vector, deduplicate(candidates), k)
	hits = minCosine(hits, threshold)
	return hits, len(candidates), err
}

func (l *Lsh) candidates(vec []float64) []int {
	candidates := make([]int, 0, 100)
	for _, embedding := range l.embeddings {
		h := embedding.embed(vec)
		candidates = append(candidates, l.hash[h]...)
	}

	return candidates
}

func deduplicate(ids []int) []int {
	hash := make(map[int]struct{}, len(ids)/2)
	result := make([]int, 0, len(ids))
	var empty struct{}
	for _, id := range ids {
		if _, ok := hash[id]; !ok {
			result = append(result, id)
			hash[id] = empty
		}
	}
	return result
}

func minCosine(hits []Hit, threshold float64) []Hit {
	// we know these are sorted so just return those up to the first that's below the threshold.
	for i, hit := range hits {
		if hit.Cosine < threshold {
			return hits[:i]
		}
	}
	return hits
}

// Hit result for an NN
type Hit struct {
	ID     int
	Vector []float64
	Cosine float64
}

// ByScore sorts hits descending by score
type ByScore []Hit

func (a ByScore) Len() int           { return len(a) }
func (a ByScore) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByScore) Less(i, j int) bool { return a[i].Cosine > a[j].Cosine }

func (l *Lsh) knn(vector []float64, candidates []int, k int) ([]Hit, error) {
	hits := make([]Hit, len(candidates))
	for i, id := range candidates {
		vec := l.vectors[id]
		cosine, err := cosine(vector, vec)
		if err != nil {
			return []Hit{}, fmt.Errorf("error computing knn %q", err)
		}

		hits[i] = Hit{id, vec, cosine}
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
