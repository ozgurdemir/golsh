package golsh

import (
	"fmt"
	"sort"
)

// Lsh creates a new Lsh object
type Lsh struct {
	vectors    *map[int]Vector
	embeddings []embedding
	hash       map[string][]int
}

// NewLsh created a new Lsh object
func NewLsh(vectors *map[int]Vector, numEmbeddings int, d int) Lsh {
	return newLsh(vectors, numEmbeddings, d, &gauss{})
}

func newLsh(vectors *map[int]Vector, numEmbeddings int, d int, r random) Lsh {
	size := getSize(vectors)

	// create global embeddings
	embeddings := make([]embedding, numEmbeddings, numEmbeddings)
	for i := 0; i < numEmbeddings; i++ {
		embeddings[i] = newEmbedding(d, size, r)
	}

	// embed input vectors
	hash := make(map[string][]int)
	for id, vector := range *vectors {
		for _, embedding := range embeddings {
			h := embedding.embed(vector)
			hash[h] = append(hash[h], id)
		}
	}

	return Lsh{vectors, embeddings, hash}
}

func getSize(vectors *map[int]Vector) int {
	for _, vector := range *vectors {
		return len(vector)
	}
	return 0
}

// Vector fetches the vector for a given id
func (l *Lsh) Vector(id int) (Vector, bool) {
	vector, ok := (*l.vectors)[id]
	return vector, ok
}

// Ann finds approximate nearest neughbour using LSH cosine
func (l *Lsh) Ann(vector Vector, k int) ([]Hit, error) {
	candidates := l.candidates(vector)
	nn, err := l.knn(vector, deduplicate(candidates), k)
	return nn, err
}

func (l *Lsh) candidates(vec Vector) []int {
	candidates := make([]int, 0, 100)
	for _, embedding := range l.embeddings {
		h := embedding.embed(vec)
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

// Hit result for an NN
type Hit struct {
	ID     int
	Vector *Vector
	Cosine float64
}

// ByScore sorts hits descending by score
type ByScore []Hit

func (a ByScore) Len() int           { return len(a) }
func (a ByScore) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByScore) Less(i, j int) bool { return a[i].Cosine > a[j].Cosine }

func (l *Lsh) knn(vector Vector, candidates []int, k int) ([]Hit, error) {
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
