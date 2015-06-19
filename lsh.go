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

// AnnVector returns the ann vectors
func (l *Lsh) AnnVector(vector Vector, k int) ([]Vector, error) {
	nn, err := l.Ann(vector, k)
	vectors := make([]Vector, len(nn), len(nn))
	if err != nil {
		return vectors, err
	}
	for i, id := range nn {
		result, ok := l.Vector(id)
		if !ok {
			return []Vector{}, fmt.Errorf("vector not found %d", id)
		}
		vectors[i] = result
	}
	return vectors, err
}

// Ann finds approximate nearest neughbour using LSH cosine
func (l *Lsh) Ann(vector Vector, k int) ([]int, error) {
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

type hit struct {
	id     int
	vector Vector
	score  float64
}

type byScore []hit

func (a byScore) Len() int           { return len(a) }
func (a byScore) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byScore) Less(i, j int) bool { return a[i].score > a[j].score }

func (l *Lsh) knn(vector Vector, candidates []int, k int) ([]int, error) {
	hits := make([]hit, len(candidates), len(candidates))
	for i, id := range candidates {
		vec := (*l.vectors)[id]
		cosine, err := cosine(vector, vec)
		if err != nil {
			return []int{}, fmt.Errorf("error computing knn %q", err)
		}

		hits[i] = hit{id, vec, cosine}
	}

	sortHits(&hits)

	result := make([]int, k, k)
	for i := 0; i < k; i++ {
		result[i] = hits[i].id
	}

	return result, nil
}

func sortHits(hits *[]hit) {
	sort.Sort(byScore(*hits))
}
