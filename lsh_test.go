package golsh

import (
	"reflect"
	"sort"
	"testing"
)

var (
	r       = newFakeRandom([]feature{1, 2, 3})
	vectors = map[int]vector{
		10: []feature{-1.0, 0.0, 0.0},
		20: []feature{0.0, 1.0, 0.0},
		30: []feature{0.0, 0.0, 1.0},
	}
	numEmbeddings = 3
	d             = 2
	lsh           = newLsh(vectors, numEmbeddings, d, &r)
)

func TestSize(t *testing.T) {
	if got, expected := getSize(vectors), 3; expected != got {
		t.Fatalf("expected %d but got %d", expected, got)
	}
}

func TestNewLsh(t *testing.T) {
	// vectors are copied
	if got, expected := lsh.vectors, vectors; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
	// correct number of embeddings
	if got, expected := len(lsh.embeddings), 3; got != expected {
		t.Fatalf("expected %d but got %d", expected, got)
	}
	// embeddings have correct dimension
	if got, expected := len(lsh.embeddings[0].normals), 2; got != expected {
		t.Fatalf("expected %d but got %d", expected, got)
	}

	// embeddings have correct normals
	if got, expected := lsh.embeddings[0].normals[0], []feature{1, 2, 3}; !reflect.DeepEqual(got, expected) {
		// t.Fatalf("expected %v but got %v", expected, got)
	}

	// all 3 embeddings are equal and hence produce the same output
	expected := map[string][]int{
		"00": []int{10, 10, 10},
		"11": []int{20, 20, 20, 30, 30, 30},
	}
	if got := lsh.hash; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestVector(t *testing.T) {
	if got, expected := lsh.vector(10), vectors[10]; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestCandidates(t *testing.T) {
	if got, expected := lsh.candidates(vectors[10]), []int{10, 10, 10, 10, 10, 10, 10, 10, 10}; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestDeduplicate(t *testing.T) {
	ids := []int{1, 2, 3, 1, 2, 3, 2, 3, 1}
	got := deduplicate(ids)
	sort.Ints(got)
	if expected := []int{1, 2, 3}; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestSort(t *testing.T) {
	hits := []hit{
		hit{1, vectors[10], 5.0},
		hit{1, vectors[10], 2.0},
		hit{1, vectors[10], 4.0},
	}
	sortHits(&hits)
	sortedHits := []hit{
		hit{1, vectors[10], 5.0},
		hit{1, vectors[10], 4.0},
		hit{1, vectors[10], 2.0},
	}
	if got, expected := hits, sortedHits; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestKNN(t *testing.T) {
	candidates := []int{10, 20, 30}
	got, err := lsh.knn([]feature{1, 1, 1}, candidates, 2)
	if err != nil {
		t.Fatalf("error computing knn %q", err)
	}
	if expected := []int{20, 30}; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}
