package golsh

import (
	"reflect"
	"sort"
	"testing"
)

var (
	r       = newFakeRandom([]float64{1, 2, 3})
	vectors = map[int]Vector{
		10: []float64{-1.0, 0.0, 0.0},
		20: []float64{0.0, 1.0, 0.0},
		30: []float64{0.0, 0.0, 1.0},
	}
	numEmbeddings = 3
	d             = 2
	lsh           = newLsh(&vectors, numEmbeddings, d, &r)
)

func TestSize(t *testing.T) {
	if got, expected := getSize(&vectors), 3; expected != got {
		t.Fatalf("expected %d but got %d", expected, got)
	}
}

func TestNewLsh(t *testing.T) {
	// vectors are copied
	if got, expected := *lsh.vectors, vectors; !reflect.DeepEqual(got, expected) {
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
	if got, expected := lsh.embeddings[0].normals[0], []float64{1, 2, 3}; !reflect.DeepEqual(got, expected) {
		// t.Fatalf("expected %v but got %v", expected, got)
	}

	// all 3 embeddings are equal (1,2,3) and hence produce the same output
	expected := map[string][]int{
		"0-00": []int{10},
		"0-11": []int{20, 30},
		"1-00": []int{10},
		"1-11": []int{20, 30},
		"2-00": []int{10},
		"2-11": []int{20, 30},
	}
	if got := lsh.hash; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestVector(t *testing.T) {
	vector, _ := lsh.Vector(10)
	if got, expected := vector, vectors[10]; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestCandidates(t *testing.T) {
	if got, expected := lsh.candidates(vectors[10]), []int{10, 10, 10}; !reflect.DeepEqual(got, expected) {
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
	vec := vectors[10]
	hits := []Hit{
		Hit{1, &vec, 5.0},
		Hit{1, &vec, 2.0},
		Hit{1, &vec, 4.0},
	}
	sortHits(&hits)
	sortedHits := []Hit{
		Hit{1, &vec, 5.0},
		Hit{1, &vec, 4.0},
		Hit{1, &vec, 2.0},
	}
	if got, expected := hits, sortedHits; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestKNN(t *testing.T) {
	candidates := []int{10, 20, 30}
	result, err := lsh.knn([]float64{1, 1, 1}, candidates, 2)
	if err != nil {
		t.Fatalf("error computing knn %q", err)
	}
	if got, expected := 2, len(result); got != expected {
		t.Fatalf("expected %v but got %v", expected, got)
	}

	if got, expected := []int{20, 30}, []int{result[0].ID, result[1].ID}; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}
