package golsh

import (
	"log"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

var (
	r       = newFakeRandom([]float64{1, 2, 3})
	vectors = [][]float64{
		[]float64{-1.0, 0.0, 0.0},
		[]float64{0.0, 1.0, 0.0},
		[]float64{0.0, 0.0, 1.0},
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
	if got, expected := lsh.embeddings[0].normals[0], []float64{1, 2, 3}; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}

}

func TestVector(t *testing.T) {
	vector := lsh.Vector(0)
	if got, expected := vector, vectors[0]; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestCandidates(t *testing.T) {
	if got, expected := deduplicate(lsh.candidates(vectors[2])), []int{1, 2}; !reflect.DeepEqual(got, expected) {
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

func TestMinCosine(t *testing.T) {
	vec := vectors[0]
	hits := []Hit{
		Hit{1, vec, 0.1},
		Hit{1, vec, 0.6},
		Hit{1, vec, 0.8},
	}
	filteredHits := []Hit{
		Hit{1, vec, 0.8},
		Hit{1, vec, 0.6},
	}
	sort.Sort(ByScore(hits))
	if got, expected := minCosine(hits, 0.6), filteredHits; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestSort(t *testing.T) {
	vec := vectors[0]
	hits := []Hit{
		Hit{1, vec, 5.0},
		Hit{1, vec, 2.0},
		Hit{1, vec, 4.0},
	}
	sortHits(&hits)
	sortedHits := []Hit{
		Hit{1, vec, 5.0},
		Hit{1, vec, 4.0},
		Hit{1, vec, 2.0},
	}
	if got, expected := hits, sortedHits; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestKNN(t *testing.T) {
	candidates := []int{0, 1, 2}
	result, err := lsh.knn([]float64{1, 1, 1}, candidates, 2)
	if err != nil {
		t.Fatalf("error computing knn %q", err)
	}
	if got, expected := 2, len(result); got != expected {
		t.Fatalf("expected %v but got %v", expected, got)
	}

	if expected, got := []int{1, 2}, []int{result[0].ID, result[1].ID}; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestKNN1(t *testing.T) {
	candidates := []int{0, 1, 2}
	result, err := lsh.knn([]float64{-1, -1, -1}, candidates, 1)
	if err != nil {
		t.Fatalf("error computing knn %q", err)
	}
	if got, expected := len(result), 1; got != expected {
		t.Fatalf("expected %v but got %v", expected, got)
	}

	if expected, got := []int{0}, []int{result[0].ID}; !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func BenchmarkDeduplicate(b *testing.B) {
	data := make([]int, 1500)
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		for k := 0; k < len(data); k++ {
			data[k] = rand.Intn(20000)
		}
		b.StartTimer()
		deduplicate(data)
	}
}

func randomVector(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = rand.Float64()
	}
	return v
}

func BenchmarkLSH(b *testing.B) {
	vectors := make([][]float64, 100000)
	cols := 200
	count := 30
	for i := range vectors {
		vectors[i] = randomVector(cols)
	}

	r := NewLsh(vectors, 3, 18)
	b.ResetTimer()
	/*
		sames := 0
		tests := 0
	*/
	for i := 0; i < b.N; i++ {
		for k, v := range vectors {
			if k%1000 == 0 {
				log.Println(k)
			}
			hits, candidates, err := r.Ann(v, count, 0.1)
			if len(hits) != count {
				log.Printf("expected %d results, got: %d at %d/%d. tested %d candidates", count, len(hits), i, k, candidates)
			}
			if err != nil {
				b.Fatal(err)
			}
			/*
				tests++
					same := true
					b.StopTimer()
					for k, val := range v {
						if math.Abs(float64(val)-float64(hits[0].Vector[k])) > 0.001 {
							same = false
							break
						}
					}
					b.StartTimer()
					if same {
						sames++
					}
			*/

		}
	}

}
