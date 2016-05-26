package golsh

import (
	"reflect"
	"testing"
)

func TestFakeRandom(t *testing.T) {
	r := newFakeRandom([]float32{1.0, 2.0, 3.0})

	if got, expected := r.draw(), float32(1.0); expected != got {
		t.Fatalf("expected %f but got %f", expected, got)
	}

	if got, expected := r.draw(), float32(2.0); expected != got {
		t.Fatalf("expected %f but got %f", expected, got)
	}

	if got, expected := r.draw(), float32(3.0); expected != got {
		t.Fatalf("expected %f but got %f", expected, got)
	}
}

func TestNewEmbedding(t *testing.T) {
	r := newFakeRandom([]float32{1, 2, 3, 4})
	got := newEmbedding(2, 4, &r)

	if got, expected := len(got.normals), 2; got != expected {
		t.Fatalf("expected %d but got %d", expected, got)
	}

	if got, expected := got.normals[0], []float32{1, 2, 3, 4}; !reflect.DeepEqual(got, expected) {
		// t.Fatalf("expected %v but got %v", expected, got)
	}

	if got, expected := got.normals[1], []float32{1, 2, 3, 4}; !reflect.DeepEqual(got, expected) {
		// t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestNormal(t *testing.T) {
	r := newFakeRandom([]float32{1.0, 2.0})
	if got, expected := normal(2, &r), []float32{1.0, 2.0}; !reflect.DeepEqual(expected, got) {
		// t.Fatalf("expected %f but got %f", expected, got)
	}
}

func TestEmbed(t *testing.T) {
	normalA := []float32{1.0, 0.0}
	normalB := []float32{0.0, 1.0}
	embedding := embedding{[][]float32{normalA, normalB}}
	got := embedding.embed(1, []float32{1.0, 0.0})
	if expected := "1-10"; got != expected {
		t.Fatalf("expected %s but got %s", expected, got)
	}
}

func TestDimension(t *testing.T) {
	vecA := []float32{1.0, 1.0}
	vecB := []float32{1.0, 0.0}
	vecC := []float32{0.0, 0.0}

	if got, expected := dimension(vecA, vecB), true; !reflect.DeepEqual(expected, got) {
		t.Fatalf("expected %t but got %t", expected, got)
	}

	if got, expected := dimension(vecA, vecC), false; !reflect.DeepEqual(expected, got) {
		t.Fatalf("expected %t but got %t", expected, got)
	}
}

func TestBitString(t *testing.T) {
	got := bitToString([]bool{true, false, true, true})
	if expected := "1011"; got != expected {
		t.Fatalf("expected %s but got %s", expected, got)
	}
}

type fakeRandom struct {
	index int
	set   []float32
}

func newFakeRandom(set []float32) fakeRandom {
	return fakeRandom{-1, set}
}

func (f *fakeRandom) draw() float32 {
	f.index++
	if f.index > len(f.set)-1 {
		f.index = 0
	}
	return f.set[f.index]
}
