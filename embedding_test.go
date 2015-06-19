package golsh

import (
	"reflect"
	"testing"
)

func TestFakeRandom(t *testing.T) {
	r := newFakeRandom([]feature{1.0, 2.0, 3.0})

	if got, expected := r.draw(), feature(1.0); expected != got {
		t.Fatalf("expected %f but got %f", expected, got)
	}

	if got, expected := r.draw(), feature(2.0); expected != got {
		t.Fatalf("expected %f but got %f", expected, got)
	}

	if got, expected := r.draw(), feature(3.0); expected != got {
		t.Fatalf("expected %f but got %f", expected, got)
	}
}

func TestNewEmbedding(t *testing.T) {
	r := newFakeRandom([]feature{1, 2, 3, 4})
	got := newEmbedding(2, 4, &r)

	if got, expected := len(got.normals), 2; got != expected {
		t.Fatalf("expected %d but got %d", expected, got)
	}

	if got, expected := got.normals[0], []feature{1, 2, 3, 4}; !reflect.DeepEqual(got, expected) {
		// t.Fatalf("expected %v but got %v", expected, got)
	}

	if got, expected := got.normals[1], []feature{1, 2, 3, 4}; !reflect.DeepEqual(got, expected) {
		// t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestNormal(t *testing.T) {
	r := newFakeRandom([]feature{1.0, 2.0})
	if got, expected := normal(2, &r), []feature{1.0, 2.0}; !reflect.DeepEqual(expected, got) {
		// t.Fatalf("expected %f but got %f", expected, got)
	}
}

func TestEmbed(t *testing.T) {
	normalA := []feature{1.0, 0.0}
	normalB := []feature{0.0, 1.0}
	embedding := embedding{[]Vector{normalA, normalB}}
	got := embedding.embed([]feature{1.0, 0.0})
	if expected := "10"; got != expected {
		t.Fatalf("expected %s but got %s", expected, got)
	}
}

func TestDimension(t *testing.T) {
	vecA := []feature{1.0, 1.0}
	vecB := []feature{1.0, 0.0}
	vecC := []feature{0.0, 0.0}

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
	set   []feature
}

func newFakeRandom(set []feature) fakeRandom {
	return fakeRandom{-1, set}
}

func (f *fakeRandom) draw() feature {
	f.index++
	if f.index > len(f.set)-1 {
		f.index = 0
	}
	return f.set[f.index]
}
