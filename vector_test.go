package golsh

import "testing"

func TestCosine(t *testing.T) {
	v123 := []feature{1.0, 2.0, 3.0}

	actual, err := cosine(v123, v123)
	if err != nil {
		t.Fatalf("failed to get cosine: %s", err.Error())
	}

	if expected := feature(1.0); expected != actual {
		t.Fatalf("expected %f but got %f distance", expected, actual)
	}

	actual, err = cosine(v123, []feature{1.0, 0.0, 0.0})
	if err != nil {
		t.Fatalf("failed to get cosine: %s", err.Error())
	}

	if expected := feature(0.2672612419124244); expected != actual {
		t.Fatalf("expected %f but got %f distance", expected, actual)
	}

	actual, err = cosine([]feature{0.0, 1.0}, []feature{1.0, 0.0})
	if err != nil {
		t.Fatalf("failed to get cosine: %s", err.Error())
	}

	// orthogonal vectors should have 0 similarity
	if expected := feature(0.0); expected != actual {
		t.Fatalf("expected %f but got %f distance", expected, actual)
	}
}
