A simple go library for approximate nearest neighbours (ANN).

#Background
In many computational problems such as NLP, Recommendation Systems and Search, items are represented as vectors in a multidimensional space. Then given a specific item it's nearest neighbours need to be find e.g. given a query find the most dimilar ones. 
A naive liner scan over the data set might be too slow for larger data sets. 

Hence, more efficient algorithms are needed. One of the most common approach is to use Locality Sensitive Hashing (LSH). This family of algorithms are very fast but might not give the exact neirest neighbours and are hence called approximate nearest neighbours. The trade off between accuracy and speed is set via parameters of the algorithm. 

#GoLsh
is a library that finds approximate nearest neighbours based on the cosine distance between two vectors ([Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)).


#Algorithm
the basic idea of the algorithm is very simple. Find a hash encodind for every vector such that similar vectors have the same hash value. Hence, finding similar items boils down to finding vectors with the same hash value which can be done very efficient using a hash table. The algorithm proceeds as follows:


1. Init the lsh: for every vector in the corpus generate a hash function by randomly splitting the space using hyperplanes
2. the number of hyperplanes used is defined by **d**
	1. the larger **d** the less vectors have to be searched 
	2. however if this number is set too high no results will be found
3. in order to allow better results **numEmbeddings** defines the number of hash functions which are generated per vector. 

#Usage
the library is initialized as follows:

	golsh.NewLsh(vectors *map[int][]float64, numEmbeddings int, d int) Lsh
	
**vectors** is a simple go map from an user defined id to the input vectors to be searched against. This function will return an golsh.Lsh object which is used for all subsequent operations. The two parameters **numEmbeddings** and **d** controll the trade off between speed and accuracy (see above).

## Get vector based on id
	
	lsh.Vector(id int) ([]float64, bool)
	
given an id will return the vector stored. This is just a convinience function and may be used if the vector to search with is part of the corpus itself.
	
## Find approximate nearest neighbours
	
	lsh.Ann(vector []float64, k int, threshold float64) ([]Hit, int, error)
	
**vector** the vector to search nearest neighbours for. **k** max number of neighbours returned. **threshold** min cosine similarity that a neighbour needs to have. This parameter is used to filter false positives.

## Result
the result is of type []golsh.Hit where Hit consists of:

	type Hit struct {
		ID     int
		Vector *[]float64
		Cosine float64
	}
	
where **ID** is the id of the result vector. **Vector** is the result vector itself and **Cosine** is the exact cosine distance between the query and this result vector. The result array is sorted by similarity that is most similar vector is pos 1.