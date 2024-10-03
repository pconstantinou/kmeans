[![codecov](https://codecov.io/github/pconstantinou/kmeans/graph/badge.svg?token=3G6C2KFPGN)](https://codecov.io/github/pconstantinou/kmeans)

# kmeans


provides a modern go implementation of the kmeans clustering algorithm.

This algorithm distinguishes itself from other kmeans clustering packages with use of generic data types
and iterators. Using these new language features can make it easier to integrate the clustering with 
existing code and structs without explicitly duplicating existing objects into arrays.

Use of iterators of observations (e.g. `iter.Seq[Observation[float64]]`) also provides 
a mechanism that allows data to be delivered incrementally.

See [3d scatter plot](https://pconstantinou.github.io/kmeans/scatter3d.html) to see a demo visualization.
This can be regenerated using `go run ./cmd/demo.go -k2 5 ; open scatter3d.html `

# Usage

## Define your observation


Provide a struct that implements `Values(i int) T` where T can be any Number type. 

```
type person struct {
	name          string
	birthDate     time.Time
	height_inches int
	weight_lbs    int
	gender        int
}

func newPerson(name string, birthDate string, height int, weight int, gender int) person {
	d, _ := time.Parse(time.DateOnly, birthDate)
	return person{
		name:          name,
		birthDate:     d,
		height_inches: height,
		weight_lbs:    weight,
		gender:        gender,
	}
}

func (p person) age() float64 {
	return float64(time.Since(p.birthDate) / (time.Hour * 24 * 365))
}


func (p person) Values(i int) float64 {
	switch i {
	case 0:
		return float64(p.weight_lbs)
	case 1:
		return p.age()
	case 2:
		return float64(p.height_inches)
	case 3:
		return float64(p.gender)
	}
	return 0.0
}

```

## Implement your observation collection. This collection is used to populate the clusters

```
type peopleObservations []person

func (p peopleObservations) Observations() iter.Seq[kmeans.Observation[float64]] {
	return func(yield func(kmeans.Observation[float64]) bool) {
		for _, o := range p {
			if !yield(o.(kmeans.Observation[float64])) {
				return
			}
		}
	}
}

func (p peopleObservations) Degree() int {
	return 4
}
```

## Build ClusterObservations

```
    var po personObservations
    // Generate k clusters
	cc, err := OptimizeClusters(k, po)
    largestIndex := cc.LargestIndex()
    // Print biggest cluster
    for o := range cc.Observations.ClusterObservations {
        fmt.Println(o)
    }
```

## Provide additional objects and find the best cluster to associated them with

```
    p := newPerson("kamala", "1964-10-20", 64, 130, 1)
    clusterIndex := cc.Nearest(p)
    fmt.Println("Best cluster is ", clusterIndex)

```