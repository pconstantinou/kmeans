# kmeans

provides a modern go implementation of the kmeans clustering algorithm.

This algorithm distinguishes itself from other kmeans clustering packages with use of generic data types
and iterators. Using these new language features can make it easier to integrate the clustering with 
existing code and structs without explicitly duplicating existing objects into arrays.

Use of iterators of observations (e.g. `iter.Seq[Observation[float64]]`) also provides 
a mechanism that allows data to be delivered incrementally.


# Usage

## Define your observation


Provide a struct that implements `Values() []T` where T can be any Number type. 

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

func (p person) Values() []float64 {
	return []float64{
		p.age(),
		float64(p.height_inches),
		float64(p.weight_lbs),
		float64(p.gender),
	}
}
```

## Implement your observation collection. Your observation collection 

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
	return len(p[0].Values())
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
    p := newPerson("kamala", "1964-10-20", 64, 130, 1)
    clusterIndex := cc.Nearest(p)
    fmt.Println("Best cluster is ", clusterIndex)

```