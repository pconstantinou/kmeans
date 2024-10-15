package kmeans

import (
	"fmt"
	"math"

	"math/rand"
)

// Clusters is a slice of clusters
type Clusters[T Number] []Cluster[T]

var ErrKMustBeGreaterThanZero = fmt.Errorf("k must be greater than 0")
var ErrEmptyObservations error = fmt.Errorf("empty observation, there is no mean for an empty set of points")

// New sets up a new set of clusters and randomly seeds their initial positions
func New[T Number](k int, dataset Observations[T]) (Clusters[T], error) {
	var c Clusters[T]
	if dataset.Degree() == 0 {
		return c, ErrEmptyObservations
	}
	if k == 0 {
		return c, ErrKMustBeGreaterThanZero
	}

	for _, o := range SelectRandomObservations(dataset, k) {
		c = append(c, Cluster[T]{
			Center:       Observation[T](o),
			Observations: NewObservationList[T](dataset.Degree()),
		})
	}
	return c, nil
}

func SelectRandomObservations[T Number](oo Observations[T], k int) []Observation[T] {
	roo := make([]Observation[T], k)
	count := 0
	for o := range oo.Observations() {
		if count < len(roo) {
			roo[count] = o
		} else {
			i := rand.Intn(count)
			if i < len(roo) {
				roo[i] = o
			}
		}
		count++
	}
	return roo
}

// Nearest returns the index of the cluster nearest to point
func (c Clusters[T]) Nearest(point Observation[T]) int {
	var ci int
	dist := -1.0

	// Find the nearest cluster for this data point
	for i, cluster := range c {
		d := Distance(point, cluster.Center, cluster.Observations.d)
		if dist < 0 || d < dist {
			dist = d
			ci = i
		}
	}

	return ci
}

// Neighbor returns the neighboring cluster of a point along with the average distance to its points
func (c Clusters[T]) Neighbor(point Observation[T], fromCluster int) (int, float64) {
	var d float64
	nc := -1

	for i, cluster := range c {
		if i == fromCluster {
			continue
		}

		cd := AverageDistance(point, cluster.Observations.All(), cluster.Observations.d)
		if nc < 0 || cd < d {
			nc = i
			d = cd
		}
	}

	return nc, d
}

// Recenter updates all cluster centers
func (c Clusters[T]) Recenter() {
	for i := 0; i < len(c); i++ {
		c[i].Recenter()
	}
}

// Reset clears all point assignments
func (c *Clusters[T]) Reset() {
	for i := 0; i < len(*c); i++ {
		(*c)[i].Reset()
	}
}

func (c Clusters[T]) SumClusterVariance() float64 {
	v := 0.0
	for _, cl := range c {
		v += cl.SumOfDistance()
	}
	return v
}

func (c Clusters[T]) Largest() int {
	s := 0
	for i := range c {
		if len(c[s].Observations.ClusterObservations) < len(c[i].Observations.ClusterObservations) {
			s = i
		}
	}
	return s
}

func (c Clusters[T]) Smallest() int {
	s := 0
	for i := range c {
		if len(c[s].Observations.ClusterObservations) > len(c[i].Observations.ClusterObservations) {
			s = i
		}
	}
	return s
}

// New sets up a new set of clusters and randomly seeds their initial positions.
// This function is exponential both in its computation of permutations for possible
// cluster centers O(k*n*degree) in cluster matching.
func OptimizeClusters[T Number](k int, dataset Observations[T]) (Clusters[T], error) {
	if dataset.Degree() == 0 {
		return nil, ErrEmptyObservations
	}
	if k == 0 {
		return nil, ErrKMustBeGreaterThanZero
	}

	perm := permutations(dataset, k)

	var optimalClusters Clusters[T]
	sum := math.MaxFloat64
	for _, centers := range perm {
		newC := makeCluster(dataset, centers)
		newSum := newC.SumClusterVariance()

		if sum > newSum {
			sum = newSum
			optimalClusters = newC
		}
	}
	return optimalClusters, nil
}

func permutations[T Number](dataset Observations[T], k int) [][]Observation[T] {
	if k >= 3 {
		return randomPermutations(dataset, k)
	}
	var ll []Observation[T]
	for o := range dataset.Observations() {
		ll = append(ll, o)
	}
	return mapList(nil, ll, k)
}

func randomPermutations[T Number](dataset Observations[T], k int) [][]Observation[T] {
	var ll [][]Observation[T]
	for range 1000 * k {
		ll = append(ll, SelectRandomObservations(dataset, k))
	}
	return ll
}

func mapList[T Number](in []Observation[T], remaining []Observation[T], l int) [][]Observation[T] {
	if len(in) >= l {
		return [][]Observation[T]{
			in,
		}
	}
	var rr [][]Observation[T]
	if len(in)+1 == l {
		for _, r := range remaining {
			rr = append(rr, append(in, r))
		}
	} else {
		for i, r := range remaining {
			newRemaining := make([]Observation[T], 0, l)
			newRemaining = append(newRemaining, remaining[:i]...)
			if len(remaining) > i+1 {
				newRemaining = append(newRemaining, remaining[i+1:]...)
			}
			rr = append(rr, mapList(append(in, r), newRemaining, l)...)
		}
	}
	return rr
}

func makeCluster[T Number](dataset Observations[T], centers []Observation[T]) Clusters[T] {
	var c Clusters[T]
	count := 0
	for _, o := range centers {
		c = append(c, Cluster[T]{
			Center:       Observation[T](o),
			Observations: NewObservationList[T](dataset.Degree()),
		})
		count++
	}
	for o := range dataset.Observations() {
		i := c.Nearest(o)
		c[i].Append(o)
	}
	return c
}
