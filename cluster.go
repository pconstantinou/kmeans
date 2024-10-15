package kmeans

import "math"

// A Cluster which data points gravitate around
type Cluster[T Number] struct {
	Center       Observation[T]
	Observations *ObservationList[T]
	sum          []float64
}

type centerObservation[T Number] []T

func (c centerObservation[T]) Values(i int) T {
	return c[i]
}

func (c *Cluster[T]) MostCentral() Observation[T] {
	d := math.MaxFloat64
	ci := 0
	for i, o := range c.Observations.ClusterObservations {
		di := Distance(o, c.Center, c.Observations.d)
		if di < d {
			ci = i
			d = di
		}
	}
	return c.Observations.ClusterObservations[ci]
}

// Reset removes all elements from the clustered and removes the center
func (c *Cluster[T]) Reset() {
	c.Observations.clear()
	c.Center = nil
	c.sum = nil
}

// Recenter updates the customer center a cluster
func (c *Cluster[T]) Recenter() {
	center, err := Center(c.Observations.All(), c.Observations.d)
	if err != nil {
		return
	}
	c.Center = centerObservation[T](center)
}

// Append adds an observation to the Cluster and recenters the cluster
func (c *Cluster[T]) Append(o Observation[T]) {
	c.Observations.Append(o)
	if c.sum == nil {
		c.sum = make([]float64, c.Observations.Degree())
	}
	center := make([]T, c.Observations.Degree())
	l := len(c.Observations.ClusterObservations)
	for i := range c.Observations.Degree() {
		c.sum[i] += float64(o.Values(i))
		center[i] = T(float64(c.sum[i] / float64(l)))
	}
	c.Center = centerObservation[T](center)
}

// SumOfDistance computes the sum of the distance of all the observations
// from the center of the cluster
func (c *Cluster[T]) SumOfDistance() float64 {
	var d float64
	count := 0
	for _, o := range c.Observations.ClusterObservations {
		d += Distance(o, c.Center, c.Observations.d)
		count++
	}
	if count == 0 {
		return 0
	}
	return d
}
