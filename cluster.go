package kmeans

import "math"

// A Cluster which data points gravitate around
type Cluster[T Number] struct {
	Center       Observation[T]
	Observations *ObservationList[T]
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

// Recenter updates the customer center a cluster
func (c *Cluster[T]) Recenter() {
	center, err := Center(c.Observations.All(), c.Observations.d)
	if err != nil {
		return
	}
	c.Center = centerObservation[T](center)
}

// Append adds an observation to the Cluster
func (c *Cluster[T]) Append(o Observation[T]) {
	c.Observations.Append(o)
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
