package kmeans

import (
	"iter"
	"math"
	"slices"
)

// Number provides a set of types used in the generic observation
type Number interface {
	int | int8 | int16 | int32 | int64 |
		uint | uint8 | uint16 | uint32 | uint64 |
		float32 | float64
}

// Observations must return an array of values where the length and meaning of each array
// is the same for all observations in a set.
// Users may either implement the observation interface or create a type or struct that wrap it
// Values may  be copied but are never modified
type Observation[T Number] interface {
	Values(i int) T
}

// Observations is a collection of Observation objects which may be provided as
// a sequence to prevent the requirement that caller implementor use a slice
type Observations[T Number] interface {
	Observations() iter.Seq[Observation[T]]
	// Degree is the number of values in each observation
	Degree() int
}

type ObservationList[T Number] struct {
	ClusterObservations []Observation[T]
	d                   int
}

func NewObservationList[T Number](degree int) *ObservationList[T] {
	return &ObservationList[T]{
		d: degree,
	}
}

func (o *ObservationList[T]) Append(v Observation[T]) {
	o.ClusterObservations = append(o.ClusterObservations, v)
}

func (o *ObservationList[T]) All() iter.Seq[Observation[T]] {
	return slices.Values(o.ClusterObservations)
}

func (o *ObservationList[T]) Degree() int {
	return o.d
}

func (o *ObservationList[T]) clear() {
	o.ClusterObservations = nil
}

func (o observations[T, O]) Degree() int {
	return len(o[0].observation)
}

func NormalizeObservations[O Number](oo Observations[O]) Observations[float64] {
	mins, maxes := ObservationRange(oo.Observations(), oo.Degree())
	ranges := make([]float64, len(mins))
	for i := range mins {
		ranges[i] = float64(maxes[i]) - float64(mins[i])
	}

	var noo observations[float64, O]
	for o := range oo.Observations() {
		nvv := make([]float64, oo.Degree())
		for i := range oo.Degree() {
			nvv[i] = (float64(o.Values(i)) - float64(mins[i])) / ranges[i]
		}
		noo = append(noo, NormalizedObservation[float64, O]{
			observation: nvv,
			Original:    &o,
		})
	}
	return noo
}

// ObservationSum returns slices the same size as the Degree of the observation. Each entry in the slice
// is the min and max of all the values for that slice's index into their `Value()`.
func ObservationRange[T Number](oo iter.Seq[Observation[T]], degree int) (observationMins []T, observationMax []T) {

	observationMax = make([]T, degree)
	observationMins = make([]T, degree)

	first := true
	for o := range oo {
		if first {
			for i := range degree {
				v := o.Values(i)
				observationMins[i] = v
				observationMax[i] = v
			}
			first = false
		} else {
			for i := range degree {
				v := o.Values(i)
				observationMins[i] = min(observationMins[i], v)
				observationMax[i] = max(observationMax[i], v)
			}
		}
	}
	return
}

// ObservationSum returns a slice the same size as the Degree of the observation. Each entry in the slice
// is the sum of all the values for that slice's index into their `Value()`. The count of the number of entries are
// also returned.
func ObservationSum[T Number](oo iter.Seq[Observation[T]], degree int) (sum []T, count int) {
	count = 0
	sum = make([]T, degree)
	for o := range oo {
		count++
		for i := range degree {
			sum[i] += o.Values(i)
		}
	}
	return
}

// Distance returns the euclidean distance between two coordinates
func Distance[T Number](o1, o2 Observation[T], degree int) float64 {
	var r float64
	for i := range degree {
		r += math.Pow(float64(o1.Values(i))-float64(o2.Values(i)), 2)
	}
	return r
}

func Center[T Number](os iter.Seq[Observation[T]], degree int) ([]T, error) {

	osSum, count := ObservationSum(os, degree)
	if count == 0 {
		return nil, ErrEmptyObservations
	}

	mean := make([]T, 0, len(osSum))
	for _, v := range osSum {
		mean = append(mean, T(float64(v)/float64(count)))
	}
	return mean, nil
}

// AverageDistance returns the average distance between o and all observations
func AverageDistance[T Number](o Observation[T], observations iter.Seq[Observation[T]], degree int) float64 {
	var d float64
	var l int

	for observation := range observations {
		dist := Distance(o, observation, degree)
		l++
		d += dist
	}

	if l == 0 {
		return 0
	}
	return d / float64(l)
}
