package kmeans

import "iter"

type NormalizeObservationAdapter[T Number] struct {
	mins         []T
	maxes        []T
	observations Observations[T]
	scale        []T
}

func NewNormalizeObservationAdapter[T Number](oo Observations[T], scale []T) *NormalizeObservationAdapter[T] {
	mins, maxes := ObservationRange(oo.Observations(), oo.Degree())

	adapter := NormalizeObservationAdapter[T]{
		mins:         mins,
		maxes:        maxes,
		observations: oo,
		scale:        scale,
	}
	return &adapter
}

func (n *NormalizeObservationAdapter[T]) Degree() int {
	return len(n.mins)
}

type observationValues[T Number] []T

func (o observationValues[T]) Values(i int) T {
	return o[i]
}

// Normalize transforming the observation vector to values between 0 and 1 based on the initial
// population of data provided. It will also apply scaling
func (n NormalizeObservationAdapter[T]) Normalize(ov Observation[T]) []T {
	degree := len(n.mins)
	normalized := make([]T, degree)
	if n.scale != nil {
		for i := range degree {
			normalized[i] = (1.0 + n.scale[i]) * (ov.Values(i) - n.mins[i]) / (n.maxes[i] - n.mins[i])
		}
	} else {
		for i := range degree {
			normalized[i] = (ov.Values(i) - n.mins[i]) / (n.maxes[i] - n.mins[i])
		}
	}
	return normalized
}

func (n NormalizeObservationAdapter[T]) Denormalize(o Observation[T]) []T {
	degree := len(n.mins)
	denormalized := make([]T, degree)
	if n.scale != nil {
		for i := range degree {
			denormalized[i] = o.Values(i) / (1 + n.scale[i])
		}
	}
	for i := range degree {
		denormalized[i] = ((o.Values(i) * (n.maxes[i] - n.mins[i])) + n.mins[i])
	}
	return denormalized
}

func (n NormalizeObservationAdapter[T]) Observations() iter.Seq[Observation[T]] {
	return func(yield func(Observation[T]) bool) {
		for ov := range n.observations.Observations() {
			normalized := n.Normalize(ov)
			if !yield(observationValues[T](normalized)) {
				return
			}
		}
	}
}

// NormalizedObservation makes a copy of the provided Observation maintaining a reference
// to the original unmodified observation without.
type NormalizedObservation[T, O Number] struct {
	observation []T
	Original    *Observation[O]
}

func (o NormalizedObservation[T, O]) Values(i int) T {
	return o.observation[i]
}

type observations[T, O Number] []NormalizedObservation[T, O]

func (o observations[T, O]) Observations() iter.Seq[Observation[T]] {
	return func(yield func(Observation[T]) bool) {
		for _, o := range o {
			var co Observation[T]
			co = o
			if !yield(co) {
				return
			}
		}
	}
}
