package kmeans

import (
	"fmt"
	"iter"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

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

func (p person) Values(i int) float32 {
	return []float32{
		float32(p.birthDate.Unix()),
		float32(p.height_inches),
		float32(p.weight_lbs),
		float32(p.gender),
	}[i]
}

func (p person) String() string {
	return fmt.Sprintf("%10s DOB: %s Height %3d in. Weight %3d lbs",
		p.name, p.birthDate.Format(time.DateOnly), p.height_inches, p.weight_lbs)
}

func listOfPeople() peopleObservations {
	return peopleObservations([]person{
		newPerson("phil", "1972-01-17", 73, 185, 1),
		newPerson("dave", "1971-10-17", 73, 165, 1),
		newPerson("lora", "1966-12-17", 70, 121, 0),
		newPerson("dean", "1968-12-17", 71, 150, 1),
		newPerson("cate", "2010-05-10", 60, 95, 0),
		newPerson("tessa", "2010-09-10", 61, 105, 0),
		newPerson("tony", "1965-04-01", 73, 255, 1),
		newPerson("chris", "1939-07-22", 69, 215, 1),
		newPerson("janet", "1942-11-12", 60, 120, 0),
		newPerson("sophie", "1969-08-22", 62, 130, 0),
		newPerson("jenny", "1970-04-16", 62, 130, 0),
		newPerson("marlene", "1971-09-16", 62, 120, 0),
		newPerson("jenna", "1970-04-16", 62, 130, 0),
		newPerson("jenna", "1970-04-16", 62, 130, 0),
		newPerson("lora", "1966-12-17", 70, 121, 0),
		newPerson("john", "1982-04-11", 72, 180, 1),
		newPerson("maria", "1990-07-22", 64, 135, 0),
		newPerson("michael", "1975-03-03", 68, 175, 1),
		newPerson("susan", "1985-09-10", 65, 140, 0),
		newPerson("david", "1995-11-25", 73, 190, 1),
		newPerson("anna", "1989-01-14", 62, 125, 0),
		newPerson("james", "1970-05-30", 74, 200, 1),
		newPerson("karen", "1968-08-19", 66, 145, 0),
		newPerson("robert", "1981-02-04", 69, 165, 1),
		newPerson("linda", "1978-06-07", 63, 130, 0),
		newPerson("richard", "1992-12-21", 71, 185, 1),
		newPerson("nancy", "1993-03-15", 67, 150, 0),
		newPerson("charles", "1969-10-08", 70, 170, 1),
		newPerson("barbara", "1987-11-03", 64, 137, 0),
		newPerson("joseph", "1973-04-28", 72, 180, 1),
		newPerson("patricia", "1991-05-06", 62, 120, 0),
		newPerson("thomas", "1983-07-12", 73, 195, 1),
		newPerson("sandra", "1967-09-24", 65, 138, 0),
		newPerson("christopher", "1996-01-19", 74, 205, 1),
		newPerson("betty", "1980-02-23", 66, 142, 0),
		newPerson("daniel", "1977-08-31", 68, 178, 1),
		newPerson("dorothy", "1994-06-18", 63, 128, 0),
		newPerson("paul", "1988-12-27", 70, 175, 1),
		newPerson("elizabeth", "1974-11-14", 64, 132, 0),
		newPerson("mark", "1990-03-09", 71, 185, 1),
		newPerson("mary", "1982-10-26", 62, 123, 0),
		newPerson("steven", "1976-07-05", 73, 193, 1),
		newPerson("judy", "1986-04-16", 65, 140, 0),
		newPerson("kevin", "1965-01-30", 69, 167, 1),
	})
}

type peopleObservations []person

func (p peopleObservations) Observations() iter.Seq[Observation[float32]] {
	return func(yield func(Observation[float32]) bool) {
		for _, o := range p {
			if !yield(o) {
				return
			}
		}
	}
}

func (p peopleObservations) Degree() int {
	return 4
}

func TestObservationRange(t *testing.T) {
	personObservations := listOfPeople()
	oMin, oMax := ObservationRange(personObservations.Observations(), personObservations.Degree())
	assert.Len(t, oMin, personObservations.Degree())
	assert.Len(t, oMax, personObservations.Degree())
	for i := range oMin {
		assert.LessOrEqual(t, oMin[i], oMax[i])
	}
}

func TestObservationSum(t *testing.T) {
	personObservations := listOfPeople()
	sums, c := ObservationSum(personObservations.Observations(), personObservations.Degree())
	assert.Len(t, sums, personObservations.Degree())
	assert.Len(t, personObservations, c)
}

func TestNormalizeObservations(t *testing.T) {
	personObservations := listOfPeople()
	normal := NormalizeObservations(personObservations)
	assert.Len(t, normal, len(personObservations))
}

func TestClusters(t *testing.T) {
	oo := listOfPeople()
	c, err := New(2, oo)
	assert.NoError(t, err)

	for o := range oo.Observations() {
		i := c.Nearest(o)
		c[i].Append(o)
	}
}

func TestBestClusters(t *testing.T) {
	oo := listOfPeople()
	noo := NormalizeObservations(oo)
	var vv []float64
	for k := 2; k <= 7; k++ {
		cc, err := OptimizeClusters(k, noo)
		vv = append(vv, cc.SumClusterVariance())
		assert.NoError(t, err)
		assert.NotEmpty(t, cc)
		for i, cl := range cc {
			fmt.Printf("\nCluster %d Size: %d Variance: %2.5f \n", i, len(cl.Observations.ClusterObservations), cl.SumOfDistance())
			for _, o := range cl.Observations.ClusterObservations {
				source := o.(NormalizedObservation[float64, float32]).Original
				fmt.Println((*source))
			}
		}
		bi := cc.Largest()
		mostCentral := cc[bi].MostCentral().(NormalizedObservation[float64, float32]).Original
		fmt.Println("Most Central", *mostCentral)
	}
	for i, v := range vv {
		fmt.Println(i+2, ",", v)
	}
}

type noValues int
type noObservations int

func (n noValues) Values(i int) int {
	return 0
}

func (n noObservations) Observations() iter.Seq[Observation[int]] {
	var aa []noValues
	for range 5 {
		aa = append(aa, noValues(0))
	}
	return func(yield func(Observation[int]) bool) {
		for _, o := range aa {
			if !yield(Observation[int](o)) {
				return
			}
		}
	}
}

func (n noObservations) Degree() int {
	return 0
}

func TestClustersNormalizedErrorCheck(t *testing.T) {
	oo := listOfPeople()
	noo := NewNormalizeObservationAdapter(oo, nil)

	_, err := OptimizeClusters(0, noo)
	assert.ErrorIs(t, err, ErrKMustBeGreaterThanZero)
	_, err = New(0, noo)
	assert.ErrorIs(t, err, ErrKMustBeGreaterThanZero)

	_, err = OptimizeClusters(2, noObservations(1))
	assert.ErrorIs(t, err, ErrEmptyObservations)
	_, err = New(2, noObservations(1))
	assert.ErrorIs(t, err, ErrEmptyObservations)
}

func TestBestClustersNormalized(t *testing.T) {
	oo := listOfPeople()
	noo := NewNormalizeObservationAdapter(oo, nil)
	var vv []float64
	for k := 2; k <= 7; k++ {
		cc, err := OptimizeClusters(k, noo)
		vv = append(vv, cc.SumClusterVariance())
		assert.NoError(t, err)
		assert.NotEmpty(t, cc)
		for i, cl := range cc {
			PrintCluster(i, cl)
		}
		bi := cc.Largest()
		mostCentral := cc[bi].MostCentral()
		fmt.Println("Most Central", mostCentral)
		fmt.Println("Most Central Denormalized", noo.Denormalize(mostCentral))

		neighbor, _ := cc.Neighbor(mostCentral, cc.Smallest())
		PrintCluster(neighbor, cc[neighbor])
		cc.Reset()
	}
	for i, v := range vv {
		fmt.Println(i+2, ",", v)
	}
}

func PrintCluster[T Number](i int, cl Cluster[T]) {
	fmt.Printf("\nCluster %d Size: %d Variance: %2.5f \n", i, len(cl.Observations.ClusterObservations), cl.SumOfDistance())
	for _, o := range cl.Observations.ClusterObservations {
		fmt.Println((o))
	}

}

func TestBestClustersScaledNormalized(t *testing.T) {
	oo := listOfPeople()
	noo := NewNormalizeObservationAdapter(oo, []float32{0, .5, 1, .5})
	var vv []float64
	for k := 2; k <= 7; k++ {
		cc, err := OptimizeClusters(k, noo)
		vv = append(vv, cc.SumClusterVariance())
		assert.NoError(t, err)
		assert.NotEmpty(t, cc)
		for i, cl := range cc {
			PrintCluster(i, cl)
		}
		bi := cc.Largest()
		mostCentral := cc[bi].MostCentral()
		fmt.Println("Most Central", mostCentral)
		fmt.Println("Most Central Denormalized", noo.Denormalize(mostCentral))

		neighbor, _ := cc.Neighbor(mostCentral, cc.Smallest())
		PrintCluster(neighbor, cc[neighbor])
	}
	for i, v := range vv {
		fmt.Println(i+2, ",", v)
	}
}
