package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"iter"
	"os"
	"time"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/components"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/pconstantinou/kmeans"
)

var kClusters int
var kMaxClusters int

func init() {
	flag.IntVar(&kClusters, "k", 2, "starting number of clusters")
	flag.IntVar(&kMaxClusters, "k2", 3, "final number of clusters")

	flag.Parse()
}

func main() {
	page := components.NewPage()

	f, _ := os.Create("Cluster.csv")
	buf := bufio.NewWriter(f)
	defer buf.Flush()
	defer f.Close()
	pp := listOfPeople()
	noo := kmeans.NormalizeObservations(pp)
	// noo := kmeans.Observations[float64](pp)
	vv := make(map[int]float64)
	for k := kClusters; k <= kMaxClusters; k++ {
		fmt.Println(">>>>>>> K = ", k)
		cc, _ := kmeans.OptimizeClusters(k, noo)
		vv[k] = cc.SumClusterVariance()
		for i, cl := range cc {
			fmt.Printf("\nCluster %d Size: %d Variance: %2.5f \n", i, len(cl.Observations.ClusterObservations), cl.SumOfDistance())
			for _, o := range cl.Observations.ClusterObservations {
				source := observationToPerson(o)
				fmt.Println((source))
				fmt.Fprintf(buf, "\n%d,%d", k, i)
				for i := range noo.Degree() {
					fmt.Fprintf(buf, ", %f", (source).Values(i))
				}
			}
		}
		page.AddCharts(
			scatter3DBase(k, cc),
		)
	}

	f, err := os.Create("./scatter3d.html")
	if err != nil {
		panic(err)
	}
	page.Render(io.MultiWriter(f))

}

func observationToPerson(o kmeans.Observation[float64]) person {
	source := o
	if s, ok := o.(kmeans.NormalizedObservation[float64, float64]); ok {
		source = *s.Original
	}
	return source.(person)
}

func observationToSlice(o kmeans.Observation[float64], d int) []interface{} {
	var vv []interface{}
	for i := range d {
		v := o.Values(i)
		vv = append(vv, v)
	}
	return vv
}

func genScatter3dData(c kmeans.Cluster[float64]) []opts.Chart3DData {
	data := make([]opts.Chart3DData, 0)
	deg := c.Observations.Degree()
	for o := range c.Observations.All() {
		person := observationToPerson(o)
		d := opts.Chart3DData{
			Name:  person.name,
			Value: observationToSlice(person, deg),
		}
		data = append(data, d)
	}
	return data
}

func scatter3DBase(k int, cc kmeans.Clusters[float64]) *charts.Scatter3D {
	scatter3d := charts.NewScatter3D()
	scatter3d.SetGlobalOptions(
		charts.WithTitleOpts(opts.Title{Title: fmt.Sprintf("Person k=%d (gender not rendered)", k)}),
		charts.WithXAxis3DOpts(opts.XAxis3D{Name: "Weight (lbs)", Show: opts.Bool(true), Min: 75}),
		charts.WithZAxis3DOpts(opts.ZAxis3D{Name: "Height (in.)", Min: 55}),
		charts.WithYAxis3DOpts(opts.YAxis3D{Name: "Age (Years)", Min: 10}),
		charts.WithLegendOpts(opts.Legend{Orient: "vertical", Left: "left", Bottom: "center"}),
		charts.WithInitializationOpts(opts.Initialization{Width: "1200px", Height: "900px"}),
	)
	for i, c := range cc {
		scatter3d.AddSeries(fmt.Sprintf("Cluster %d (n=%d, v=%2.2f)", i,
			len(c.Observations.ClusterObservations), c.SumOfDistance()), genScatter3dData(c))
	}
	return scatter3d
}

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

func (p person) String() string {
	return fmt.Sprintf("%10s DOB: %s Height %3d in. Weight %3d lbs",
		p.name, p.birthDate.Format(time.DateOnly), p.height_inches, p.weight_lbs)
}

func listOfPeople() peopleObservations {
	return peopleObservations([]person{

		newPerson("phil", "1972-01-17", 73, 185, 1),
		newPerson("dave", "1971-10-17", 73, 165, 1),
		newPerson("lora2", "1966-12-17", 70, 121, 0),
		newPerson("dean", "1968-12-17", 71, 150, 1),
		newPerson("cate", "2010-05-10", 60, 95, 0),
		newPerson("tessa", "2010-09-10", 61, 105, 0),
		newPerson("tony", "1965-04-01", 73, 255, 1),
		newPerson("chris", "1939-07-22", 69, 215, 1),
		newPerson("janet", "1942-11-12", 60, 120, 0),
		newPerson("sophie", "1969-08-22", 62, 130, 0),
		newPerson("jenn", "1970-04-16", 62, 130, 0),
		newPerson("marlene", "1971-09-16", 62, 120, 0),
		newPerson("briteny", "1970-04-16", 62, 130, 0),
		newPerson("irini", "1970-04-16", 62, 130, 0),
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

func (p peopleObservations) Observations() iter.Seq[kmeans.Observation[float64]] {
	return func(yield func(kmeans.Observation[float64]) bool) {
		for _, o := range p {
			var co kmeans.Observation[float64]
			co = o
			if !yield(co) {
				return
			}
		}
	}
}

func (p peopleObservations) Degree() int {
	return 4
}
