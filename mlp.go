package main

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"encoding/csv"
	"os"
	"gonum.org/v1/gonum/stat/distuv"
	"io"
)

// Network is a neural network with 3 layers
type Network struct {
	inputs       	int
	hiddens      	int
	outputs      	int
	learningRate 	fixed
	hiddenWeights 	*Matrix
	outputWeights 	*Matrix
	hidden_max		fixed
	hidden_min		fixed
	out_max			fixed
	out_min			fixed
	score			int
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(input, hidden, output int, rate fixed) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}
	net.hiddenWeights = NewMatrix(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.hidden_min = Min(net.hiddenWeights)
	net.hidden_max = Max(net.hiddenWeights)
	net.outputWeights = NewMatrix(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
	net.out_min = Min(net.outputWeights)
	net.out_max = Max(net.outputWeights)

	return
}

// Train the neural network
func (net *Network) Train(inputData []fixed, targetData []fixed) {
	// feedforward
	inputs := NewMatrix(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	// find errors
	targets := NewMatrix(len(targetData), 1, targetData)
	outputErrors := subtract(targets, finalOutputs)
	hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	copy := Copy(outputErrors)
	copy.MulElem(copy, outputErrors)
	//net.errors = math.Sqrt(mat.Sum(copy))

	// backpropagate
	net.outputWeights = add(net.outputWeights,
		scale(net.learningRate,
			dot(multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T())))

	net.hiddenWeights = add(net.hiddenWeights,
		scale(net.learningRate,
			dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T())))
}

// Predict uses the neural network to predict the value given input data
func (net Network) Predict(inputData []fixed) Matrix {
	// feedforward
	inputs := NewMatrix(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)
	return *finalOutputs
}

func sigmoid(r, c int, z fixed) fixed {
	if(z < -fixed(0xA000000000000)){
		return fixed(0)
	}
	return DivideFixed(ONE, ONE + exp(-z))
}

func sigmoidPrime(m *Matrix) *Matrix {
	rows, _ := m.Dims()
	o := make([]fixed, rows)
	for i := range o {
		o[i] = ONE
	}
	ones := NewMatrix(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

func relu(r, c int, z fixed) fixed {
	if z>0 {
		return z
	}
	return 0
}

func relup(r, c int, z fixed) fixed {
	if z>0 {
		return 1
	}
	return 0
}

func reluPrime(m *Matrix) *Matrix {
	return apply(relup, m)
}

func dot(m, n *Matrix) *Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := NewMatrix(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v fixed) fixed, m *Matrix) *Matrix {
	r, c := m.Dims()
	o := NewMatrix(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s fixed, m *Matrix) *Matrix {
	o := Copy(m)
	o.Scale(s)
	return o
}

func multiply(m, n *Matrix) *Matrix {
	r, c := m.Dims()
	o := NewMatrix(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n *Matrix) *Matrix {
	r, c := m.Dims()
	o := NewMatrix(r, c, nil)
	o.Add(m, n)
	return o
}

func addScalar(i fixed, m *Matrix) *Matrix {
	r, c := m.Dims()
	a := make([]fixed, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := NewMatrix(r, c, a)
	return add(m, n)
}

func subtract(m, n *Matrix) *Matrix {
	r, c := m.Dims()
	o := NewMatrix(r, c, nil)
	o.Sub(m, n)
	return o
}

// randomly generate a float64 array
func randomArray(size int, v float64) (data []fixed) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]fixed, size)
	for i := 0; i < size; i++ {
		// data[i] = rand.NormFloat64() * math.Pow(v, -0.5)
		data[i] = floatToFixed(dist.Rand())
	}
	return
}

func addBiasNodeTo(m *Matrix, b fixed) *Matrix {
	r, _ := m.Dims()
	a := NewMatrix(r+1, 1, nil)

	a.Set(0, 0, b)
	for i := 0; i < r; i++ {
		a.Set(i+1, 0, m.At(i, 0))
	}
	return a
}

// pretty print a Gonum matrix
/*func matrixPrint(X Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}*/

func save(net Network, dataset string) {
	var oweights string
	var hweights string
	switch dataset {
			case "numbers":
				hweights = "data/numbers_hweights.model"
				oweights = "data/numbers_oweights.model"
			case "fashion":
				hweights = "data/fashion_hweights.model"
				oweights = "data/fashion_oweights.model"
			default:
				hweights = "data/numbers_hweights.model"
				oweights = "data/numbers_oweights.model"
	}
	h, err := os.Create(hweights)
	o, err2 := os.Create(oweights)
	defer h.Close()
	defer o.Close()
	if err == nil {
		r, err := net.hiddenWeights.MarshalBinaryTo()
		if err == nil {
    		_, err = io.Copy(h, r)
  		}
	}
	if err2 == nil {
		r, err := net.outputWeights.MarshalBinaryTo()
		if err == nil {
    		_, err = io.Copy(o, r)
  		}
	}
}

func save_plot(net Network, dataset string, value [][]string) {
	var file *os.File
	switch dataset {
			case "numbers":
				file, _ = os.Create("data/numbers_plot.csv")
			case "fashion":
				file, _ = os.Create("data/fashion_plot.csv")
			default:
				file, _ = os.Create("data/numbers_plot.csv")
	}
    defer file.Close()
    w := csv.NewWriter(file)
    defer w.Flush()
    w.WriteAll(value)
}

// load a neural network from file
func load(net *Network, dataset string) {
	var oweights string
	var hweights string
	switch dataset {
			case "numbers":
				hweights = "data/numbers_hweights.model"
				oweights = "data/numbers_oweights.model"
			case "fashion":
				hweights = "data/fashion_hweights.model"
				oweights = "data/fashion_oweights.model"
			default:
				hweights = "data/numbers_hweights.model"
				oweights = "data/numbers_oweights.model"
	}
	h, err := os.Create(hweights)
	o, err2 := os.Create(oweights)
	defer h.Close()
	defer o.Close()
	if err == nil {
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	if err2 == nil {
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	return
}

// predict a number from an image
// image should be 28 x 28 PNG file
func predictFromImage(net Network, path string) int {
	input := dataFromImage(path)
	output := net.Predict(input)
	//matrixPrint(output)
	best := 0
	highest := fixed(0)
	for i := 0; i < net.outputs; i++ {
		if output.At(i, 0) > highest {
			best = i
			highest = output.At(i, 0)
		}
	}
	return best
}

// get the pixel data from an image
func dataFromImage(filePath string) (pixels []fixed) {
	// read the file
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, err := png.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}

	// create a grayscale image
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}
	// make a pixel array
	pixels = make([]fixed, len(gray.Pix))
	// populate the pixel array subtract Pix from 255 because that's how
	// the MNIST database was trained (in reverse)
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = floatToFixed((float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001)
	}
	return
}

func max_weights(m, n Matrix) *Matrix {
	r, c := m.Dims()
	o := NewMatrix(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			o.Set(i, j, fixedMax(m.At(i, j), n.At(i, j)))
		}
	} 
	return o
}

func min_weights(m, n Matrix) *Matrix {
	r, c := m.Dims()
	o := NewMatrix(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			o.Set(i, j, fixedMin(m.At(i, j), n.At(i, j)))
		}
	} 
	return o
}
