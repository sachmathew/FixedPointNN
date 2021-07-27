package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
	"github.com/vardius/progress-go"
	"log"
	"gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
)

func main() {
	// 784 inputs - 28 x 28 pixels, each pixel is an input
	// 100 hidden nodes - an arbitrary number
	// 10 outputs - digits 0 to 9
	// 0.1 is the learning rate
	net := CreateNetwork(784, 200, 10, fixed(0x199999999999))

	numbers := flag.String("numbers", "", "Either train or predict to evaluate neural network using mnist numbers dataset")
	fashion := flag.String("fashion", "", "Either train or predict to evaluate neural network using mnist fashion dataset")
	file := flag.String("file", "", "File name of 28 x 28 PNG file to evaluate")
	flag.Parse()

	// train or mass predict to determine the effectiveness of the trained network
	switch *numbers {
	case "train":
		mnistTrain(&net, "numbers")
	case "plot":
		mnistTrainForPlot(&net, "numbers")
	case "predict":
		load(&net, "numbers")
		mnistPredict(&net, "numbers")
	case "val":
		generateValidation("numbers")
	case "activation":
		showActivation()
	default:
		// don't do anything
	}

	switch *fashion {
	case "train":
		mnistTrain(&net, "fashion")
	case "plot":
		mnistTrainForPlot(&net, "fashion")
	case "predict":
		load(&net, "fashion")
		mnistPredict(&net, "fashion")
	case "val":
		generateValidation("fashion")
	default:
		// don't do anything
	}

	// predict individual digit images
	if *file != "" {
		// print the image out nicely on the terminal
		printImage(getImage(*file))
		// load the neural network from file
		load(&net, "numbers")
		// predict which number it is
		fmt.Println("prediction:", predictFromImage(net, *file))
	}
}

func showActivation() {
	t1 := time.Now()
    p := plot.New()

    sig := plotter.NewFunction(func(x float64) float64 { return toFloat(sigmoid(0, 0, floatToFixed(x)))})
    sig.Samples = 2048

    p.Title.Text = "sigmoid plot"
    p.Add(sig)

    p.X.Min = -64
	p.X.Max = 64
	p.Y.Min = 0
	p.Y.Max = 1
 
    if err := p.Save(8*vg.Inch, 4*vg.Inch, "plot.png"); err != nil {
        panic(err)
    }
    elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to generate activation image: %s\n", elapsed)
}

func generateValidation(dataset string) {
	t1 := time.Now()
	var testFile *os.File
	var file *os.File
	switch dataset {
			case "numbers":
				testFile, _ = os.Open("mnist_dataset/mnist_test.csv")
				file, _ = os.Create("mnist_dataset/mnist_validation.csv")
			case "fashion":
				testFile, _ = os.Open("mnist_dataset/fashion_mnist_test.csv")
				file, _ = os.Create("mnist_dataset/fashion_mnist_validation.csv")
			default:
				testFile, _ = os.Open("mnist_dataset/mnist_test.csv")
				file, _ = os.Create("mnist_dataset/mnist_validation.csv")
	}
    defer file.Close()
    r := csv.NewReader(bufio.NewReader(testFile))
    w := csv.NewWriter(file)
    defer w.Flush()
    bar := progress.New(0, 1000)
	_, _ = bar.Start()
	defer func() {
		if _, err := bar.Stop(); err != nil {
			log.Printf("failed to finish progress: %v", err)
		}
	}()
	for i := 0; i < 1000; i++ {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		err = w.Write(record)
		_, _ = bar.Advance(1)
	}
	bar.Stop()
	testFile.Close()
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to generate validation set: %s\n", elapsed)
}

func mnistTrain(net *Network, dataset string) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()
	var testFile *os.File
	bar := progress.New(0, 5)
	_, _ = bar.Start()
	defer func() {
		if _, err := bar.Stop(); err != nil {
			log.Printf("failed to finish progress: %v", err)
		}
	}()
	for epochs := 1; epochs <= 5; epochs++ {
		switch dataset {
			case "numbers":
				testFile, _ = os.Open("mnist_dataset/mnist_train.csv")
			case "fashion":
				testFile, _ = os.Open("mnist_dataset/fashion_mnist_train.csv")
			default:
				testFile, _ = os.Open("mnist_dataset/mnist_train.csv")
		}
		r := csv.NewReader(bufio.NewReader(testFile))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]fixed, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = floatToFixed((x / 255.0 * 0.999) + 0.001)
			}

			targets := make([]fixed, 10)
			for i := range targets {
				targets[i] = fixed(0x004189374bc7)
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = fixed(0xffbe76c8b439)

			net.Train(inputs, targets)
		}
		testFile.Close()
		_, _ = bar.Advance(1)
	}
	bar.Stop()
	save(*net, dataset)
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
	mnistPredict(net, dataset)
}

func mnistTrainForPlot(net *Network, dataset string) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()
	value := [][]string{}
	var testFile *os.File
	var checkFile *os.File
	bar := progress.New(0, 5)
	_, _ = bar.Start()
	defer func() {
		if _, err := bar.Stop(); err != nil {
			log.Printf("failed to finish progress: %v", err)
		}
	}()
	for epochs := 1; epochs <= 5; epochs++ {
		switch dataset {
			case "numbers":
				testFile, _ = os.Open("mnist_dataset/mnist_train.csv")
			case "fashion":
				testFile, _ = os.Open("mnist_dataset/fashion_mnist_train.csv")
			default:
				testFile, _ = os.Open("mnist_dataset/mnist_train.csv")
		}
		tr := csv.NewReader(bufio.NewReader(testFile))
		count := 1
		for {
			record, err := tr.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]fixed, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = floatToFixed((x / 255.0 * 0.999) + 0.001)
			}

			targets := make([]fixed, 10)
			for i := range targets {
				targets[i] = fixed(0x004189374bc7)
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = fixed(0xffbe76c8b439)

			net.Train(inputs, targets)
			if(count % 1000 == 0){
				switch dataset {
					case "numbers":
						checkFile, _ = os.Open("mnist_dataset/mnist_validation.csv")
					case "fashion":
						checkFile, _ = os.Open("mnist_dataset/fashion_mnist_validation.csv")
					default:
						checkFile, _ = os.Open("mnist_dataset/mnist_validation.csv")
				}
				defer checkFile.Close()
				cr := csv.NewReader(bufio.NewReader(checkFile))
				score := 0
				for {
					record, err := cr.Read()
					if err == io.EOF {
						break
					}
					inputs := make([]fixed, net.inputs)
					for i := range inputs {
						if i == 0 {
							inputs[i] = 1.0
						}
						x, _ := strconv.ParseFloat(record[i], 64)
						inputs[i] = floatToFixed((x / 255.0 * 0.999) + 0.001)
					}
					outputs := net.Predict(inputs)
					best := 0
					highest := fixed(0)
					for i := 0; i < net.outputs; i++ {
						if outputs.At(i, 0) > highest {
							best = i
							highest = outputs.At(i, 0)
						}
					}
					target, _ := strconv.Atoi(record[0])
					if best == target {
						score++
					}
				}
				net.score = score
				net.hidden_max = Max(net.hiddenWeights)
				net.hidden_min = Min(net.hiddenWeights)
				net.out_max = Max(net.outputWeights)
				net.out_min = Min(net.outputWeights)
				value = append(value, []string{strconv.Itoa(epochs), strconv.Itoa(count), strconv.FormatFloat(toFloat(net.hidden_max), 'f', -1, 64), strconv.FormatFloat(toFloat(net.hidden_min), 'f', -1, 64), strconv.FormatFloat(toFloat(net.hidden_max - net.hidden_min), 'f', -1, 64), strconv.FormatFloat(toFloat(net.out_max), 'f', -1, 64), strconv.FormatFloat(toFloat(net.out_min), 'f', -1, 64), strconv.FormatFloat(toFloat(net.out_max - net.out_min), 'f', -1, 64), strconv.Itoa(net.score),})
				checkFile.Close()
			}
			count++
		}
		testFile.Close()
		_, _ = bar.Advance(1)
	}
	bar.Stop()
	save(*net, dataset)
	save_plot(*net, dataset, value)
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to collect for plotting: %s\n", elapsed)
	mnistPredict(net, dataset)
}

func mnistPredict(net *Network, dataset string) {
	t1 := time.Now()
	var checkFile *os.File
	switch dataset {
			case "numbers":
				checkFile, _ = os.Open("mnist_dataset/mnist_test.csv")
			case "fashion":
				checkFile, _ = os.Open("mnist_dataset/fashion_mnist_test.csv")
			default:
				checkFile, _ = os.Open("mnist_dataset/mnist_test.csv")
	}
	defer checkFile.Close()
	score := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]fixed, net.inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = floatToFixed((x / 255.0 * 0.999) + 0.001)
		}
		outputs := net.Predict(inputs)
		best := 0
		highest := fixed(0)
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Println("score:", score)
}

// print out image on iTerm2; equivalent to imgcat on iTerm2
func printImage(img image.Image) {
	var buf bytes.Buffer
	png.Encode(&buf, img)
	imgBase64Str := base64.StdEncoding.EncodeToString(buf.Bytes())
	fmt.Printf("\x1b]1337;File=inline=1:%s\a\n", imgBase64Str)
}

// get the file as an image
func getImage(filePath string) image.Image {
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, _, err := image.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}
	return img
}
