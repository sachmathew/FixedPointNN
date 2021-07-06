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
	"gonum.org/v1/gonum/mat"
)

func main() {
	// 784 inputs - 28 x 28 pixels, each pixel is an input
	// 100 hidden nodes - an arbitrary number
	// 10 outputs - digits 0 to 9
	// 0.1 is the learning rate
	net := CreateNetwork(784, 200, 10, 0.1)

	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	file := flag.String("file", "", "File name of 28 x 28 PNG file to evaluate")
	flag.Parse()

	// train or mass predict to determine the effectiveness of the trained network
	switch *mnist {
	case "train":
		mnistTrain(&net)
	case "plot":
		mnistTrainForPlot(&net)
	case "predict":
		load(&net)
		mnistPredict(&net)
	case "val":
		generateValidation()
	default:
		// don't do anything
	}

	// predict individual digit images
	if *file != "" {
		// print the image out nicely on the terminal
		printImage(getImage(*file))
		// load the neural network from file
		load(&net)
		// predict which number it is
		fmt.Println("prediction:", predictFromImage(net, *file))
	}

}

func generateValidation() {
	t1 := time.Now()
	testFile, _ := os.Open("mnist_dataset/mnist_test.csv")
	file, _ := os.Create("mnist_dataset/mnist_validation.csv")
    defer file.Close()
    r := csv.NewReader(bufio.NewReader(testFile))
    w := csv.NewWriter(file)
    defer w.Flush()
	for i := 0; i < 1000; i++ {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		err = w.Write(record)
	}
	testFile.Close()
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to generate validation set: %s\n", elapsed)
}

func mnistTrain(net *Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()
	for epochs := 0; epochs < 1; epochs++ {
		testFile, _ := os.Open("mnist_dataset/mnist_train.csv")
		r := csv.NewReader(bufio.NewReader(testFile))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]float64, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.999) + 0.001
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.001
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.999

			net.Train(inputs, targets)
		}
		testFile.Close()
	}
	save(*net)
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
}

func mnistTrainForPlot(net *Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()
	value := [][]string{}
	for epochs := 1; epochs <= 5; epochs++ {
		fmt.Println("epoch:", epochs)
		testFile, _ := os.Open("mnist_dataset/mnist_train.csv")
		tr := csv.NewReader(bufio.NewReader(testFile))
		count := 1
		for {
			record, err := tr.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]float64, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.999) + 0.001
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.001
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.999

			net.Train(inputs, targets)
			if(count % 1000 == 0){
				checkFile, _ := os.Open("mnist_dataset/mnist_validation.csv")
				defer checkFile.Close()
				cr := csv.NewReader(bufio.NewReader(checkFile))
				score := 0
				for {
					record, err := cr.Read()
					if err == io.EOF {
						break
					}
					inputs := make([]float64, net.inputs)
					for i := range inputs {
						if i == 0 {
							inputs[i] = 1.0
						}
						x, _ := strconv.ParseFloat(record[i], 64)
						inputs[i] = (x / 255.0 * 0.999) + 0.001
					}
					outputs := net.Predict(inputs)
					best := 0
					highest := 0.0
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
				fmt.Println("score:", score)
				net.score = score
				net.hidden_max = mat.Max(net.hiddenWeights)
				net.hidden_min = mat.Min(net.hiddenWeights)
				net.out_max = mat.Max(net.outputWeights)
				net.out_min = mat.Min(net.outputWeights)
				value = append(value, []string{strconv.Itoa(epochs), strconv.Itoa(count), strconv.FormatFloat(net.hidden_max, 'f', -1, 64), strconv.FormatFloat(net.hidden_min, 'f', -1, 64), strconv.FormatFloat(net.hidden_max - net.hidden_min, 'f', -1, 64), strconv.FormatFloat(net.out_max, 'f', -1, 64), strconv.FormatFloat(net.out_min, 'f', -1, 64), strconv.FormatFloat(net.out_max - net.out_min, 'f', -1, 64), strconv.Itoa(net.score),})
				checkFile.Close()
			}
			count++
		}
		testFile.Close()
	}
	save(*net)
	save_plot(*net, value)
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to collect for plotting: %s\n", elapsed)
}

func mnistPredict(net *Network) {
	t1 := time.Now()
	checkFile, _ := os.Open("mnist_dataset/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.999) + 0.001
		}
		outputs := net.Predict(inputs)
		best := 0
		highest := 0.0
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
