package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"strconv"

	"net/http"
	_ "net/http/pprof"

	"gocv.io/x/gocv"
)

type Frame struct {
	KPS []gocv.KeyPoint
	Des gocv.Mat
}

func NewFrame(kps []gocv.KeyPoint, des gocv.Mat) Frame {
	return Frame{KPS: kps, Des: des}
}

type Mapp struct {
	Frames []Frame
}

func NewMap() Mapp {
	return Mapp{}
}

func (m *Mapp) AddFrame(frame Frame) []Frame {
	m.Frames = append(m.Frames, frame)
	return m.Frames
}

func drawKPS(img *gocv.Mat, kps []gocv.KeyPoint, color color.RGBA) {
	gocv.DrawKeyPoints(*img, kps, img, color, gocv.DrawRichKeyPoints)
}

func drawMatches(img *gocv.Mat, matches map[gocv.KeyPoint]gocv.KeyPoint, color color.RGBA) {
	for u, v := range matches {
		gocv.Line(img, image.Pt(int(u.X), int(u.Y)), image.Pt(int(v.X), int(v.Y)), color, 2)
	}
}

func drawInofs(img *gocv.Mat, text string, org image.Point, color color.RGBA) {
	gocv.PutText(img, text, org, gocv.FontHersheyPlain, 1.2, color, 2)
}

func detectFeatures(surf *gocv.SIFT, mapp *Mapp, img gocv.Mat) Frame {
	mask := gocv.NewMat()
	kps, des := surf.DetectAndCompute(img, mask)
	fmt.Println("KPS: ", len(kps))
	return NewFrame(kps, des)
}

func matchFeatures(mapp Mapp, img gocv.Mat) map[gocv.KeyPoint]gocv.KeyPoint {
	bf := gocv.NewBFMatcherWithParams(gocv.NormL1, false)
	defer bf.Close()

	framesCount := len(mapp.Frames)
	matches := bf.KnnMatch(mapp.Frames[framesCount-1].Des, mapp.Frames[framesCount-2].Des, 2)

	rets := make(map[gocv.KeyPoint]gocv.KeyPoint)
	for _, m := range matches {
		if m[0].Distance < 0.4*m[1].Distance {
			p1 := mapp.Frames[framesCount-1].KPS[m[0].QueryIdx]
			p2 := mapp.Frames[framesCount-2].KPS[m[0].TrainIdx]

			rets[p1] = p2

			// fmt.Println("Type of p2: ", reflect.TypeOf(p1))
		}
	}

	//TODO: filter kps with ransac

	return rets
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tgo-slam [videofile] [maxFeatures] [featuresQuality] [minDist]")
		return
	}

	filename := os.Args[1]

	scale := 0.25
	if len(os.Args) >= 3 {
		scale, _ = strconv.ParseFloat(os.Args[2], 64)
	}

	// app, _ := application.Create(application.Options{Title: "SLAM", Width: 800, Height: 600})

	// currentTime := time.Now()

	err := http.ListenAndServe("localhost:6060", nil)
	if err != nil {
		log.Fatal(err)
	}

	// f, err := os.Create("log/" + currentTime.Format("2006-01-02 15:04:05.000000000") + ".log")
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// defer f.Close()

	// pprof.StartCPUProfile(f)
	// defer pprof.StopCPUProfile()

	video, err := gocv.VideoCaptureFile(filename)
	if err != nil {
		fmt.Println("Error reading video from: ", filename)
		return
	}
	defer video.Close()

	window := gocv.NewWindow("With line")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	smaller := gocv.NewMat()
	defer smaller.Close()

	dest := gocv.NewMat()
	defer dest.Close()

	mapp := NewMap()
	blue := color.RGBA{0, 0, 255, 0}
	// violet := color.RGBA{255, 0, 255, 0}
	red := color.RGBA{255, 0, 0, 0}
	green := color.RGBA{0, 255, 0, 0}

	surf := gocv.NewSIFT()

	for {
		if ok := video.Read(&img); !ok {
			fmt.Println("Error reading video from: ", filename)
			return
		}

		if img.Empty() {
			fmt.Println("Image from video: ", filename, "is empty")
			continue
		}

		gocv.Resize(img, &smaller, image.Point{}, scale, scale, gocv.InterpolationDefault)

		// gocv.FindHomography()
		gocv.CvtColor(smaller, &dest, gocv.ColorBGRAToGray)

		mapp.AddFrame(detectFeatures(&surf, &mapp, dest))
		drawKPS(&smaller, mapp.Frames[len(mapp.Frames)-1].KPS, blue)

		framesCount := len(mapp.Frames)
		if framesCount > 1 {
			matches := matchFeatures(mapp, smaller)

			fmt.Println("Matches: ", len(matches))
			gocv.PutText(&smaller, fmt.Sprintf("Matches: %d ", len(matches)), image.Pt(10, 40), gocv.FontHersheyPlain, 1.2, green, 2)

			drawMatches(&smaller, matches, red)
		}

		drawInofs(&smaller, fmt.Sprintf("KPS: %d ", len(mapp.Frames[len(mapp.Frames)-1].KPS)), image.Pt(10, 20), green)

		// time.Sleep(time.Second / 200)
		window.IMShow(smaller)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
