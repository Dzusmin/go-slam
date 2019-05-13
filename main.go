package main

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"time"

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

func drawFeatures(img *gocv.Mat, features gocv.Mat, color color.RGBA) {
	for i := 0; i < features.Rows(); i++ {
		v := features.GetVecfAt(0, i)
		// if circles are found
		if len(v) > 1 {
			x := int(v[0])
			y := int(v[1])

			gocv.Circle(img, image.Pt(x, y), 5, color, 2)
		}
	}
}

func drawMatches(img *gocv.Mat, matches map[gocv.KeyPoint]gocv.KeyPoint, color color.RGBA) {
	for u, v := range matches {
		gocv.Line(img, image.Pt(int(u.X), int(u.Y)), image.Pt(int(v.X), int(v.Y)), color, 2)
	}
}

func matchFeatures(mapp Mapp, img gocv.Mat) map[gocv.KeyPoint]gocv.KeyPoint {
	bf := gocv.NewBFMatcherWithParams(gocv.NormHamming, false)
	defer bf.Close()

	framesCount := len(mapp.Frames)
	matches := bf.KnnMatch(mapp.Frames[framesCount-2].Des, mapp.Frames[framesCount-1].Des, 2)

	rets := make(map[gocv.KeyPoint]gocv.KeyPoint)
	fmt.Println(len(matches))
	for _, n := range matches {
		if n[0].Distance < 0.75*n[1].Distance {
			p1 := mapp.Frames[framesCount-2].KPS[n[0].QueryIdx]
			p2 := mapp.Frames[framesCount-2].KPS[n[0].TrainIdx]
			rets[p1] = p2
		}
	}

	return rets
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tgo-slam [videofile]")
		return
	}

	filename := os.Args[1]

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

	dest := gocv.NewMat()
	defer dest.Close()

	corners := gocv.NewMat()
	defer corners.Close()

	mapp := NewMap()
	blue := color.RGBA{0, 0, 255, 0}
	red := color.RGBA{255, 0, 0, 0}
	green := color.RGBA{0, 255, 0, 0}

	for {
		if ok := video.Read(&img); !ok {
			fmt.Println("Error reading video from: ", filename)
			return
		}
		if img.Empty() {
			fmt.Println("Image from video: ", filename, "is empty")
			continue
		}

		gocv.CvtColor(img, &dest, gocv.ColorBGRAToGray)

		gocv.GoodFeaturesToTrack(dest, &corners, 5000, 0.01, 7.0)

		if corners.Empty() {
			fmt.Println("No corners found")
		} else {
			orb := gocv.NewORB()
			kps, des := orb.DetectAndCompute(img, corners)
			mapp.AddFrame(NewFrame(kps, des))
			drawFeatures(&img, corners, blue)
		}

		framesCount := len(mapp.Frames)

		if framesCount > 1 {
			matches := matchFeatures(mapp, img)
			drawMatches(&img, matches, red)
		}

		gocv.PutText(&img, fmt.Sprintf("Count: %d ", corners.Total()), image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, green, 2)

		time.Sleep(time.Second / 20)
		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
