package main

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"strconv"

	"gocv.io/x/gocv"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tgo-slam [videofile] [line] [axis (x/y)] [with] ")
		return
	}

	filename := os.Args[1]

	line, _ := strconv.Atoi(os.Args[2])
	axis := os.Args[3]
	width, _ := strconv.Atoi(os.Args[4])

	video, err := gocv.VideoCaptureFile(filename)
	if err != nil {
		fmt.Println("Error reading video from: %v", filename)
		return
	}
	defer video.Close()

	window := gocv.NewWindow("With line")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	imgFG := gocv.NewMat()
	defer imgFG.Close()

	imgCleaned := gocv.NewMat()
	defer imgCleaned.Close()

	mog2 := gocv.NewBackgroundSubtractorMOG2()
	defer mog2.Close()

	count := 0
	for {
		if ok := video.Read(&img); !ok {
			fmt.Println("Error reading video from: %v", filename)
			return
		}
		if img.Empty() {
			continue
		}

		mog2.Apply(img, &imgFG)
		kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
		gocv.Erode(imgFG, &imgCleaned, kernel)
		kernel.Close()

		moments := gocv.Moments(imgCleaned, true)
		area := moments["m00"]

		if area >= 1 {
			x := int(moments["m10"] / area)
			y := int(moments["m01"] / area)

			if axis == "y" {
				if x > 0 && x < img.Cols() && y > line && y < line+width {
					count++
				}
				gocv.Line(&img, image.Pt(0, line), image.Pt(img.Cols(), line), color.RGBA{255, 0, 0, 0}, 2)
			}
			if axis == "x" {
				if y > 0 && y < img.Rows() && x > line && x < line+width {
					count++
				}
				gocv.Line(&img, image.Pt(line, 0), image.Pt(line, img.Rows()), color.RGBA{255, 0, 0, 0}, 2)
			}

			gocv.PutText(&img, fmt.Sprintf("Count: %d ", count), image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, color.RGBA{0, 255, 0, 0}, 2)

			window.IMShow(img)
			if window.WaitKey(1) >= 0 {
				break
			}
		}
	}

}
