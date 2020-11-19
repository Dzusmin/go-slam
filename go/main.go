package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"strconv"
	"time"

	"net/http"
	_ "net/http/pprof"

	"gocv.io/x/gocv"
)

type Vecb []uint8

func GetVecbAt(m gocv.Mat, row int, col int) Vecb {
	ch := m.Channels()
	v := make(Vecb, ch)

	for c := 0; c < ch; c++ {
		v[c] = m.GetUCharAt(row, col*ch+c)
	}

	return v
}

func (v Vecb) SetVecbAt(m gocv.Mat, row int, col int) {
	ch := m.Channels()

	for c := 0; c < ch; c++ {
		m.SetUCharAt(row, col*ch+c, v[c])
	}
}

type Matcher interface {
	KnnMatch(query, train gocv.Mat, k int) [][]gocv.DMatch
	Close() error
}

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
	if len(kps) == 0 {
		return
	}
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

func detectFeatures(matcher *gocv.ORB, mask *gocv.Mat, mapp *Mapp, img gocv.Mat) Frame {
	kps, des := matcher.DetectAndCompute(img, *mask)
	fmt.Println("KPS: ", len(kps))
	return NewFrame(kps, des)
}

func matchFeatures(matcher Matcher, mapp Mapp, img gocv.Mat) map[gocv.KeyPoint]gocv.KeyPoint {
	rets := make(map[gocv.KeyPoint]gocv.KeyPoint)
	framesCount := len(mapp.Frames)

	matchingStart := time.Now()

	if mapp.Frames[framesCount-1].Des.Empty() || mapp.Frames[framesCount-2].Des.Empty() {
		return rets
	}

	matches := matcher.KnnMatch(mapp.Frames[framesCount-2].Des, mapp.Frames[framesCount-1].Des, 2)

	fmt.Println("Matching key points takes: ", time.Now().Sub(matchingStart).Nanoseconds())
	fmt.Println("Matchers before filtering: ", len(matches))

	start := time.Now()

	for _, n := range matches {
		if n[0].Distance < 0.75*n[1].Distance {
			p1 := mapp.Frames[framesCount-2].KPS[n[0].QueryIdx]
			p2 := mapp.Frames[framesCount-1].KPS[n[0].TrainIdx]

			rets[p1] = p2
		}
	}
	if len(rets) < 4 {
		return rets
	}

	src := gocv.NewMatWithSize(len(rets), 1, gocv.MatTypeCV64FC2)
	defer src.Close()
	dest := gocv.NewMatWithSize(len(rets), 1, gocv.MatTypeCV64FC2)
	defer dest.Close()
	i := 0
	for srcPoints, destPoints := range rets {
		src.SetDoubleAt(i, 0, float64(srcPoints.X))
		src.SetDoubleAt(i, 1, float64(srcPoints.Y))
		dest.SetDoubleAt(i, 0, float64(destPoints.X))
		dest.SetDoubleAt(i, 1, float64(destPoints.Y))
		i = i + 1
	}

	m := gocv.NewMat()
	defer m.Close()
	gocv.FindHomography(src, &dest, gocv.HomograpyMethodRANSAC, 2, &m, 3000, 0.7)

	betterRets := make(map[gocv.KeyPoint]gocv.KeyPoint)
	j := 0
	for srcPoints, destPoints := range rets {
		// fmt.Println("m for", j, m.GetUCharAt(j, 0) > 0)
		if m.GetUCharAt(j, 0) > 0 {
			betterRets[srcPoints] = destPoints
		}

		j = j + 1
	}

	fmt.Println("Rets len: ", len(rets))
	fmt.Println("Better rets len: ", len(betterRets))
	fmt.Println("Filtering matches takes: ", time.Now().Sub(start).Nanoseconds())
	return betterRets
}

func findMask(new gocv.Mat, old gocv.Mat, zeros *gocv.Mat) {
	// newWorkFrame := gocv.NewMat()
	// defer newWorkFrame.Close()

	// oldWorkFrame := gocv.NewMat()
	// defer oldWorkFrame.Close()

	// gocv.CvtColor(new, &newWorkFrame, gocv.ColorBGRToHSVFull)
	// gocv.CvtColor(old, &oldWorkFrame, gocv.ColorBGRToHSVFull)

	// mask := gocv.NewMat()
	// defer mask.Close()

	// grayMask := gocv.NewMat()
	// defer grayMask.Close()

	// gocv.AbsDiff(newWorkFrame, oldWorkFrame, &mask)
	// gocv.CvtColor(mask, &grayMask, gocv.ColorBGRToGray)

	// gocv.AdaptiveThreshold(grayMask, zeros, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinary, 11, 2)

	//TODO:https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
	//TODO:https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tgo-slam [videofile] [scale:float] [profiler:bool] [bfmatcher:bool]")
		return
	}

	filename := os.Args[1]

	scale := 0.8
	if len(os.Args) >= 3 {
		scale, _ = strconv.ParseFloat(os.Args[2], 64)
	}

	profiler := false
	if len(os.Args) >= 4 {
		profiler, _ = strconv.ParseBool(os.Args[3])
	}

	bfmatcher := true
	if len(os.Args) >= 5 {
		bfmatcher, _ = strconv.ParseBool(os.Args[4])
	}

	if profiler {
		go func() {
			log.Println(http.ListenAndServe("localhost:6060", nil))
		}()
	}

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

	oldFrame := gocv.NewMat()
	defer oldFrame.Close()

	workFrame := gocv.NewMat()
	defer workFrame.Close()

	dest := gocv.NewMat()
	defer dest.Close()

	mapp := NewMap()
	blue := color.RGBA{0, 0, 255, 0}
	// violet := color.RGBA{255, 0, 255, 0}
	red := color.RGBA{255, 0, 0, 0}
	green := color.RGBA{0, 255, 0, 0}

	surf := gocv.NewORB()
	defer surf.Close()

	mask := gocv.NewMat()
	defer mask.Close()

	var matcher Matcher

	if bfmatcher {
		m := gocv.NewBFMatcherWithParams(gocv.NormHamming2, false)
		matcher = &m
	} else {
		m := gocv.NewFlannBasedMatcher()
		matcher = &m
	}

	defer matcher.Close()

	matches := make(map[gocv.KeyPoint]gocv.KeyPoint)

	for {
		framesCount := len(mapp.Frames)
		now := time.Now()
		if ok := video.Read(&img); !ok {
			fmt.Println("Error reading video from: ", filename)
			return
		}

		if img.Empty() {
			fmt.Println("Image from video: ", filename, "is empty")
			continue
		}

		gocv.Resize(img, &smaller, image.Point{}, scale, scale, gocv.InterpolationDefault)
		gocv.CvtColor(smaller, &dest, gocv.ColorBGRAToGray)

		if framesCount > 1 {
			findMask(smaller, oldFrame, &mask)
		}

		mapp.AddFrame(detectFeatures(&surf, &mask, &mapp, dest))
		drawKPS(&smaller, mapp.Frames[len(mapp.Frames)-1].KPS, blue)
		if framesCount > 1 {
			matches = matchFeatures(matcher, mapp, smaller)
		}

		oldFrame = smaller.Clone()

		diff := time.Now().Sub(now)

		fmt.Println("Matches: ", len(matches))
		drawMatches(&smaller, matches, red)
		gocv.PutText(&smaller, fmt.Sprintf("Matches: %d ", len(matches)), image.Pt(10, 40), gocv.FontHersheyPlain, 1.2, green, 2)
		drawInofs(&smaller, fmt.Sprintf("KPS: %d ", len(mapp.Frames[len(mapp.Frames)-1].KPS)), image.Pt(10, 20), green)
		drawInofs(&smaller, fmt.Sprintf("ms: %d ", diff.Milliseconds()), image.Pt(10, 60), green)

		if framesCount > 1 {
			// time.Sleep(time.Second * 10)

			window.IMShow(smaller)
		}
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
