package main

import (
	"fmt"
	"time"

	"gocv.io/x/gocv"
)

func matchFeatures(matcher Matcher, mapp Mapp, img gocv.Mat) map[gocv.KeyPoint]gocv.KeyPoint {
	rets := make(map[gocv.KeyPoint]gocv.KeyPoint)
	framesCount := len(mapp.Frames)

	matchingStart := time.Now()

	if mapp.Frames[framesCount-1].Des.Empty() || mapp.Frames[framesCount-2].Des.Empty() {
		return rets
	}

	matches := matcher.KnnMatch(mapp.Frames[framesCount-1].Des, mapp.Frames[framesCount-2].Des, 2)

	fmt.Println("Matching key points takes: ", time.Now().Sub(matchingStart).Nanoseconds())
	fmt.Println("Matchers before filtering: ", len(matches))

	start := time.Now()

	for _, n := range matches {
		if n[0].Distance < 0.75*n[1].Distance {
			p1 := mapp.Frames[framesCount-1].KPS[n[0].QueryIdx]
			p2 := mapp.Frames[framesCount-2].KPS[n[0].TrainIdx]

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
	gocv.FindHomography(src, &dest, gocv.HomograpyMethodRANSAC, 3, &m, 2000, 0.85)

	betterRets := make(map[gocv.KeyPoint]gocv.KeyPoint)
	j := 0
	for srcPoints, destPoints := range rets {
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
