package main

import (
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

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
