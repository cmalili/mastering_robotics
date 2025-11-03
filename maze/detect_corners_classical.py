#!/usr/bin/env python3
"""
detect_corners_classical.py
Minimal corner detection with classic methods (Shi–Tomasi).

Examples
--------
# Shi–Tomasi (recommended default)
python detect_corners_classical.py the_maze.jpg --method shi --max-corners 1500 --quality 0.01 --min-dist 6 --subpix
"""


import argparse, json
import numpy as np
import cv2
from pathlib import Path

def draw_points(img, pts, color=(0,0,255), radius=3):
    for (x, y) in pts.astype(int):
        cv2.circle(img, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

def shi_tomasi(gray, max_corners=1000, quality=0.01, min_dist=5, block_size=9, subpix=False):
    """Shi–Tomasi Good Features to Track."""
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist,
        blockSize=block_size, useHarrisDetector=False, k=0.04
    )
    if corners is None:
        return np.empty((0,2), np.float32)
    pts = corners.reshape(-1, 2).astype(np.float32)

    if subpix and len(pts) > 0:
        # Subpixel refinement (optional)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
        pts_ref = cv2.cornerSubPix(gray, pts.reshape(-1,1,2), (7,7), (-1,-1), term)
        pts = pts_ref.reshape(-1,2).astype(np.float32)
    return pts

"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="data/images/maze.jpg", help="Path to input image.")
    ap.add_argument("--out", type=str, default="data/images/maze_detected_corners.jpg", help="Overlay output path (default: input_stem+'_corners.png').")

    # Shi–Tomasi params
    ap.add_argument("--max-corners", type=int, default=15)
    ap.add_argument("--quality", type=float, default=0.2)
    ap.add_argument("--min-dist", type=float, default=500)
    ap.add_argument("--block-size", type=int, default=9)
    ap.add_argument("--subpix", action="store_true", help="Enable subpixel refinement for shi/harris.")

    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light denoise helps on paper scans
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)

    pts = shi_tomasi(gray_blur,
                     max_corners=args.max_corners,
                     quality=args.quality,
                     min_dist=args.min_dist,
                     block_size=args.block_size,
                     subpix=args.subpix)

    overlay = img.copy()
    draw_points(overlay, pts, color=(0,0,255), radius=10)

    out_path = args.out
    cv2.imwrite(out_path, overlay)

    # Print a compact JSON summary (first 50 corners)
    summary = {
        "num_corners": int(len(pts)),
        "first_50": pts[:50].round(2).tolist(),
        "overlay": out_path,
        "image_size": {"w": img.shape[1], "h": img.shape[0]},
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
"""