#!/usr/bin/env python3
"""
outer_corners_maxarea.py
Choose four points FROM the detected set that form the convex quadrilateral with MAXIMUM AREA.

Pipeline
--------
1) Import Shi–Tomasi detector from your working file.
2) Detect corners (or load Nx2 points from --points-npy).
3) Compute convex hull of those points (hull vertices are a subset of your points).
4) Search the hull for the maximum-area quadrilateral (O(h^3), h = #hull vertices).
5) Order the result as TL,TR,BR,BL (by sums/diffs) and save overlay / .npy / JSON.

Usage
-----
python outer_corners_maxarea.py the_maze.jpg \
  --max-corners 1500 --quality 0.01 --min-dist 6 --block-size 3 \
  --overlay ref/maxarea_overlay.png --corners-npy ref/ref_corners.npy
# Or if you already saved points as .npy (Nx2 float):
python outer_corners_maxarea.py the_maze.jpg --points-npy corners.npy
"""

import argparse, json
import numpy as np
import cv2
from pathlib import Path


def draw_points(img, pts, color=(0,0,255), radius=3):
    for (x, y) in pts.astype(int):
        cv2.circle(img, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

def shi_tomasi(gray, max_corners=1000, quality=0.2, min_dist=500, block_size=9, subpix=False):
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

# ---------------------- small helpers ----------------------

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _polygon_area2(poly: np.ndarray) -> float:
    """Twice the signed area; >0 for CCW in standard coords (y up).
    With image coords (y down), the sign may flip; we only use sign to ensure CCW."""
    x = poly[:,0]; y = poly[:,1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _tri_area2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Twice the area of triangle ABC (absolute)."""
    return abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

def _order_tl_tr_br_bl(pts4: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL using sums/diffs."""
    pts = pts4.astype(np.float32)
    s = pts.sum(axis=1)                # x+y
    d = np.diff(pts, axis=1).ravel()   # x - y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.vstack([tl, tr, br, bl])

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull as Nx2 float32 in CCW order using OpenCV (hull vertices are from input)."""
    if points.shape[0] < 3:
        return points.copy()
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1,1,2), returnPoints=True)
    H = hull.reshape(-1,2).astype(np.float32)
    # Ensure CCW (for consistent indexing); image coords may invert sign
    if _polygon_area2(H) < 0:
        H = H[::-1].copy()
    return H

def _max_area_quad_from_hull(H: np.ndarray):
    """
    Return indices (i,k,j,l) into H (CCW) that form the maximum-area convex quadrilateral.
    O(h^3) algorithm:
      For every diagonal (i,j) with 2 ≤ d ≤ h-2 (d = CCW distance),
      pick k on arc (i..j) maximizing area(i,k,j), and l on arc (j..i) maximizing area(j,l,i).
      Maximize sum of the two triangle areas.
    """
    h = H.shape[0]
    if h < 4:
        return None
    # Duplicate for easy wrap-around slicing
    HH = np.vstack([H, H])
    best = (None, -1.0)  # (indices tuple, area2)
    for i in range(h):
        for j in range(i+2, i+h-1):  # ensure each side has at least one vertex; exclude neighbors and wrap-neighbor
            d = j - i  # CCW distance on HH
            arc1 = HH[i+1:j]           # candidates k between i and j
            arc2 = HH[j+1:i+h]         # candidates l between j and i+h

            if arc1.shape[0] == 0 or arc2.shape[0] == 0:
                continue

            Ai = HH[i]; Aj = HH[j]

            # Max area(i,k,j) over arc1
            vBA = Aj - Ai
            vKA = arc1 - Ai
            areas1 = np.abs(vBA[0]*vKA[:,1] - vBA[1]*vKA[:,0])  # vectorized 2*area
            k_rel = int(np.argmax(areas1))
            k_idx = i + 1 + k_rel

            # Max area(j,l,i) over arc2
            vAB = Ai - Aj
            vLB = arc2 - Aj
            areas2 = np.abs(vAB[0]*vLB[:,1] - vAB[1]*vLB[:,0])
            l_rel = int(np.argmax(areas2))
            l_idx = j + 1 + l_rel

            total = float(areas1[k_rel] + areas2[l_rel])  # 2*area (no /2 needed for comparison)
            if total > best[1]:
                best = ((i % h, k_idx % h, j % h, l_idx % h), total)

    return best[0]  # (i,k,j,l)

# --------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="data/images/frame_100.png")
    ap.add_argument("--overlay", type=str, default="data/images/maze_overlay.png", help="Save visualization PNG.")

    ap.add_argument("--corners-json", type=str, default="data/calibration/maze_corners.json", help="Save TL,TR,BR,BL to .json")

    # Detection params (used only if --points-npy is not given)
    ap.add_argument("--max-corners", type=int, default=1500)
    ap.add_argument("--quality", type=float, default=0.001)
    ap.add_argument("--min-dist", type=float, default=50)
    ap.add_argument("--block-size", type=int, default=9)
    ap.add_argument("--subpix", action="store_true")

    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)

   
    P = shi_tomasi(gray_blur,
                   max_corners=args.max_corners,
                   quality=args.quality,
                   min_dist=args.min_dist,
                   block_size=args.block_size,
                   subpix=args.subpix)

    if P.shape[0] < 4:
        print(json.dumps({"status":"error","reason":"<4 detected points"}, indent=2))
        return

    # (Optional) deduplicate very-close points to stabilize hull
    if P.shape[0] > 0:
        P = np.unique(np.round(P, 1), axis=0).astype(np.float32)

    # 2) convex hull of detected points
    H = _convex_hull(P)
    h = H.shape[0]

    if h < 4:
        print(json.dumps({"status":"error","reason":"Convex hull has <4 vertices; cannot form quadrilateral"}, indent=2))
        return

    # 3) max-area quadrilateral on the hull
    idxs = _max_area_quad_from_hull(H)
    if idxs is None:
        print(json.dumps({"status":"error","reason":"Failed to find a valid quadrilateral"}, indent=2))
        return
    i,k,j,l = idxs
    quad = np.vstack([H[i], H[k], H[j], H[l]]).astype(np.float32)

    # 4) order as TL,TR,BR,BL (for homography convenience)
    ordered = _order_tl_tr_br_bl(quad)

    # 5) overlay
    overlay_path = args.overlay or (Path(args.image).with_suffix("").as_posix() + "_maxarea_outer.png")
    vis = img.copy()
    # draw detected points (light)
    for (x,y) in P.astype(int):
        cv2.circle(vis, (x,y), 1, (0,255,0), -1, lineType=cv2.LINE_AA)
    # draw hull (optional)
    for a,b in zip(H.astype(int), np.roll(H.astype(int), -1, axis=0)):
        cv2.line(vis, tuple(a), tuple(b), (150,150,150), 1, cv2.LINE_AA)
    # draw max-area quadrilateral
    Q = quad.astype(int)
    for a,b in zip(Q, np.roll(Q, -1, axis=0)):
        cv2.line(vis, tuple(a), tuple(b), (0,255,255), 2, cv2.LINE_AA)
    # labels for ordered corners
    for lab,(x,y) in zip(["TL","TR","BR","BL"], ordered.astype(int)):
        cv2.circle(vis, (x,y), 7, (0,0,255), -1)
        cv2.putText(vis, lab, (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imwrite(overlay_path, vis)
    

    # 7) report
    print(json.dumps({
        "status": "ok",
        "num_detected": int(P.shape[0]),
        "num_hull_vertices": int(h),
        "outer_corners_TL_TR_BR_BL": ordered.round(2).tolist(),
        "overlay": overlay_path
    }, indent=2))

    image_corners = ordered.round(2).tolist()
    the_corners = {
        "TL": image_corners[0], 
        "TR": image_corners[1], 
        "BR": image_corners[2], 
        "BL": image_corners[3]
    }

    save_json(the_corners, args.corners_json)      

if __name__ == "__main__":
    main()
