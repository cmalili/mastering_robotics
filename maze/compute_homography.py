#!/usr/bin/env python3
"""
compute_homography_from_corners.py
Compute homography H : pixels -> target (canonical rectangle OR robot frame) from 4 corners.

Inputs
------
- Image (for visualization only)
- 4 corners TL,TR,BR,BL in pixels (from your detector), saved as a .npy with shape (4,2)

Modes
-----
1) --mode rect  : map to a fronto-parallel rectangle of size (W,H).
   - If you don't pass --rect-size, it auto-estimates W,H from edge lengths.

2) --mode robot : map to your robot base XY coordinates (mm or meters).
   - Pass 8 floats (TLx TLy TRx TRy BRx BRy BLx BLy) or a .npy with shape (4,2).

Outputs
-------
- H saved to .npy (pixels -> target)
- Optional rectified image (rect mode)
- Verification overlay: projects target grid back into the input image

Examples
--------
# A) Canonical rectified plane (auto size)
python compute_homography_from_corners.py the_maze.jpg \
  --corners-npy ref/ref_corners.npy \
  --mode rect --out-npy ref/H_rect.npy \
  --rectified ref/rectified.png \
  --overlay ref/hverify_rect.png

# B) Canonical rectified plane (explicit size W,H)
python compute_homography_from_corners.py the_maze.jpg \
  --corners-npy ref/ref_corners.npy \
  --mode rect --rect-size 1000 800 \
  --out-npy ref/H_rect.npy --rectified ref/rectified.png --overlay ref/hverify_rect.png

# C) Robot frame (mm) with 8 numbers TL TR BR BL in order
python compute_homography_from_corners.py the_maze.jpg \
  --corners-npy ref/ref_corners.npy \
  --mode robot --robot-corners 0 0  300 0  300 200  0 200 \
  --out-npy ref/H_ref.npy --overlay ref/hverify_robot.png --grid-step 20

# D) Robot frame (mm) from a .npy (4x2) TL,TR,BR,BL
python compute_homography_from_corners.py the_maze.jpg \
  --corners-npy ref/ref_corners.npy \
  --mode robot --robot-corners-npy ref/robot_corners.npy \
  --out-npy ref/H_ref.npy --overlay ref/hverify_robot.png --grid-step 25
"""

from __future__ import annotations
import argparse, json, os
import numpy as np
import cv2
from pathlib import Path

def read_corners_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    C = np.array([
        data["TL"], data["TR"], data["BR"], data["BL"]
    ], dtype=np.float32)

    if C.shape != (4,2):
        raise SystemExit(f"{path} must be shape (4,2) TL,TR,BR,BL")
    return C

def _ensure_dir(p: str):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def _order_tl_tr_br_bl(pts4: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL using sums/diffs (robust if order unknown)."""
    pts = pts4.astype(np.float32)
    s = pts.sum(axis=1)              # x+y
    d = np.diff(pts, axis=1).ravel() # x - y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.vstack([tl, tr, br, bl])

def _draw_backprojection_overlay(img_bgr: np.ndarray, H: np.ndarray,
                                 tgt_corners: np.ndarray, grid_step: float,
                                 out_png: str):
    """
    Draw projected target corners + a coarse grid back onto the image using H^{-1}.
    tgt_corners: (4,2) TL,TR,BR,BL in target coords.
    grid_step: step in target units (pixels for rect mode; mm for robot mode).
    """
    dbg = img_bgr.copy()
    H_inv = np.linalg.inv(H)

    # Draw projected corners
    tgt = tgt_corners.reshape(1,-1,2).astype(np.float32)
    img_pts = cv2.perspectiveTransform(tgt, H_inv)[0]
    for (x,y), lab in zip(img_pts.astype(int), ["TL","TR","BR","BL"]):
        cv2.circle(dbg, (x,y), 7, (0,0,255), -1)
        cv2.putText(dbg, lab, (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Grid extents from target corners' AABB
    minX = float(tgt_corners[:,0].min()); maxX = float(tgt_corners[:,0].max())
    minY = float(tgt_corners[:,1].min()); maxY = float(tgt_corners[:,1].max())
    if grid_step is None or grid_step <= 0:
        # heuristic: 10 divisions along min dimension
        grid_step = max(1.0, min(maxX-minX, maxY-minY)/10.0)

    # Vertical lines at X = constant
    xs = np.arange(minX, maxX + 1e-6, grid_step)
    for X in xs:
        seg = np.array([[X, minY], [X, maxY]], dtype=np.float32).reshape(1,-1,2)
        seg_px = cv2.perspectiveTransform(seg, H_inv)[0]
        p1 = tuple(np.round(seg_px[0]).astype(int)); p2 = tuple(np.round(seg_px[1]).astype(int))
        cv2.line(dbg, p1, p2, (0,255,0), 1, cv2.LINE_AA)

    # Horizontal lines at Y = constant
    ys = np.arange(minY, maxY + 1e-6, grid_step)
    for Y in ys:
        seg = np.array([[minX, Y], [maxX, Y]], dtype=np.float32).reshape(1,-1,2)
        seg_px = cv2.perspectiveTransform(seg, H_inv)[0]
        p1 = tuple(np.round(seg_px[0]).astype(int)); p2 = tuple(np.round(seg_px[1]).astype(int))
        cv2.line(dbg, p1, p2, (0,255,0), 1, cv2.LINE_AA)

    _ensure_dir(out_png)
    cv2.imwrite(out_png, dbg)
    return out_png

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="data/images/maze.jpg", help="Path to input image (used for overlay/rectified).")
    ap.add_argument("--corners-json", default="data/calibration/maze_corners.json", help="4x2 TL,TR,BR,BL pixel corners from detector.")

    ap.add_argument("--out-npy", default="data/calibration/homography.npy", help="Save H (pixels->target) here as .npy")
    ap.add_argument("--overlay", type=str, default="data/images/backprojection.png", help="Save backprojection overlay PNG here.")
    ap.add_argument("--grid-step", type=float, default=None, help="Grid step in target units (px for rect, mm for robot).")

    # Robot mode options
    ap.add_argument("--robot-corners-json", type=str, default="data/calibration/robot_corners.json",
                    help="4x2 TL,TR,BR,BL in robot units (alternative to --robot-corners).")

    args = ap.parse_args()

    # --- Load image & pixel corners ---
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    
    C = read_corners_json(args.corners_json)
    # Ensure ordering is TL,TR,BR,BL (safe if file is already ordered)
    C = _order_tl_tr_br_bl(C)

    D = read_corners_json(args.robot_corners_json)
    D = _order_tl_tr_br_bl(D)
    
    # --- Compute H : pixels -> target ---
    H = cv2.getPerspectiveTransform(C.astype(np.float32), D.astype(np.float32))

    # --- Save H ---
    _ensure_dir(args.out_npy)
    np.save(args.out_npy, H)

    # --- Optional overlay: draw backprojected grid/corners ---
    overlay_path = None
    if args.overlay:
        overlay_path = _draw_backprojection_overlay(
            img, H, D, args.grid_step, args.overlay
        )

    # --- Simple condition check (optional) ---
    Hn = H / (H[2,2] if abs(H[2,2]) > 1e-12 else 1.0)
    try:
        cond = float(np.linalg.cond(Hn))
    except Exception:
        cond = float("nan")

    # --- Report ---
    print(json.dumps({
        "status": "ok",
        "H_shape": list(H.shape),
        "H": Hn.round(6).tolist(),          # normalized for readability
        "out_npy": args.out_npy,
        "overlay": overlay_path or "",
        "cond_number": cond
    }, indent=2))
    
if __name__ == "__main__":
    main()
