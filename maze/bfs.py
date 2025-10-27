#!/usr/bin/env python3
"""
Fast maze solver with clearance, supporting a 4x4 grid-centerline path.

Two modes:
  1) mode='grid'  (default) : Treat the maze as an R x C grid (default 4x4).
     - Detect which edges of each cell are open by scanning narrow bands on cell edges.
     - Use Euclidean Distance Transform to enforce a minimum clearance (pixels).
     - BFS on the small cell graph; path connects cell centers → maximal clearance visually.
  2) mode='pixel' : Pixel-level BFS on corridor mask with clearance gate (fast fallback).

Usage:
  python a_star.py the_maze.jpg                      # 4x4 grid-centerline with default clearance
  python a_star.py the_maze.jpg --clearance 6
  python a_star.py the_maze.jpg --rows 4 --cols 4    # Other grid sizes supported if needed
  python a_star.py the_maze.jpg --mode pixel         # Pixel BFS fallback

Output:
  - <input>_solution.png  : overlay with the route in red (centerline in grid mode)
  - JSON summary to stdout (entrance/exit, moves, etc.)

Dependencies:
  pip install opencv-python-headless numpy
"""

from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import cv2


# --------------------------- Parameters ---------------------------

@dataclass
class Params:
    maxdim: int = 1200           # downscale long side to this many px (0 = no downscale)
    blur_ksize: int = 5          # Gaussian blur (odd >=3). 0 disables blur
    wall_open: int = 3           # morphology OPEN (despeckle) on walls; 0 disables
    wall_close: int = 9          # morphology CLOSE (seal tiny gaps) on walls; 0 disables
    rows: int = 4                # grid rows
    cols: int = 4                # grid cols
    prefer_top_bottom: bool = True  # pick top↔bottom entrances if available
    clearance_px: int = 6        # minimum clearance (pixels) from walls
    auto_relax: bool = True      # relax clearance automatically if no grid path exists
    mode: str = "grid"           # "grid" (centerline) or "pixel" (pixel BFS with clearance)


# --------------------------- Utilities ---------------------------

def to_py(o):
    """Make JSON-safe by converting NumPy scalars/arrays to Python types."""
    if isinstance(o, dict):   return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, list):   return [to_py(x) for x in o]
    if isinstance(o, tuple):  return [to_py(x) for x in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    return o

def resize_keep_aspect(gray: np.ndarray, maxdim: int) -> Tuple[np.ndarray, float]:
    if maxdim <= 0: return gray, 1.0
    h, w = gray.shape[:2]; m = max(h, w)
    if m <= maxdim: return gray, 1.0
    s = maxdim / float(m)
    out = cv2.resize(gray, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
    return out, s

def binarize_and_clean(gray: np.ndarray, blur_ksize: int, wall_open: int, wall_close: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (corridor_mask, wall_mask) as uint8 0/1; auto-detect corridor polarity."""
    g = gray
    if blur_ksize and blur_ksize >= 3:
        k = blur_ksize if (blur_ksize % 2 == 1) else blur_ksize + 1
        g = cv2.GaussianBlur(g, (k, k), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pos = (th == 255).astype(np.uint8)  # white corridors
    neg = 1 - pos                       # black corridors (if inverted)

    # choose corridor polarity with more interior mass
    def interior_mass(mask):
        h, w = mask.shape; b = max(2, min(h, w)//100)
        return mask[b:h-b, b:w-b].sum()
    corr = pos if interior_mass(pos) >= interior_mass(neg) else neg
    walls = 1 - corr

    if wall_open and wall_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (wall_open, wall_open))
        walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, k, iterations=1)
    if wall_close and wall_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (wall_close, wall_close))
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k, iterations=1)

    corr = 1 - walls
    return corr, walls

def crop_to_walls(gray: np.ndarray, walls: np.ndarray, pad: int = 3) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """Crop to the bounding box of wall pixels (plus small pad)."""
    ys, xs = np.where(walls > 0)
    if ys.size == 0:
        h, w = gray.shape[:2]
        return gray, (1 - walls), (0, h, 0, w)
    y0, y1 = max(0, ys.min()-pad), min(gray.shape[0], ys.max()+1+pad)
    x0, x1 = max(0, xs.min()-pad), min(gray.shape[1], xs.max()+1+pad)
    return gray[y0:y1, x0:x1], (1 - walls)[y0:y1, x0:x1], (y0, y1, x0, x1)

def compress_moves_cells(path_cells: List[Tuple[int,int]]) -> str:
    """Cell-to-cell moves: 'U,R,R,D,...' (one char per step)."""
    if len(path_cells) < 2: return ""
    def step(a,b):
        (r1,c1),(r2,c2) = a,b
        if (r2,c2)==(r1-1,c1): return "U"
        if (r2,c2)==(r1+1,c1): return "D"
        if (r2,c2)==(r1,c1-1): return "L"
        if (r2,c2)==(r1,c1+1): return "R"
        return "?"
    return ",".join(step(path_cells[i], path_cells[i+1]) for i in range(len(path_cells)-1))

def bfs_cells(nbrs_fn, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """BFS on a tiny R×C grid graph."""
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal: break
        for v in nbrs_fn(u):
            if v not in prev:
                prev[v] = u; q.append(v)
    if goal not in prev: return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur); cur = prev[cur]
    return path[::-1]

def bfs_pixel(cm: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """4-connected BFS on a 0/1 corridor mask."""
    H, W = cm.shape
    prev = -np.ones((H, W, 2), dtype=np.int32)
    q = deque([start]); seen = np.zeros((H, W), np.uint8); seen[start] = 1
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        y, x = q.popleft()
        if (y, x) == goal: break
        for dy, dx in dirs:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and cm[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = 1; prev[ny, nx] = (y, x); q.append((ny, nx))
    path = []
    if prev[goal][0] != -1 or goal == start:
        cur = goal
        while True:
            path.append(cur)
            if cur == start: break
            y, x = cur; py, px = prev[y, x]; cur = (int(py), int(px))
        path.reverse()
    return path


# -------------- Grid-centerline solver (clearance-enforced) --------------

def solve_grid_centerline(image_path: str, p: Params) -> Dict:
    """
    1) Preprocess → corridors, walls; crop to wall ROI.
    2) Distance transform on corridors → per-pixel clearance from walls.
    3) Build a 4x4 (rows×cols) cell graph: an edge is allowed only if the band near the shared edge
       has enough corridor coverage and min(distance) >= clearance_px.
    4) Detect entrance (top gap) and exit (bottom gap) on the outer wall line and map to columns.
    5) BFS on the cell graph; draw center-to-center polylines; return summary.
    """
    gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        return {"status":"error","reason":f"Cannot read {image_path}"}
    gray, scale = resize_keep_aspect(gray0, p.maxdim)
    H0, W0 = gray0.shape[:2]

    corr, walls = binarize_and_clean(gray, p.blur_ksize, p.wall_open, p.wall_close)
    gray_roi, corr_roi, (y0,y1,x0,x1) = crop_to_walls(gray, walls, pad=3)
    walls_roi = 1 - corr_roi
    H, W = gray_roi.shape

    # Euclidean distance to nearest wall, in ROI pixels
    corr_for_dt = corr_roi.copy()
    corr_for_dt[0,:]=0; corr_for_dt[-1,:]=0; corr_for_dt[:,0]=0; corr_for_dt[:,-1]=0
    dist = cv2.distanceTransform((corr_for_dt*255).astype(np.uint8), cv2.DIST_L2, 3)

    rows, cols = p.rows, p.cols

    # Cell geometry helpers (ROI coordinates)
    def cell_box(r,c):
        y0c = int(round(r * (H / rows))); y1c = int(round((r+1) * (H / rows)))
        x0c = int(round(c * (W / cols))); x1c = int(round((c+1) * (W / cols)))
        return y0c, y1c, x0c, x1c

    def cell_center(r,c):
        y0c,y1c,x0c,x1c = cell_box(r,c)
        return ((y0c+y1c)//2, (x0c+x1c)//2)

    def edge_band(r,c,side, inner_frac=0.6, band_frac=0.12):
        """Return slices (by, bx) for a narrow band near 'side' inside cell (r,c)."""
        y0c,y1c,x0c,x1c = cell_box(r,c)
        hh = y1c - y0c; ww = x1c - x0c
        band = max(2, int(round(min(hh, ww) * band_frac)))
        if side == "N":
            xs0 = x0c + int(round((1-inner_frac)/2 * ww)); xs1 = x1c - int(round((1-inner_frac)/2 * ww))
            return (slice(y0c, y0c+band), slice(xs0, xs1))
        if side == "S":
            xs0 = x0c + int(round((1-inner_frac)/2 * ww)); xs1 = x1c - int(round((1-inner_frac)/2 * ww))
            return (slice(y1c-band, y1c), slice(xs0, xs1))
        if side == "W":
            ys0 = y0c + int(round((1-inner_frac)/2 * hh)); ys1 = y1c - int(round((1-inner_frac)/2 * hh))
            return (slice(ys0, ys1), slice(x0c, x0c+band))
        # "E"
        ys0 = y0c + int(round((1-inner_frac)/2 * hh)); ys1 = y1c - int(round((1-inner_frac)/2 * hh))
        return (slice(ys0, ys1), slice(x1c-band, x1c))

    def band_open_with_clearance(by: slice, bx: slice, min_cov=0.20) -> bool:
        """Is this band an open corridor with enough clearance everywhere?"""
        band_corr = corr_roi[by, bx]
        cov = band_corr.mean() if band_corr.size else 0.0
        if cov < min_cov:  # mostly wall → closed
            return False
        # On corridor pixels, the min distance must exceed requested clearance
        band_dist = dist[by, bx]
        if band_dist.size == 0: return False
        # Only look where corridor exists
        mask = (band_corr > 0)
        if not mask.any(): return False
        min_d = float(band_dist[mask].min())
        return min_d >= max(1, int(p.clearance_px))

    # Build openness per cell side
    cells = [[{"N":False,"E":False,"S":False,"W":False} for _ in range(cols)] for __ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            for side in ("N","E","S","W"):
                by, bx = edge_band(r,c,side)
                cells[r][c][side] = band_open_with_clearance(by, bx)

    # --- Detect entrance/exit on the ROI outer wall lines and map to top/bottom cells ---
    def largest_gaps_along_wall_lines(roi_walls: np.ndarray, prefer_top_bottom=True):
        band = max(20, min(H, W)//25)
        thick = max(6, min(H, W)//120)
        def h_gaps(y, x0, x1):
            y0b = max(0, y - thick//2); y1b = min(H-1, y + thick//2)
            has_wall = (roi_walls[y0b:y1b+1, x0:x1+1].max(axis=0) > 0)
            gaps=[]; in_gap=False; s=0
            for i, w in enumerate(has_wall):
                if not w and not in_gap: in_gap=True; s=i
                elif w and in_gap: gaps.append((s,i-1)); in_gap=False
            if in_gap: gaps.append((s, len(has_wall)-1))
            return [(x0+a, x0+b, b-a+1, x0+(a+b)//2) for (a,b) in gaps]
        # strongest wall rows near top/bottom
        Ty = int(np.argmax(roi_walls[:band, :].sum(axis=1)))
        By = H - band + int(np.argmax(roi_walls[H-band:H, :].sum(axis=1)))
        top_gaps = h_gaps(Ty, 0, W-1); bot_gaps = h_gaps(By, 0, W-1)
        if prefer_top_bottom and top_gaps and bot_gaps:
            tg = max(top_gaps, key=lambda t: t[2]); bg = max(bot_gaps, key=lambda t: t[2])
            return ("top", tg[3]), ("bottom", bg[3])
        # Fallback: choose two largest gaps overall (top/bottom/left/right). For 4x4 examples, top/bottom is typical.
        raise RuntimeError("Could not find top/bottom openings; consider --mode pixel.")

    try:
        (side1, cx1), (side2, cx2) = largest_gaps_along_wall_lines(walls_roi, prefer_top_bottom=p.prefer_top_bottom)
        if side1 != "top" or side2 != "bottom":
            # grid mode expects vertical entrance/exit; else bail to pixel mode
            raise RuntimeError("Openings not top/bottom; use pixel mode or extend mapping.")
    except Exception as e:
        if p.mode == "grid":
            return {"status":"error","reason":f"Grid mode failed: {e}"}
        else:
            raise

    # Map gap centers (columns) to top/bottom cell columns
    def col_index_from_x(x):
        # clamp into [0, cols-1]
        c = int(np.clip(np.floor((x / max(1, W)) * cols), 0, cols-1))
        return c
    c_start = col_index_from_x(cx1)
    c_goal  = col_index_from_x(cx2)

    # Keep ONLY one border opening on top and bottom
    for c in range(cols):
        cells[0][c]["N"] = (c == c_start) and cells[0][c]["N"]
        cells[rows-1][c]["S"] = (c == c_goal) and cells[rows-1][c]["S"]

    # --- Build neighbors with mutual openness and clearance ---
    def neighbors(rc):
        r, c = rc
        out = []
        # up
        if r > 0 and cells[r][c]["N"] and cells[r-1][c]["S"]:
            out.append((r-1, c))
        # down
        if r < rows-1 and cells[r][c]["S"] and cells[r+1][c]["N"]:
            out.append((r+1, c))
        # left
        if c > 0 and cells[r][c]["W"] and cells[r][c-1]["E"]:
            out.append((r, c-1))
        # right
        if c < cols-1 and cells[r][c]["E"] and cells[r][c+1]["W"]:
            out.append((r, c+1))
        return out

    start, goal = (0, c_start), (rows-1, c_goal)
    path_cells = bfs_cells(neighbors, start, goal)

    # Auto-relax clearance if no grid path (optional)
    if not path_cells and p.auto_relax:
        # Decrease clearance gradually down to 1 px
        orig_clear = p.clearance_px
        for rclear in range(orig_clear-1, 0, -max(1, orig_clear//10)):
            for rr in range(rows):
                for cc in range(cols):
                    for side in ("N","E","S","W"):
                        by, bx = edge_band(rr,cc,side)
                        cells[rr][cc][side] = band_open_with_clearance(by, bx, min_cov=0.20) if (side not in ("N","S") or (rr not in (0, rows-1))) else cells[rr][cc][side]
            path_cells = bfs_cells(neighbors, start, goal)
            if path_cells:
                p.clearance_px = rclear
                break

    if not path_cells:
        return {"status":"error","reason":"No grid path found with requested clearance. Try --mode pixel or lower --clearance."}

    # --- Render centerline path on original-size image ---
    overlay_small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pts = []
    for (r,c) in path_cells:
        cy, cx = cell_center(r,c)                # ROI coords
        oy, ox = y0 + cy, x0 + cx               # map back to downscaled canvas
        pts.append((ox, oy))                    # OpenCV uses (x,y)
    for i in range(len(pts)-1):
        cv2.line(overlay_small, pts[i], pts[i+1], (0,0,255), 2)

    overlay = cv2.resize(overlay_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
    out_png = os.path.splitext(image_path)[0] + "_solution.png"
    cv2.imwrite(out_png, overlay)

    res = {
        "status": "ok",
        "mode": "grid",
        "grid_size": {"rows": rows, "cols": cols},
        "entrance": {"cell":[0, c_start], "side":"top"},
        "exit": {"cell":[rows-1, c_goal], "side":"bottom"},
        "path_cells": path_cells,
        "moves": compress_moves_cells(path_cells),
        "clearance_px_used": int(p.clearance_px),
        "solution_image": out_png,
        "debug": {
            "roi_box": [y0,y1,x0,x1],
            "scale": float(scale),
            "roi_size": [H,W]
        }
    }
    return res


# -------------- Pixel-level fallback (clearance-enforced) --------------

def solve_pixel_bfs(image_path: str, p: Params) -> Dict:
    """Fast pixel BFS with clearance gate (distance transform)."""
    gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        return {"status":"error","reason":f"Cannot read {image_path}"}
    gray, scale = resize_keep_aspect(gray0, p.maxdim)
    H0, W0 = gray0.shape[:2]

    corr, walls = binarize_and_clean(gray, p.blur_ksize, p.wall_open, p.wall_close)
    gray_roi, corr_roi, (y0,y1,x0,x1) = crop_to_walls(gray, walls, pad=3)
    walls_roi = 1 - corr_roi
    H, W = gray_roi.shape

    # openings on top/bottom wall lines
    band = max(20, min(H, W)//25)
    top_band = walls_roi[:band,:];   bot_band = walls_roi[H-band:H,:]
    Ty = int(np.argmax(top_band.sum(axis=1)))
    By = H - band + int(np.argmax(bot_band.sum(axis=1)))
    def h_gaps(y, x0, x1):
        thick = max(6, min(H,W)//120)
        y0b=max(0,y-thick//2); y1b=min(H-1,y+thick//2)
        has_wall = (walls_roi[y0b:y1b+1, x0:x1+1].max(axis=0) > 0)
        gaps=[]; in_gap=False; s=0
        for i,w in enumerate(has_wall):
            if not w and not in_gap: in_gap=True; s=i
            elif w and in_gap: gaps.append((s,i-1)); in_gap=False
        if in_gap: gaps.append((s, len(has_wall)-1))
        return [(x0+a, x0+b, b-a+1, x0+(a+b)//2) for (a,b) in gaps]
    top_gap = max(h_gaps(Ty, 0, W-1), key=lambda t: t[2])
    bot_gap = max(h_gaps(By, 0, W-1), key=lambda t: t[2])

    start = (Ty+1, top_gap[3]); goal = (By-1, bot_gap[3])

    # clearance gate via distance transform
    corr_for_dt = corr_roi.copy()
    corr_for_dt[0,:]=0; corr_for_dt[-1,:]=0; corr_for_dt[:,0]=0; corr_for_dt[:,-1]=0
    dist = cv2.distanceTransform((corr_for_dt*255).astype(np.uint8), cv2.DIST_L2, 3)

    def build_cm(clear_r):
        cm = (dist >= max(1, int(clear_r))).astype(np.uint8)
        cm[0,:]=0; cm[-1,:]=0; cm[:,0]=0; cm[:,-1]=0
        # seed openings
        cm[start]=1; cm[goal]=1
        for (y,x) in (start, goal):
            y0b, y1b = max(0,y-1), min(H-1,y+1)
            x0b, x1b = max(0,x-1), min(W-1,x+1)
            cm[y0b:y1b+1, x0b:x1b+1] = 1
        return cm

    target = int(max(1, p.clearance_px))
    clearance_used = target
    path = bfs_pixel(build_cm(clearance_used), start, goal)
    if not path and p.auto_relax:
        for rclear in range(target-1, 0, -max(1, target//10)):
            path = bfs_pixel(build_cm(rclear), start, goal)
            if path:
                clearance_used = rclear
                break
    if not path:
        return {"status":"error","reason":"No pixel path found with requested clearance."}

    overlay_small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (y,x) in path:
        oy, ox = y0 + y, x0 + x
        overlay_small[oy, ox] = (0,0,255)
    overlay = cv2.resize(overlay_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
    out_png = os.path.splitext(image_path)[0] + "_solution.png"
    cv2.imwrite(out_png, overlay)

    return {
        "status":"ok",
        "mode":"pixel",
        "entrance":{"side":"top","pos_px": int(round((x0+top_gap[3]) / max(1e-6, 1/scale)))},
        "exit":{"side":"bottom","pos_px": int(round((x0+bot_gap[3]) / max(1e-6, 1/scale)))},
        "path_length_px": len(path),
        "clearance_px_used": clearance_used,
        "solution_image": out_png
    }


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to maze image")
    ap.add_argument("--mode", choices=["grid","pixel"], default="grid", help="grid=centerline, pixel=fallback")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--clearance", type=int, default=6, help="clearance in pixels")
    ap.add_argument("--maxdim", type=int, default=1200)
    ap.add_argument("--open", type=int, default=3, help="walls OPEN kernel")
    ap.add_argument("--close", type=int, default=9, help="walls CLOSE kernel")
    ap.add_argument("--no-topbottom", action="store_true", help="do not prefer top↔bottom entrances (grid mode)")
    ap.add_argument("--no-relax", action="store_true", help="disable auto-clearance relaxation")
    args = ap.parse_args()

    p = Params(
        maxdim=args.maxdim,
        wall_open=args.open,
        wall_close=args.close,
        rows=args.rows,
        cols=args.cols,
        prefer_top_bottom=not args.no_topbottom,
        clearance_px=args.clearance,
        auto_relax=not args.no_relax,
        mode=args.mode
    )

    if p.mode == "grid":
        res = solve_grid_centerline(args.image, p)
        # If grid fails for any reason, you can uncomment to auto-fallback:
        # if res.get("status") != "ok":
        #     print("Grid mode failed; falling back to pixel mode...", flush=True)
        #     res = solve_pixel_bfs(args.image, p)
    else:
        res = solve_pixel_bfs(args.image, p)

    print(json.dumps(to_py(res), indent=2))


if __name__ == "__main__":
    main()
