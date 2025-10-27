#!/usr/bin/env python3
"""
Fast maze solver (no NetworkX). Works on scans/photos of mazes with black walls.

Algorithm (O(N) time, N = #pixels in cropped maze):
1) Read image, downscale for speed (optional).
2) Otsu threshold → corridors (1) vs walls (0); clean with morphology.
3) Crop to the bounding box of walls (removes paper margins/ruled lines).
4) Detect entrance/exit as the LARGEST gaps along the outer wall lines:
   - Prefer TOP↔BOTTOM openings; else LEFT↔RIGHT; else farthest pair.
5) Close the border everywhere except at the two openings.
6) 4-connected BFS through corridor pixels (no diagonals).
7) Reconstruct the path, compress to segments (e.g., D133,R30,...), draw overlay.

Usage:
  python a_star_fast.py path/to/maze.jpg
"""

from __future__ import annotations
import argparse, json, os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np


@dataclass
class Params:
    maxdim: int = 1200            # downscale long side to this many pixels (0 = disable)
    blur_ksize: int = 5           # Gaussian blur kernel (odd >=3). 0 = disable
    wall_open: int = 3            # morphology OPEN (remove salt noise) on walls; 0 = disable
    wall_close: int = 9           # morphology CLOSE (seal tiny ink gaps) on walls; 0 = disable
    prefer_top_bottom: bool = True


# ---------- Small helpers ----------

def resize_keep_aspect(gray: np.ndarray, maxdim: int) -> Tuple[np.ndarray, float]:
    if maxdim <= 0:
        return gray, 1.0
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= maxdim:
        return gray, 1.0
    s = maxdim / float(m)
    out = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return out, s


def binarize_and_clean(gray: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """Return (corridor_mask, wall_mask) as uint8 0/1."""
    g = gray
    if p.blur_ksize and p.blur_ksize >= 3:
        k = p.blur_ksize if p.blur_ksize % 2 == 1 else p.blur_ksize + 1
        g = cv2.GaussianBlur(g, (k, k), 0)

    # Otsu threshold
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_as_corr = (th == 255).astype(np.uint8)
    black_as_corr = 1 - white_as_corr

    # pick the polarity with larger interior mass
    def interior_mass(mask):
        h, w = mask.shape; b = max(2, min(h, w) // 100)
        return mask[b:h-b, b:w-b].sum()
    corr = white_as_corr if interior_mass(white_as_corr) >= interior_mass(black_as_corr) else black_as_corr

    walls = 1 - corr
    if p.wall_open and p.wall_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.wall_open, p.wall_open))
        walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, k, iterations=1)
    if p.wall_close and p.wall_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.wall_close, p.wall_close))
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k, iterations=1)
    corr = 1 - walls
    return corr, walls


def crop_to_walls(gray: np.ndarray, walls: np.ndarray, pad: int = 3) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """Crop to wall bounding box; returns (gray_roi, corr_roi, (y0,y1,x0,x1))."""
    ys, xs = np.where(walls > 0)
    if ys.size == 0:
        h, w = gray.shape[:2]
        return gray, (1 - walls), (0, h, 0, w)
    y0, y1 = max(0, ys.min() - pad), min(gray.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(gray.shape[1], xs.max() + 1 + pad)
    return gray[y0:y1, x0:x1], (1 - walls)[y0:y1, x0:x1], (y0, y1, x0, x1)


def largest_gaps_along_wall_lines(roi_walls: np.ndarray, prefer_top_bottom=True) -> Tuple[str,int, str,int]:
    """
    Detect the outer wall lines (top/bottom or left/right) and return two openings:
    (side1, center1), (side2, center2). Centers are column (for top/bottom) or row (for left/right) within ROI.
    """
    H, W = roi_walls.shape
    band = max(20, min(H, W) // 25)  # search band near each edge
    thick = max(6, min(H, W) // 120)

    def h_gaps(y, x0, x1):
        y0 = max(0, y - thick // 2); y1 = min(H - 1, y + thick // 2)
        has_wall = (roi_walls[y0:y1+1, x0:x1+1].max(axis=0) > 0)  # per col
        gaps, in_gap, s = [], False, 0
        for i, w in enumerate(has_wall):
            if not w and not in_gap: in_gap, s = True, i
            elif w and in_gap: gaps.append((s, i-1)); in_gap = False
        if in_gap: gaps.append((s, len(has_wall)-1))
        return [(x0+a, x0+b, b-a+1, x0+(a+b)//2) for (a,b) in gaps]  # (start, end, len, center)

    def v_gaps(x, y0, y1):
        x0 = max(0, x - thick // 2); x1 = min(W - 1, x + thick // 2)
        has_wall = (roi_walls[y0:y1+1, x0:x1+1].max(axis=1) > 0)  # per row
        gaps, in_gap, s = [], False, 0
        for i, w in enumerate(has_wall):
            if not w and not in_gap: in_gap, s = True, i
            elif w and in_gap: gaps.append((s, i-1)); in_gap = False
        if in_gap: gaps.append((s, len(has_wall)-1))
        return [(y0+a, y0+b, b-a+1, y0+(a+b)//2) for (a,b) in gaps]  # (start, end, len, center)

    # score “heaviest” wall line near each side
    top_band = roi_walls[:band, :];    Ty = int(np.argmax(top_band.sum(axis=1)))
    bot_band = roi_walls[H-band:H, :]; By = H - band + int(np.argmax(bot_band.sum(axis=1)))
    lef_band = roi_walls[:, :band];    Lx = int(np.argmax(lef_band.sum(axis=0)))
    rig_band = roi_walls[:, W-band:W]; Rx = W - band + int(np.argmax(rig_band.sum(axis=0)))

    top_gaps = h_gaps(Ty, 0, W-1)
    bot_gaps = h_gaps(By, 0, W-1)
    lef_gaps = v_gaps(Lx, 0, H-1)
    rig_gaps = v_gaps(Rx, 0, H-1)

    pick = lambda gaps: max(gaps, key=lambda g: g[2]) if gaps else None

    if prefer_top_bottom and top_gaps and bot_gaps:
        t = pick(top_gaps); b = pick(bot_gaps)
        return ("top", t[3]), ("bottom", b[3])
    if lef_gaps and rig_gaps:
        l = pick(lef_gaps); r = pick(rig_gaps)
        return ("left", l[3]), ("right", r[3])

    # fallback: globally two largest gaps regardless of side
    all_gaps = []
    for side, gs in [("top",top_gaps), ("bottom",bot_gaps), ("left",lef_gaps), ("right",rig_gaps)]:
        for a,b,l,c in gs: all_gaps.append((side,c,l))
    if len(all_gaps) >= 2:
        all_gaps.sort(key=lambda t: t[2], reverse=True)
        (s1,c1,_), (s2,c2,_) = all_gaps[:2]
        return (s1,c1), (s2,c2)

    raise RuntimeError("No openings found on any outer wall line")


def bfs_path(cm: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """4-connected BFS in corridor mask (1=free, 0=blocked)."""
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
    # reconstruct
    path = []
    if prev[goal][0] != -1 or goal == start:
        cur = goal
        while True:
            path.append(cur)
            if cur == start: break
            y, x = cur; py, px = prev[y, x]; cur = (int(py), int(px))
        path.reverse()
    return path


def compress_moves(path: List[Tuple[int,int]]) -> List[str]:
    """Convert pixel steps to ['D133','R30', ...]."""
    if len(path) < 2: return []
    def step(a,b):
        (y1,x1),(y2,x2) = a,b
        if y2==y1-1 and x2==x1: return "U"
        if y2==y1+1 and x2==x1: return "D"
        if x2==x1-1 and y2==y1: return "L"
        if x2==x1+1 and y2==y1: return "R"
        return "?"
    moves = [step(path[i], path[i+1]) for i in range(len(path)-1)]
    comp, cur, cnt = [], moves[0], 1
    for m in moves[1:]:
        if m == cur: cnt += 1
        else: comp.append(f"{cur}{cnt}"); cur, cnt = m, 1
    comp.append(f"{cur}{cnt}")
    return comp


def solve(image_path: str, p: Params) -> Dict:
    # 1) load + downscale
    gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert gray0 is not None, f"Cannot read {image_path}"
    gray, scale = resize_keep_aspect(gray0, p.maxdim)
    H0, W0 = gray0.shape[:2]

    # 2) binarize + clean
    corr, walls = binarize_and_clean(gray, p)

    # 3) crop to walls (reduce search space)
    gray_roi, corr_roi, (y0,y1,x0,x1) = crop_to_walls(gray, walls, pad=3)
    walls_roi = 1 - corr_roi
    H, W = gray_roi.shape

    # 4) detect entrances
    (side1, c1), (side2, c2) = largest_gaps_along_wall_lines(walls_roi, prefer_top_bottom=p.prefer_top_bottom)

    # 5) close borders except at entrances
    cm = corr_roi.copy()
    cm[0,:] = 0; cm[-1,:] = 0; cm[:,0] = 0; cm[:,-1] = 0
    def seed(side, center):
        if side == "top":    return (1, center)
        if side == "bottom": return (H-2, center)
        if side == "left":   return (center, 1)
        if side == "right":  return (center, W-2)
    start = seed(side1, c1); goal = seed(side2, c2)
    cm[start] = 1; cm[goal] = 1

    # 6) BFS in ROI
    path = bfs_path(cm, start, goal)
    if not path:
        return {"status":"error", "reason":"No path found by BFS (check morphology/downscale)"}

    # 7) overlay on original image size
    overlay_small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (y,x) in path:
        oy, ox = y0 + y, x0 + x
        overlay_small[oy, ox] = (0,0,255)
    overlay = cv2.resize(overlay_small, (W0, H0), interpolation=cv2.INTER_NEAREST)

    # 8) outputs
    moves = compress_moves(path)
    entrance_x = int(round((x0 + c1) / scale)) if side1 in ("top","bottom") else int(round((y0 + c1) / scale))
    exit_x     = int(round((x0 + c2) / scale)) if side2 in ("top","bottom") else int(round((y0 + c2) / scale))

    out_png = os.path.splitext(image_path)[0] + "_solution.png"
    cv2.imwrite(out_png, overlay)

    return {
        "status": "ok",
        "entrance": {"side": side1, "pos_px": entrance_x},
        "exit": {"side": side2, "pos_px": exit_x},
        "path_length_px": len(path),
        "moves": moves,
        "solution_image": out_png,
        "debug": {"roi_box": [y0,y1,x0,x1], "scale": scale, "roi_size": [H,W]}
    }

import numpy as np

def to_py(o):
    """Recursively convert NumPy scalars/arrays to plain Python types."""
    if isinstance(o, dict):   return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, list):   return [to_py(x) for x in o]
    if isinstance(o, tuple):  return [to_py(x) for x in o]   # JSON has no tuple; use list
    if isinstance(o, np.ndarray):  return o.tolist()
    if isinstance(o, (np.integer,)):   return int(o)
    if isinstance(o, (np.floating,)):  return float(o)
    if isinstance(o, (np.bool_,)):     return bool(o)
    return o

# ...
#print(json.dumps(to_py(res), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to maze image (walls dark)")
    ap.add_argument("--maxdim", type=int, default=Params.maxdim)
    ap.add_argument("--open", type=int, default=Params.wall_open, help="walls OPEN kernel (noise removal)")
    ap.add_argument("--close", type=int, default=Params.wall_close, help="walls CLOSE kernel (seal wall gaps)")
    ap.add_argument("--no-topbottom", action="store_true", help="do not prefer top↔bottom entrances")
    args = ap.parse_args()

    p = Params(maxdim=args.maxdim, wall_open=args.open, wall_close=args.close, prefer_top_bottom=not args.no_topbottom)
    res = solve(args.image, p)
    print(json.dumps(to_py(res), indent=2))


if __name__ == "__main__":
    main()
