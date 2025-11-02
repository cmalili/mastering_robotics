#!/usr/bin/env python3
"""
Pixel-only maze solver that returns the actual solution path.

Modes
-----
- widest      : Max–min clearance (centerline by maximizing the minimum distance to walls).
- weighted    : Shortest path with distance-to-wall penalty (tunable lambda).
- shortest    : Shortest path subject to a hard minimum clearance.

Outputs (JSON)
--------------
- path_roi_downscaled     : list of [y,x] along the ROI in the downscaled image
- path_downscaled         : list of [y,x] in full downscaled image coordinates
- path_original           : list of [y,x] in the original image coordinates
- waypoints_original      : start + all turn points + end (original coords)
- moves                   : compressed 4-neighbour run-length steps (e.g., ["D133","R7",...])
- solution_image          : overlay PNG with safe region + path

Install
-------
pip install opencv-python-headless numpy
"""

from __future__ import annotations
import argparse, json, os, heapq
from typing import Dict, List, Tuple
from collections import deque

import numpy as np
import cv2


# ---------- JSON helpers ----------
def to_py(o):
    if isinstance(o, dict):   return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, list):   return [to_py(x) for x in o]
    if isinstance(o, tuple):  return [to_py(x) for x in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    return o


# ---------- Image utilities ----------
def resize_keep_aspect(gray: np.ndarray, maxdim: int) -> Tuple[np.ndarray, float]:
    if maxdim <= 0: return gray, 1.0
    h, w = gray.shape[:2]; m = max(h, w)
    if m <= maxdim: return gray, 1.0
    s = float(maxdim) / float(m)
    out = cv2.resize(gray, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
    return out, s

def binarize_and_clean(gray: np.ndarray, blur_ksize: int, wall_open: int, wall_close: int) -> Tuple[np.ndarray, np.ndarray]:
    g = gray
    if blur_ksize and blur_ksize >= 3:
        k = blur_ksize if (blur_ksize % 2 == 1) else blur_ksize + 1
        g = cv2.GaussianBlur(g, (k, k), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pos = (th == 255).astype(np.uint8); neg = 1 - pos
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
    ys, xs = np.where(walls > 0)
    if ys.size == 0:
        h, w = gray.shape[:2]
        return gray, (1 - walls), (0, h, 0, w)
    y0, y1 = max(0, ys.min()-pad), min(gray.shape[0], ys.max()+1+pad)
    x0, x1 = max(0, xs.min()-pad), min(gray.shape[1], xs.max()+1+pad)
    return gray[y0:y1, x0:x1], (1 - walls)[y0:y1, x0:x1], (y0, y1, x0, x1)


# ---------- Openings detection ----------
def detect_openings(roi_walls: np.ndarray, prefer_top_bottom=True) -> Tuple[Tuple[str,int], Tuple[str,int], Dict]:
    H, W = roi_walls.shape
    band  = max(20, min(H, W)//25)
    thick = max(6,  min(H, W)//120)
    def h_gaps(y, x0, x1):
        y0 = max(0, y - thick//2); y1 = min(H-1, y + thick//2)
        has_wall = (roi_walls[y0:y1+1, x0:x1+1].max(axis=0) > 0)
        gaps=[]; in_gap=False; s=0
        for i, w in enumerate(has_wall):
            if not w and not in_gap: in_gap=True; s=i
            elif w and in_gap: gaps.append((s, i-1)); in_gap=False
        if in_gap: gaps.append((s, len(has_wall)-1))
        return [(x0+a, x0+b, b-a+1, x0+(a+b)//2) for (a,b) in gaps]
    def v_gaps(x, y0, y1):
        x0 = max(0, x - thick//2); x1 = min(W-1, x + thick//2)
        has_wall = (roi_walls[y0:y1+1, x0:x1+1].max(axis=1) > 0)
        gaps=[]; in_gap=False; s=0
        for i, w in enumerate(has_wall):
            if not w and not in_gap: in_gap=True; s=i
            elif w and in_gap: gaps.append((s, i-1)); in_gap=False
        if in_gap: gaps.append((s, len(has_wall)-1))
        return [(y0+a, y0+b, b-a+1, y0+(a+b)//2) for (a,b) in gaps]
    top_band = roi_walls[:band, :];   Ty = int(np.argmax(top_band.sum(axis=1)))
    bot_band = roi_walls[H-band:H, :];By = H - band + int(np.argmax(bot_band.sum(axis=1)))
    lef_band = roi_walls[:, :band];   Lx = int(np.argmax(lef_band.sum(axis=0)))
    rig_band = roi_walls[:, W-band:W];Rx = W - band + int(np.argmax(rig_band.sum(axis=0)))
    top_gaps = h_gaps(Ty, 0, W-1)
    bot_gaps = h_gaps(By, 0, W-1)
    lef_gaps = v_gaps(Lx, 0, H-1)
    rig_gaps = v_gaps(Rx, 0, H-1)
    dbg = {"Ty":Ty, "By":By, "Lx":Lx, "Rx":Rx,
           "top_gaps": len(top_gaps), "bot_gaps": len(bot_gaps),
           "lef_gaps": len(lef_gaps), "rig_gaps": len(rig_gaps)}
    def pick_big(gs): return max(gs, key=lambda g: g[2]) if gs else None
    if prefer_top_bottom and top_gaps and bot_gaps:
        t = pick_big(top_gaps); b = pick_big(bot_gaps)
        return ("top", t[3]), ("bottom", b[3]), dbg
    if lef_gaps and rig_gaps:
        l = pick_big(lef_gaps); r = pick_big(rig_gaps)
        return ("left", l[3]), ("right", r[3]), dbg
    allg = []
    for side, gs in [("top", top_gaps), ("bottom", bot_gaps), ("left", lef_gaps), ("right", rig_gaps)]:
        for a,b,l,c in gs: allg.append((side, c, l))
    if len(allg) >= 2:
        allg.sort(key=lambda t: t[2], reverse=True)
        (s1,c1,_), (s2,c2,_) = allg[:2]
        return (s1,c1), (s2,c2), dbg
    raise RuntimeError("No border openings detected.")


# ---------- Unit conversion ----------
def downscaled_px_from_request(
    scale: float,
    *,
    px_original: int | None = None,
    px_downscaled: int | None = None,
    mm: float | None = None,
    maze_width_mm: float | None = None,
    roi_width_orig_px: int | None = None
) -> int:
    if px_downscaled is not None:
        return max(1, int(px_downscaled))
    if (mm is not None) and (maze_width_mm is not None) and (roi_width_orig_px is not None):
        px_per_mm = roi_width_orig_px / max(maze_width_mm, 1e-6)
        return max(1, int(np.ceil(mm * px_per_mm * scale)))
    if px_original is not None:
        return max(1, int(np.ceil(px_original * scale)))
    return 0  # “no minimum”


# ---------- Path helpers ----------
def bfs_mask(cm: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """Shortest path on a 0/1 mask (4-neighbour)."""
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

def widest_path(dist: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int], include_gate: bool=False) -> Tuple[float, List[Tuple[int,int]]]:
    """Max–min clearance route on 4-neighbour grid."""
    H, W = dist.shape
    d = dist.copy()
    if not include_gate:
        big = float(dist.max() + 1.0)
        d[start] = big; d[goal] = big
    cap = np.full((H, W), -np.inf, dtype=float)
    parent_y = -np.ones((H, W), dtype=np.int32); parent_x = -np.ones((H, W), dtype=np.int32)
    cap[start] = d[start]
    heap = [(-cap[start], start)]
    visited = np.zeros((H, W), dtype=np.uint8)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    while heap:
        negc, (y, x) = heapq.heappop(heap)
        if visited[y, x]: continue
        visited[y, x] = 1
        if (y, x) == goal: break
        for dy, dx in dirs:
            ny, nx = y+dy, x+dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W: continue
            if dist[ny, nx] <= 0: continue
            cand = min(cap[y, x], d[ny, nx])
            if cand > cap[ny, nx]:
                cap[ny, nx] = cand
                parent_y[ny, nx] = y; parent_x[ny, nx] = x
                heapq.heappush(heap, (-cand, (ny, nx)))
    if cap[goal] <= 0 or parent_y[goal] == -1:
        return 0.0, []
    path=[]; gy,gx=goal
    while not (gy==start[0] and gx==start[1]):
        path.append((gy,gx)); py,px=parent_y[gy,gx],parent_x[gy,gx]; gy,gx=int(py),int(px)
    path.append(start); path.reverse()
    return float(cap[goal]), path

def dijkstra_weighted(dist: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int], lam: float=6.0, eps: float=1.0) -> Tuple[List[Tuple[int,int]], float, float]:
    """Shortest path with cost(v)=1 + lam/(eps + dist(v)); returns path, min(dist), avg(dist)."""
    H, W = dist.shape
    INF = 1e18
    def w(y,x):
        if dist[y,x] <= 0: return INF
        return 1.0 + lam / (eps + float(dist[y,x]))
    g = np.full((H,W), INF, dtype=float)
    parent_y = -np.ones((H,W), dtype=np.int32); parent_x = -np.ones((H,W), dtype=np.int32)
    g[start] = 0.0
    heap = [(0.0, start)]
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    while heap:
        cost,(y,x) = heapq.heappop(heap)
        if cost != g[y,x]: continue
        if (y,x) == goal: break
        for dy,dx in dirs:
            ny,nx=y+dy,x+dx
            if ny<0 or ny>=H or nx<0 or nx>=W: continue
            c = w(ny,nx)
            if c>=INF: continue
            new = cost + c
            if new < g[ny,nx]:
                g[ny,nx] = new
                parent_y[ny,nx] = y; parent_x[ny,nx] = x
                heapq.heappush(heap, (new,(ny,nx)))
    if parent_y[goal] == -1:
        return [], 0.0, 0.0
    path=[]; y,x=goal
    while not (y==start[0] and x==start[1]):
        path.append((y,x)); py,px=parent_y[y,x],parent_x[y,x]; y,x=int(py),int(px)
    path.append(start); path.reverse()
    dvals = [float(dist[y,x]) for (y,x) in path]
    return path, float(min(dvals)), float(sum(dvals)/max(1,len(dvals)))

def compress_straight_runs(path: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Return only start + turn points + end."""
    if not path: return []
    out = [path[0]]
    def dir_of(a,b):
        (y1,x1),(y2,x2)=a,b
        return (y2-y1, x2-x1)
    cur_dir = None
    for i in range(1, len(path)):
        d = dir_of(path[i-1], path[i])
        if cur_dir is None:
            cur_dir = d
        elif d != cur_dir:
            out.append(path[i-1])
            cur_dir = d
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out

def compress_moves(path: List[Tuple[int,int]]) -> List[str]:
    """Run-length encode U/D/L/R steps."""
    if len(path) < 2: return []
    def step(a,b):
        (y1,x1),(y2,x2)=a,b
        if y2==y1-1 and x2==x1: return "U"
        if y2==y1+1 and x2==x1: return "D"
        if x2==x1-1 and y2==y1: return "L"
        if x2==x1+1 and y2==y1: return "R"
        return "?"
    moves=[step(path[i], path[i+1]) for i in range(len(path)-1)]
    comp=[]; cur=moves[0]; cnt=1
    for m in moves[1:]:
        if m==cur: cnt+=1
        else: comp.append(f"{cur}{cnt}"); cur=m; cnt=1
    comp.append(f"{cur}{cnt}")
    return comp


# ---------- Overlay ----------
def draw_safe_and_path_on_canvas(canvas_bgr, safe_mask_roi, path_pts_roi, roi_box, safe_alpha=0.25):
    y0, y1, x0, x1 = roi_box
    roi = canvas_bgr[y0:y1, x0:x1]
    green = np.zeros_like(roi); green[:,:,1] = 255
    blended = ((1.0 - safe_alpha) * roi + safe_alpha * green).astype(np.uint8)
    mask3 = safe_mask_roi.astype(bool)[:, :, None]
    roi[:] = np.where(mask3, blended, roi)
    for (y,x) in path_pts_roi:
        oy, ox = y0 + y, x0 + x
        canvas_bgr[oy, ox] = (0,0,255)
    return canvas_bgr


# ---------- Main solve ----------
def solve_centerline(image_path: str, args) -> Dict:
    # Load + downscale
    gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        return {"status":"error","reason":f"Cannot read {image_path}"}
    gray, scale = resize_keep_aspect(gray0, args.maxdim)
    H0, W0 = gray0.shape[:2]

    # Binarize + crop
    corr, walls = binarize_and_clean(gray, args.blur, args.open, args.close)
    gray_roi, corr_roi, (y0,y1,x0,x1) = crop_to_walls(gray, walls, pad=3)
    walls_roi = 1 - corr_roi
    H, W = gray_roi.shape

    # Openings & seeds
    try:
        (s1,c1), (s2,c2), dbg = detect_openings(walls_roi, prefer_top_bottom=not args.prefer_lr)
    except Exception as e:
        return {"status":"error","reason":f"Opening detection failed: {e}"}
    band = max(20, min(H, W)//25)
    Ty = int(np.argmax(walls_roi[:band, :].sum(axis=1)))
    By = H - band + int(np.argmax(walls_roi[H-band:H, :].sum(axis=1)))
    Lx = int(np.argmax(walls_roi[:, :band].sum(axis=0)))
    Rx = int(np.argmax(walls_roi[:, W-band:W].sum(axis=0))) + (W - band)
    def seed(side, center):
        if side == "top":    return (Ty+1, center)
        if side == "bottom": return (By-1, center)
        if side == "left":   return (center, Lx+1)
        if side == "right":  return (center, Rx-1)
        raise ValueError(side)
    start = seed(s1,c1); goal = seed(s2,c2)

    # EDT (downscaled px)
    corr_dt = corr_roi.copy()
    corr_dt[0,:]=0; corr_dt[-1,:]=0; corr_dt[:,0]=0; corr_dt[:,-1]=0
    dist = cv2.distanceTransform((corr_dt*255).astype(np.uint8), cv2.DIST_L2, 3)

    # Solve by mode
    if args.mode == "shortest":
        roi_width_orig_px = int(round((x1 - x0) / max(scale, 1e-6)))
        T = downscaled_px_from_request(
            scale,
            px_original=None if args.clearance_ref=="downscaled" else args.min_clearance,
            px_downscaled=args.min_clearance if args.clearance_ref=="downscaled" else None,
            mm=args.min_clearance_mm, maze_width_mm=args.maze_width_mm, roi_width_orig_px=roi_width_orig_px
        )
        if T <= 0:
            return {"status":"error","reason":"--min-clearance is required for mode=shortest"}
        safe = (dist >= T).astype(np.uint8); safe[0,:]=0; safe[-1,:]=0; safe[:,0]=0; safe[:,-1]=0
        safe[start] = 1; safe[goal] = 1
        path_roi = bfs_mask(safe, start, goal)
        achieved_min = float(min(dist[y,x] for (y,x) in path_roi)) if path_roi else 0.0
        used_T = T
        mode_stats = {
            "min_clearance_downscaled_px": int(T),
            "min_clearance_equiv_original_px": float(T / max(scale,1e-6)),
            "achieved_min_clearance_downscaled_px": float(achieved_min),
            "achieved_min_clearance_equiv_original_px": float(achieved_min / max(scale,1e-6)),
        }
        if not path_roi:
            base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            dbg_img = draw_safe_and_path_on_canvas(base.copy(), safe, [], (y0,y1,x0,x1))
            dbg_img = cv2.resize(dbg_img, (W0, H0), interpolation=cv2.INTER_NEAREST)
            out_dbg = os.path.splitext(image_path)[0] + "_safe_region_infeasible.png"
            cv2.imwrite(out_dbg, dbg_img)
            return {"status":"error","reason":"Requested minimum clearance is infeasible.","debug_overlay":os.path.basename(out_dbg), **mode_stats}
    elif args.mode == "widest":
        include_gate = bool(args.include_gate)
        achieved_min, path_roi = widest_path(dist, start, goal, include_gate=include_gate)
        if not path_roi:
            return {"status":"error","reason":"No path through corridor (check binarization)."}
        used_T = int(np.floor(achieved_min))
        safe = (dist >= max(1, used_T)).astype(np.uint8); safe[0,:]=0; safe[-1,:]=0; safe[:,0]=0; safe[:,-1]=0
        mode_stats = {
            "achieved_min_clearance_downscaled_px": float(achieved_min),
            "achieved_min_clearance_equiv_original_px": float(achieved_min / max(scale,1e-6)),
        }
    elif args.mode == "weighted":
        path_roi, achieved_min, achieved_avg = dijkstra_weighted(dist, start, goal, lam=float(args.lam), eps=float(args.eps))
        if not path_roi:
            return {"status":"error","reason":"No path through corridor (check binarization)."}
        used_T = int(np.floor(achieved_min))
        safe = (dist >= max(1, used_T)).astype(np.uint8); safe[0,:]=0; safe[-1,:]=0; safe[:,0]=0; safe[:,-1]=0
        mode_stats = {
            "achieved_min_clearance_downscaled_px": float(achieved_min),
            "achieved_avg_clearance_downscaled_px": float(achieved_avg),
            "achieved_min_clearance_equiv_original_px": float(achieved_min / max(scale,1e-6)),
            "achieved_avg_clearance_equiv_original_px": float(achieved_avg / max(scale,1e-6)),
            "weighted_lambda": float(args.lam),
            "weighted_eps": float(args.eps),
        }
    else:
        return {"status":"error","reason":"Unknown mode"}

    # Build coordinate frames for the returned path
    # ROI (downscaled) -> full downscaled
    path_downscaled = [(y0 + y, x0 + x) for (y, x) in path_roi]
    # Downscaled -> original
    invs = 1.0 / max(scale, 1e-6)
    path_original = [(int(round((y0 + y) * invs)), int(round((x0 + x) * invs))) for (y, x) in path_roi]

    # Waypoints = start + turns + end (original coords)
    turns_roi = compress_straight_runs(path_roi)
    waypoints_original = [(int(round((y0 + y) * invs)), int(round((x0 + x) * invs))) for (y, x) in turns_roi]

    # Moves = run-length encoding in ROI coords
    moves = compress_moves(path_roi)

    # Draw overlay at original size (safe region at threshold used_T)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out_small = draw_safe_and_path_on_canvas(base.copy(), safe, path_roi, (y0,y1,x0,x1))
    overlay = cv2.resize(out_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
    out_png = os.path.splitext(image_path)[0] + "_centerline.png"
    cv2.imwrite(out_png, overlay)

    # Entrance/exit positions along their borders in original pixels
    if (s1 in ("top","bottom")):
        entrance_pos_px = int(round((x0 + c1) * invs))
    else:
        entrance_pos_px = int(round((y0 + c1) * invs))
    if (s2 in ("top","bottom")):
        exit_pos_px = int(round((x0 + c2) * invs))
    else:
        exit_pos_px = int(round((y0 + c2) * invs))

    result = {
        "status":"ok",
        "mode": args.mode,
        "entrance": {"side": s1, "pos_px": entrance_pos_px},
        "exit":     {"side": s2, "pos_px": exit_pos_px},
        "solution_image": os.path.basename(out_png),
        "scale": float(scale),
        "threshold_used_downscaled_px": int(used_T),
        "threshold_equiv_original_px": float(used_T / max(scale,1e-6)),
        "path_length_px": len(path_roi),

        # --- HERE ARE THE REQUESTED PATHS ---
        "path_roi_downscaled": [[int(y), int(x)] for (y,x) in path_roi],
        "path_downscaled": [[int(y), int(x)] for (y,x) in path_downscaled],
        "path_original": [[int(y), int(x)] for (y,x) in path_original],
        "waypoints_original": [[int(y), int(x)] for (y,x) in waypoints_original],
        "moves": moves
    }
    result.update(mode_stats)

    # Optional exports
    if args.save_csv:
        with open(args.save_csv, "w") as f:
            f.write("y_original,x_original\n")
            for (y,x) in path_original:
                f.write(f"{y},{x}\n")
        result["saved_csv"] = os.path.abspath(args.save_csv)
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(to_py(result), f, indent=2)
        result["saved_json"] = os.path.abspath(args.save_json)

    return result


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to maze image")

    ap.add_argument("--mode", choices=["widest","weighted","shortest"], default="widest",
                    help="centerline strategy")

    # For mode=shortest (hard min clearance)
    ap.add_argument("--min-clearance", type=int, default=None, help="minimum clearance (original px; auto-scaled)")
    ap.add_argument("--clearance-ref", choices=["original","downscaled"], default="original",
                    help="interpret --min-clearance as original or downscaled px")
    ap.add_argument("--min-clearance-mm", type=float, default=None, help="minimum clearance (mm)")
    ap.add_argument("--maze-width-mm", type=float, default=None, help="maze drawing width (mm) for mm conversion")

    # For mode=weighted
    ap.add_argument("--lam", type=float, default=6.0, help="center preference strength (larger -> more centered)")
    ap.add_argument("--eps", type=float, default=1.0, help="stability term in cost 1 + lam/(eps+dist)")

    # Speed/robustness
    ap.add_argument("--maxdim", type=int, default=1200, help="downscale long side (0=no downscale)")
    ap.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel (odd >=3). 0=disable")
    ap.add_argument("--open", type=int, default=3, help="walls OPEN kernel, 0=disable")
    ap.add_argument("--close", type=int, default=9, help="walls CLOSE kernel, 0=disable")

    # Behavior
    ap.add_argument("--prefer-lr", action="store_true", help="prefer left-right openings if top-bottom absent")
    ap.add_argument("--include-gate", action="store_true", help="widest: include gate slit in clearance score")

    # Exports
    ap.add_argument("--save-csv", type=str, default=None, help="write original-pixel path to CSV here")
    ap.add_argument("--save-json", type=str, default=None, help="write full JSON (including paths) here")

    args = ap.parse_args()

    # Validation for shortest mode
    if args.mode == "shortest" and (args.min_clearance is None and args.min_clearance_mm is None):
        print(json.dumps({"status":"error","reason":"--min-clearance (or --min-clearance-mm) is required for mode=shortest"}))
        return
    if args.min_clearance_mm is not None and args.maze_width_mm is None:
        print(json.dumps({"status":"error","reason":"--maze-width-mm is required with --min-clearance-mm"}))
        return

    res = solve_centerline(args.image, args)
    print(json.dumps(to_py(res), indent=2))


if __name__ == "__main__":
    main()
