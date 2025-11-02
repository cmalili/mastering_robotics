#!/usr/bin/env python3
"""
calib_tools.py
Deterministic tools for calibration (no LLM logic here).

Artifacts produced:
- ref/maze_ref.jpg                  (reference image you calibrate against)
- ref/ref_corners.npy               (TL,TR,BR,BL pixel corners, float32)
- ref/corners_overlay.png           (debug overlay)
- ref/H_ref.npy                     (3x3 homography: reference image pixels -> robot base XY units)
- ref/hverify_overlay.png           (verification overlay with projected grid/corners)
- ref/corresp.txt                   (u v Xr Yr lines; loaded by a tool here)

Conventions:
- Pixels: (u=x, v=y) with origin top-left.
- Corners order: TL, TR, BR, BL.
- Robot coordinates: (Xr, Yr) in your chosen **consistent** unit (e.g., mm).
"""

from __future__ import annotations
from typing import Annotated, Dict, List, Tuple, Optional
import os, json
import numpy as np
import cv2
from dataclasses import dataclass

# ---------- Small helpers ----------

def _ensure_dir(p: str):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def _to_py(o):
    if isinstance(o, dict): return {k:_to_py(v) for k,v in o.items()}
    if isinstance(o, (list, tuple)): return [_to_py(x) for x in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if hasattr(o, "__dict__"): return _to_py(vars(o))
    return o

# ---------- Corner detection ----------

def _order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL given Nx2 (x,y)."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)        # x+y
    d = np.diff(pts, axis=1)   # x - y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _refine_corners(gray: np.ndarray, corners_xy: np.ndarray, win: int = 7) -> np.ndarray:
    gray_f = cv2.GaussianBlur(gray, (3,3), 0)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    c = corners_xy.reshape(-1,1,2).astype(np.float32)
    cv2.cornerSubPix(gray_f, c, (win,win), (-1,-1), term)
    return c.reshape(-1,2)

# ---------- Tools (callable by agents) ----------

def capture_reference_image(
    out_path: Annotated[str, "Where to save the captured image (e.g., ref/maze_ref.jpg)."],
    camera_index: Annotated[int, "OpenCV camera index."] = 0,
    width: Annotated[int, "Capture width."] = 1920,
    height: Annotated[int, "Capture height."] = 1080,
    warmup_frames: Annotated[int, "Throw away this many frames to stabilize exposure."] = 10
) -> Dict:
    """Capture a single frame from a camera and save to disk."""
    _ensure_dir(out_path)
    cap = cv2.VideoCapture(camera_index)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok = cap.isOpened()
    if not ok:
        return {"status":"error","reason":f"Cannot open camera {camera_index}"}
    for _ in range(max(1,warmup_frames)):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"status":"error","reason":"Failed to read a frame"}
    cv2.imwrite(out_path, frame)
    return {"status":"ok","image_path": out_path}

def detect_reference_corners(
    image_path: Annotated[str, "Path to the REFERENCE image (maze_ref.jpg)."],
    out_npy: Annotated[str, "Save corners (TL,TR,BR,BL) here as .npy."] = "ref/ref_corners.npy",
    overlay_path: Annotated[str, "Save a debug overlay PNG path."] = "ref/corners_overlay.png",
    edge_frac: float = 0.35,          # search depth (fraction of width/height) from each border
    sample_step: int = 3,             # row/col stride for scanning
    angle_prior_deg: float = 20.0,    # optional sanity check (vertical ~0°, horizontal ~90° ± this)
    debug_dir: Optional[str] = None   # if set, writes walls/edges debug images here
) -> Dict:
    """
    Corner detector tolerant to open outer corners and notebook lines.

    Strategy:
      - Binarize to isolate walls (thick dark strokes), suppress page ruling via morphology.
      - For each side (L,R,T,B): scan from the border inward and pick the first wall pixel.
      - Fit one line to each side with cv2.fitLine (robust to gaps/outliers).
      - Intersect (L with T/B) and (R with T/B) -> TL,TR,BR,BL (float32).
    """
    def _ok(**kw): o={"status":"ok"}; o.update(kw); return o
    def _err(r, **kw): o={"status":"error","reason":str(r)}; o.update(kw); return o

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return _err(f"Cannot read {image_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1) Walls segmentation (thick black strokes) ---
    # Otsu binarization + polarity selection
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pos = (th == 255).astype(np.uint8)           # light = 1
    neg = 1 - pos                                # dark = 1
    # Choose corridors as the brighter mass in the interior; walls are the inverse
    b = max(2, min(h, w) // 100)
    interior = lambda m: m[b:h-b, b:w-b].sum()
    corridors = pos if interior(pos) >= interior(neg) else neg
    walls = ((1 - corridors) * 255).astype(np.uint8)

    # Morphology tuned to image size: bridge small gaps & remove thin lines
    k_close = max(7, min(h, w) // 40)     # connect broken outer segments
    k_open  = max(3, min(h, w) // 150)    # suppress notebook ruling
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close)), 1)
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open)), 1)
    # Light orientation reinforcement (helps on faint strokes)
    walls_h = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w//20), 3)), 1)
    walls_v = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(15, h//20))), 1)
    walls_or = cv2.bitwise_or(walls_h, walls_v)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "walls.png"), walls)
        cv2.imwrite(os.path.join(debug_dir, "walls_or.png"), walls_or)

    binw = (walls_or > 0).astype(np.uint8)

    # --- 2) Scan inward from borders to collect side points ---
    max_x_l = int(w * edge_frac)
    min_x_r = int(w * (1.0 - edge_frac))
    max_y_t = int(h * edge_frac)
    min_y_b = int(h * (1.0 - edge_frac))

    # LEFT: for each row, first wall pixel from x=0→
    pts_left = []
    for y in range(0, h, max(1, sample_step)):
        col = binw[y, :max_x_l]
        nz = np.where(col > 0)[0]
        if nz.size:
            x = int(nz[0])
            pts_left.append((x, y))
    # RIGHT: for each row, first wall pixel from x=w-1←
    pts_right = []
    for y in range(0, h, max(1, sample_step)):
        col = binw[y, min_x_r:w]
        nz = np.where(col > 0)[0]
        if nz.size:
            x = int(min_x_r + nz[-1])
            pts_right.append((x, y))
    # TOP: for each col, first wall pixel from y=0↓
    pts_top = []
    for x in range(0, w, max(1, sample_step)):
        row = binw[:max_y_t, x]
        nz = np.where(row > 0)[0]
        if nz.size:
            y = int(nz[0])
            pts_top.append((x, y))
    # BOTTOM: for each col, first wall pixel from y=h-1↑
    pts_bot = []
    for x in range(0, w, max(1, sample_step)):
        row = binw[min_y_b:h, x]
        nz = np.where(row > 0)[0]
        if nz.size:
            y = int(min_y_b + nz[-1])
            pts_bot.append((x, y))

    def as_float32(pts):
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    if min(len(pts_left), len(pts_right), len(pts_top), len(pts_bot)) < 20:
        return _err("Not enough side points; try increasing edge_frac or check lighting.")

    # --- 3) Fit one line per side (robust to gaps) ---
    def fit_line(pts):
        pts32 = as_float32(pts)
        vx, vy, x0, y0 = cv2.fitLine(pts32, cv2.DIST_L2, 0, 0.01, 0.01)
        v = np.array([float(vx), float(vy)])
        p = np.array([float(x0), float(y0)])
        return p, v

    pL, vL = fit_line(pts_left)
    pR, vR = fit_line(pts_right)
    pT, vT = fit_line(pts_top)
    pB, vB = fit_line(pts_bot)

    # Orientation sanity (optional)
    def angle_deg(v): return float(np.degrees(np.arctan2(v[1], v[0])))
    aL, aR, aT, aB = angle_deg(vL), angle_deg(vR), angle_deg(vT), angle_deg(vB)
    # Vertical ~ 0° or 180°, Horizontal ~ ±90°
    if min(abs(aL), abs(180 - abs(aL))) > (90 - angle_prior_deg):
        pass  # it's still okay; lines can be slightly slanted
    # same for others if you want strict checks

    # --- 4) Intersections between (L,R) and (T,B) ---
    def intersect(p1, v1, p2, v2):
        A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]], dtype=np.float64)
        b = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float64)
        try:
            t1, _t2 = np.linalg.solve(A, b)
            P = p1 + t1 * v1
            return float(P[0]), float(P[1])
        except np.linalg.LinAlgError:
            return None

    TL = intersect(pL, vL, pT, vT)
    TR = intersect(pR, vR, pT, vT)
    BR = intersect(pR, vR, pB, vB)
    BL = intersect(pL, vL, pB, vB)
    if any(P is None for P in (TL, TR, BR, BL)):
        return _err("Line intersection failed (degenerate configuration).")

    corners = np.array([TL, TR, BR, BL], dtype=np.float32)

    # (Optional) snap to nearest strong wall pixel in a small window
    def snap_to_wall(pt, win=7):
        x0, y0 = int(round(pt[0])), int(round(pt[1]))
        x1 = max(0, x0 - win); x2 = min(w, x0 + win + 1)
        y1 = max(0, y0 - win); y2 = min(h, y0 + win + 1)
        patch = walls[y1:y2, x1:x2]
        ys, xs = np.where(patch > 0)
        if xs.size:
            cx = x1 + int(np.round(xs.mean()))
            cy = y1 + int(np.round(ys.mean()))
            return float(cx), float(cy)
        return float(x0), float(y0)

    corners = np.array([snap_to_wall(c) for c in corners], dtype=np.float32)

    # --- 5) Save results + overlay ---
    _ensure_dir(out_npy); _ensure_dir(overlay_path)
    np.save(out_npy, corners)

    dbg = img.copy()

    def draw_inf_line(p, v, color, thickness=2):
        # extend line across image for visualization
        t = 2000.0
        p1 = (int(p[0] - t*v[0]), int(p[1] - t*v[1]))
        p2 = (int(p[0] + t*v[0]), int(p[1] + t*v[1]))
        cv2.line(dbg, p1, p2, color, thickness)

    draw_inf_line(pL, vL, (0,255,255))  # yellow verticals
    draw_inf_line(pR, vR, (0,255,255))
    draw_inf_line(pT, vT, (255,255,0))  # cyan horizontals
    draw_inf_line(pB, vB, (255,255,0))

    labels = ["TL","TR","BR","BL"]
    for i, (x, y) in enumerate(corners.astype(int)):
        cv2.circle(dbg, (x, y), 8, (0,0,255), -1)
        cv2.putText(dbg, labels[i], (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imwrite(overlay_path, dbg)
    return _ok(ref_corners_path=out_npy, overlay_path=overlay_path, corners_xy=corners.tolist())

def load_correspondences(
    file_path: Annotated[str, "Text file with lines: u_px v_px Xr Yr (>=4 rows)."]
) -> Dict:
    """Load pixel↔robot correspondences from text file."""
    if not os.path.exists(file_path):
        return {"status":"error","reason":f"File not found: {file_path}"}
    arr=[]
    with open(file_path,"r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts=line.split()
            if len(parts)<4: continue
            u,v,Xr,Yr = map(float, parts[:4])
            arr.append((u,v,Xr,Yr))
    if len(arr)<4:
        return {"status":"error","reason":"Need at least 4 correspondences (u v Xr Yr)"}
    A = np.array(arr, dtype=np.float32)
    return {"status":"ok","pix_uv":A[:,0:2].tolist(),"rob_xy":A[:,2:4].tolist(),"n":int(len(A))}

def fit_homography_pixels_to_robot(
    pix_uv: Annotated[List[List[float]], "List of [u,v] pixels."],
    rob_xy: Annotated[List[List[float]], "List of [Xr,Yr] robot coords (mm or meters)."],
    out_npy: Annotated[str, "Save path for H_ref.npy"] = "ref/H_ref.npy"
) -> Dict:
    """Compute H_ref (pixels->robot) using RANSAC; return stats."""
    P = np.array(pix_uv, dtype=np.float32)
    R = np.array(rob_xy, dtype=np.float32)
    H, mask = cv2.findHomography(P, R, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H is None:
        return {"status":"error","reason":"findHomography failed"}
    proj = cv2.perspectiveTransform(P.reshape(1,-1,2), H)[0]
    inliers = (mask.ravel()>0)
    diffs = proj[inliers] - R[inliers]
    dists = np.linalg.norm(diffs, axis=1) if len(diffs) else np.array([])
    stats = {
        "n_points": int(len(P)),
        "n_inliers": int(inliers.sum()),
        "rms": float(np.sqrt(np.mean(dists**2))) if dists.size else float("nan"),
        "median": float(np.median(dists)) if dists.size else float("nan")
    }
    _ensure_dir(out_npy)
    np.save(out_npy, H)
    return {"status":"ok","H_ref_path": out_npy, "stats": stats}

def verify_homography_on_reference(
    ref_image_path: Annotated[str, "Same frame used for reference."],
    H_ref_path: Annotated[str, "np.save path with 3x3 homography (pixels->robot)."],
    ref_corners_path: Annotated[str, "npy with TL,TR,BR,BL (x,y) in pixels."],
    overlay_path: Annotated[str, "Save a verification overlay PNG."] = "ref/hverify_overlay.png",
    grid_step_mm: Annotated[float, "Optional: draw a robot-frame grid projected back to image." ] = 50.0
) -> Dict:
    """Basic visual QA: draw corners + a coarse robot grid projected back to the reference image."""
    img = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"status":"error","reason":f"Cannot read {ref_image_path}"}
    H = np.load(H_ref_path)
    corners = np.load(ref_corners_path).astype(np.float32)

    dbg = img.copy()
    # Draw corners
    for i,p in enumerate(corners.astype(int)):
        cv2.circle(dbg, tuple(p), 8, (0,0,255), -1)

    # Project a coarse robot grid back into image pixels (inverse mapping)
    try:
        H_inv = np.linalg.inv(H)
        # Construct a small grid around the estimated corner bounding box in robot space:
        # First map the 4 image corners to robot space to get extents
        ref_pts = corners.reshape(1,-1,2)
        rob_pts = cv2.perspectiveTransform(ref_pts, H)[0]  # (4,2)
        minX,maxX = float(rob_pts[:,0].min()), float(rob_pts[:,0].max())
        minY,maxY = float(rob_pts[:,1].min()), float(rob_pts[:,1].max())
        xs = np.arange(minX, maxX+1e-6, grid_step_mm)
        ys = np.arange(minY, maxY+1e-6, grid_step_mm)

        # Vertical lines
        for X in xs:
            line_mm = np.array([[X,minY],[X,maxY]], dtype=np.float32).reshape(1,-1,2)
            line_px = cv2.perspectiveTransform(line_mm, H_inv)[0]
            p1 = tuple(np.round(line_px[0]).astype(int)); p2 = tuple(np.round(line_px[1]).astype(int))
            cv2.line(dbg, p1, p2, (0,255,0), 1)
        # Horizontal lines
        for Y in ys:
            line_mm = np.array([[minX,Y],[maxX,Y]], dtype=np.float32).reshape(1,-1,2)
            line_px = cv2.perspectiveTransform(line_mm, H_inv)[0]
            p1 = tuple(np.round(line_px[0]).astype(int)); p2 = tuple(np.round(line_px[1]).astype(int))
            cv2.line(dbg, p1, p2, (0,255,0), 1)
    except Exception as e:
        # Grid is optional—still save corners overlay
        pass

    _ensure_dir(overlay_path)
    cv2.imwrite(overlay_path, dbg)
    return {"status":"ok","verification_overlay": overlay_path}

# --- convenience: single shot from file instead of camera ---

def use_existing_reference_image(
    src_image_path: Annotated[str, "Already-captured reference image to copy/use."],
    dst_image_path: Annotated[str, "Where to save/use it (ref/maze_ref.jpg)."]
) -> Dict:
    img = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"status":"error","reason":f"Cannot read {src_image_path}"}
    _ensure_dir(dst_image_path)
    cv2.imwrite(dst_image_path, img)
    return {"status":"ok","image_path": dst_image_path}
