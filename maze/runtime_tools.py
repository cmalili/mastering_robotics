#!/usr/bin/env python3
"""
runtime_tools.py
Deterministic tools for the runtime pipeline (no LLM logic here).

Steps (each is exposed as a callable tool):
  1) capture_current_image(...) -> {"image_path": ...}
  2) detect_current_corners(...) -> {"cur_corners_path", "overlay_path", "corners_xy":[[x,y],...]}
  3) compose_current_homography(...) -> {"H_cur_path", "G_ref_from_cur": [[...]]}
  4) solve_maze_current(...) -> {"solver_json", "overlay_path", "path_original": [...]}
  5) map_path_to_robot(...) -> {"robot_csv", "stats": {...}}
  6) preview_overlay(...) -> {"preview_path"}
  7) robot_execute(...) -> {"status": "ok"|"error", "reason": ...}

Conventions:
- Pixels as (x=u, y=v), but solver uses (y,x) lists. Pay attention to conversions.
- Corners order: TL, TR, BR, BL (float32, image pixels).
- Homographies:
    H_ref:  pixels(ref) -> robot (XY in mm or meters)
    G_ref<-cur: current pixels -> reference pixels (image-to-image)
    H_cur = H_ref @ G_ref<-cur : current pixels -> robot

Safe defaults:
- robot_execute(...) runs in dry-run mode unless dryrun=False and you provide your own executor.

Dependencies:
  - centerline_pixel.py (must be importable)
"""

from __future__ import annotations
from typing import Annotated, Dict, List, Tuple, Optional
import os, json
import numpy as np
import cv2

# ----------------- shared helpers -----------------

def _ensure_dir(p: str):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def _ok(**kwargs): 
    out = {"status":"ok"}; out.update(kwargs); return out

def _err(reason: str, **kwargs): 
    out = {"status":"error","reason":str(reason)}; out.update(kwargs); return out

# Corner helpers (same logic as in calibration tools)
def _order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)         # x+y
    d = np.diff(pts, axis=1)    # x - y
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

# ----------------- 1) Capture -----------------

def capture_current_image(
    out_path: Annotated[str, "Destination for current frame (e.g., runs/.../cur/frame.jpg)."],
    camera_index: Annotated[int, "OpenCV camera index."] = 0,
    width: Annotated[int, "Capture width."] = 1920,
    height: Annotated[int, "Capture height."] = 1080,
    warmup_frames: Annotated[int, "Frames to skip to stabilize exposure."] = 8
) -> Dict:
    _ensure_dir(out_path)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return _err(f"Cannot open camera index {camera_index}")
    if width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(max(1, warmup_frames)):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return _err("Failed to read frame")
    cv2.imwrite(out_path, frame)
    return _ok(image_path=out_path)

def use_existing_current_image(
    src_image_path: Annotated[str, "Use an already captured current image."],
    dst_image_path: Annotated[str, "Where to place it under runs/.../cur/frame.jpg."]
) -> Dict:
    img = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    if img is None: return _err(f"Cannot read {src_image_path}")
    _ensure_dir(dst_image_path)
    cv2.imwrite(dst_image_path, img)
    return _ok(image_path=dst_image_path)

# ----------------- 2) Detect current corners -----------------

def detect_current_corners(
    image_path: Annotated[str, "Current frame path."],
    out_npy: Annotated[str, "Save corners (TL,TR,BR,BL) npy path."] = "cur/cur_corners.npy",
    overlay_path: Annotated[str, "Debug overlay path."] = "cur/corners_overlay.png"
) -> Dict:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: return _err(f"Cannot read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu & polarity selection
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    pos = (th==255).astype(np.uint8); neg = 1-pos
    def interior_mass(m):
        h,w=m.shape; b=max(2, min(h,w)//100); return m[b:h-b,b:w-b].sum()
    corridors = pos if interior_mass(pos) >= interior_mass(neg) else neg
    walls = (1-corridors).astype(np.uint8) * 255
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)), 1)
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)

    cnts, _ = cv2.findContours(walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return _err("No outer contour found")
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02*peri, True)
    if len(approx) != 4:
        approx = cv2.approxPolyDP(hull, 0.05*peri, True)
    if len(approx) == 4: pts = approx.reshape(-1,2).astype(np.float32)
    else: pts = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)

    corners = _order_tl_tr_br_bl(pts)
    corners = _refine_corners(gray, corners, win=7)

    _ensure_dir(out_npy); _ensure_dir(overlay_path)
    np.save(out_npy, corners)

    # draw overlay
    dbg = img.copy(); labels = ["TL","TR","BR","BL"]
    for i,p in enumerate(corners.astype(int)):
        cv2.circle(dbg, tuple(p), 8, (0,0,255), -1)
        cv2.putText(dbg, labels[i], tuple(p+np.array([6,-6])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.imwrite(overlay_path, dbg)
    return _ok(cur_corners_path=out_npy, overlay_path=overlay_path, corners_xy=corners.tolist())

# ----------------- 3) Compose H_cur -----------------

def compose_current_homography(
    H_ref_path: Annotated[str, "np.save path of H_ref (pixels(ref)->robot)."],
    ref_corners_path: Annotated[str, "npy with TL,TR,BR,BL corners in REF image."],
    cur_corners_path: Annotated[str, "npy with TL,TR,BR,BL corners in CURRENT image."],
    out_npy: Annotated[str, "Where to save H_cur (pixels(current)->robot)."] = "cur/H_cur.npy"
) -> Dict:
    if not (os.path.exists(H_ref_path) and os.path.exists(ref_corners_path) and os.path.exists(cur_corners_path)):
        return _err("Missing input artifacts for homography composition")
    H_ref = np.load(H_ref_path)
    ref_c = np.load(ref_corners_path).astype(np.float32)
    cur_c = np.load(cur_corners_path).astype(np.float32)
    G_ref_from_cur = cv2.getPerspectiveTransform(cur_c, ref_c)  # current -> reference
    H_cur = H_ref @ G_ref_from_cur
    _ensure_dir(out_npy)
    np.save(out_npy, H_cur)
    return _ok(H_cur_path=out_npy, G_ref_from_cur=G_ref_from_cur.tolist())

# ----------------- 4) Solve on current image -----------------

def solve_maze_current(
    image_path: Annotated[str, "Current frame path to solve."],
    save_json: Annotated[str, "Where to save solver JSON."] = "solver/result.json",
    mode: Annotated[str, "widest|weighted|shortest"] = "widest",
    min_clearance: Annotated[Optional[int], "For shortest mode: minimum clearance in ORIGINAL px."] = None,
    lam: Annotated[float, "Weighted mode lambda."] = 6.0,
    eps: Annotated[float, "Weighted mode epsilon."] = 1.0,
    maxdim: Annotated[int, "Downscale long side for speed."] = 1200,
    blur: Annotated[int, "Gaussian blur kernel (odd), 0 disables."] = 5,
    wall_open: Annotated[int, "Morph OPEN on walls."] = 3,
    wall_close: Annotated[int, "Morph CLOSE on walls."] = 9,
) -> Dict:
    # Import your solver module
    try:
        from d import solve_centerline
    except Exception as e:
        return _err(f"Could not import centerline_pixel.solve_centerline: {e}")

    from types import SimpleNamespace
    args = SimpleNamespace(
        mode=mode,
        min_clearance=min_clearance,
        clearance_ref="original",
        min_clearance_mm=None,
        maze_width_mm=None,
        lam=lam,
        eps=eps,
        maxdim=maxdim,
        blur=blur,
        open=wall_open,
        close=wall_close,
        prefer_lr=False,
        include_gate=False,
        save_csv=None,
        save_json=save_json
    )
    res = solve_centerline(image_path, args)
    _ensure_dir(save_json)
    with open(save_json, "w") as f:
        json.dump(res, f, indent=2)

    if res.get("status") != "ok":
        return _err(res.get("reason","solver error"), solver_json=save_json)

    return _ok(
        solver_json=save_json,
        overlay_path=os.path.join(os.path.dirname(save_json), res["solution_image"]) if "solution_image" in res else None,
        path_original=res.get("path_original", [])
    )

# ----------------- 5) Map path to robot -----------------

def _resample_by_arclength(points: List[Tuple[float,float]], spacing: float) -> List[Tuple[float,float]]:
    if len(points)<2 or spacing<=0: return points
    P = np.array(points, dtype=np.float64)
    d = np.sqrt(((P[1:] - P[:-1])**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    if L <= spacing: return points
    n = int(np.floor(L/spacing))
    targets = np.linspace(0, L, n+1)
    out = []
    j = 0
    for t in targets:
        while j < len(s)-2 and s[j+1] < t:
            j += 1
        seg = s[j+1]-s[j]
        a = 0.0 if seg==0 else (t - s[j]) / seg
        out.append(tuple((1-a)*P[j] + a*P[j+1]))
    return out

def map_path_to_robot(
    solver_json: Annotated[str, "Path to solver JSON containing path_original (in current image pixels)."],
    H_cur_path: Annotated[str, "np.save path to H_cur (current px -> robot)."],
    z_const: Annotated[float, "Constant Z for robot waypoints (same unit as robot XY)."] = 0.0,
    spacing: Annotated[Optional[float], "Resample spacing in robot units, e.g., 2.0 mm."] = 2.0,
    out_csv: Annotated[str, "Where to save robot waypoints CSV."] = "robot/path.csv"
) -> Dict:
    if not (os.path.exists(solver_json) and os.path.exists(H_cur_path)):
        return _err("Missing inputs for mapping")

    data = json.load(open(solver_json,"r"))
    if data.get("status")!="ok" or "path_original" not in data:
        return _err("solver_json missing status=ok or path_original")
    path_px = [(int(y), int(x)) for (y,x) in data["path_original"]]  # (y,x)

    H_cur = np.load(H_cur_path)

    # Map pixels (x=u,y=v) -> robot (X,Y)
    P = np.array([[x, y] for (y,x) in path_px], dtype=np.float32).reshape(1,-1,2)
    Q = cv2.perspectiveTransform(P, H_cur)[0]  # (N,2)
    pts = [(float(X), float(Y)) for (X, Y) in Q]

    if spacing and spacing>0:
        pts = _resample_by_arclength(pts, spacing=spacing)

    # attach Z
    pts_xyz = [(X, Y, float(z_const)) for (X,Y) in pts]

    _ensure_dir(out_csv)
    with open(out_csv, "w") as f:
        f.write("X,Y,Z\n")
        for X,Y,Z in pts_xyz:
            f.write(f"{X:.3f},{Y:.3f},{Z:.3f}\n")

    stats = {
        "n_points": len(pts_xyz),
        "total_length": float(sum(np.linalg.norm(np.array(pts[i+1][:2]) - np.array(pts[i][:2])) for i in range(len(pts)-1))) if len(pts)>1 else 0.0
    }
    return _ok(robot_csv=out_csv, stats=stats)

# ----------------- 6) Preview overlay on current image -----------------

def preview_overlay(
    image_path: Annotated[str, "Current frame path."],
    solver_json: Annotated[str, "Solver JSON with path_original in current image pixels."],
    H_cur_path: Annotated[str, "For optional back-projection check (not required)."],
    out_path: Annotated[str, "Where to save preview PNG."] = "cur/preview_overlay.png",
    draw_backproject: Annotated[bool, "Also draw mapped robot path back to pixels."] = False
) -> Dict:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: return _err(f"Cannot read {image_path}")
    data = json.load(open(solver_json,"r"))
    if data.get("status")!="ok" or "path_original" not in data:
        return _err("solver_json missing status=ok or path_original")
    path_px = [(int(y), int(x)) for (y,x) in data["path_original"]]

    # draw path_original (red)
    for (y,x) in path_px:
        cv2.circle(img, (x,y), 1, (0,0,255), -1)

    if draw_backproject and os.path.exists(H_cur_path):
        # Map to robot and back to pixels (sanity): H_inv
        try:
            H = np.load(H_cur_path)
            H_inv = np.linalg.inv(H)
            # build small subset for visibility
            step = max(1, len(path_px)//500)
            P = np.array([[x, y] for (y,x) in path_px[::step]], dtype=np.float32).reshape(1,-1,2)
            Q = cv2.perspectiveTransform(P, H)[0]
            P2 = cv2.perspectiveTransform(Q.reshape(1,-1,2), H_inv)[0]
            for (x,y) in P2.astype(int):
                cv2.circle(img, (int(x),int(y)), 1, (0,255,0), -1)
        except Exception:
            pass

    _ensure_dir(out_path)
    cv2.imwrite(out_path, img)
    return _ok(preview_path=out_path)

# ----------------- 7) Robot execute (safe stub) -----------------

def robot_execute(
    robot_csv: Annotated[str, "CSV with X,Y,Z in robot units."],
    dryrun: Annotated[bool, "If True, do not contact robot; just return success."] = True,
    driver: Annotated[Optional[str], "Optional driver name, e.g., 'dobot-sdk'."] = None,
    speed: Annotated[float, "Nominal speed (driver-specific)."] = 50.0
) -> Dict:
    if not os.path.exists(robot_csv):
        return _err(f"robot_csv not found: {robot_csv}")
    if dryrun:
        return _ok(message=f"Dry-run: would execute {robot_csv} at speed {speed}")
    # Implement your actual robot call here (SDK, socket, etc.)
    # Example (pseudo):
    #   if driver=="dobot-sdk": dobot_run(robot_csv, speed)
    #   else: return _err("Unsupported driver")
    return _err("live robot execution not implemented in this stub; use dryrun=True or add your driver")
