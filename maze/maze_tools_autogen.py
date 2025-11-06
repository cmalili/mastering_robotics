#!/usr/bin/env python3
# maze_tools_autogen.py
"""
Deterministic tools for the AutoGen agents:
- start/stop live video with PiP overlay
- capture or copy current image
- (QA) detect outer corners (max-area convex quad) and overlay
- solve maze (weighted Dijkstra) and return ACTUAL pixel path
- map pixel path to robot coords using precomputed H
- execute robot motion (stub; ready for pydobotplus)

Place this file next to `runtime_swarm_weighted.py` and your `d.py`.
"""

from __future__ import annotations
from typing import Annotated, Dict, Optional
import os, json, time, threading
from pathlib import Path

import numpy as np
import cv2
import pydobotplus as pdp

# ---- bring in your helpers --------------------------------------------------
import d  # your module with Dijkstra helpers

# ---- small utils ------------------------------------------------------------

def _ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def _write_json(obj: dict, path: str):
    _ensure_dir(path)
    # convert numpy types for safety
    def _to_py(o):
        if isinstance(o, dict): return {k:_to_py(v) for k,v in o.items()}
        if isinstance(o, (list, tuple)): return [_to_py(x) for x in o]
        if isinstance(o, (np.generic,)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(path, "w") as f:
        json.dump(_to_py(obj), f, indent=2)

def _apply_homography_pts(H: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    pts = pts_uv.astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    UV1 = np.hstack([pts, ones])  # Nx3
    XYW = (H @ UV1.T).T
    w = XYW[:, 2:3]
    w[w == 0] = 1e-9
    XY = XYW[:, :2] / w
    return XY.astype(np.float32)

def _resample_polyline_xy(poly: np.ndarray, spacing: Optional[float]) -> np.ndarray:
    if poly is None or len(poly) == 0 or spacing is None or spacing <= 0:
        return poly.copy()
    seg = poly[1:] - poly[:-1]
    dists = np.sqrt((seg**2).sum(axis=1))
    if dists.size == 0:
        return poly.copy()
    cum = np.concatenate([[0.0], np.cumsum(dists)])
    total = float(cum[-1])
    if total <= 1e-6:
        return poly[[0]].copy()
    n = max(1, int(total // spacing))
    ts = np.linspace(0.0, total, num=n+1, endpoint=True)
    out = []
    j = 0
    for t in ts:
        while (j+1) < len(cum) and cum[j+1] < t:
            j += 1
        if (j+1) >= len(cum):
            out.append(poly[-1])
            break
        denom = (cum[j+1] - cum[j]) or 1e-9
        alpha = (t - cum[j]) / denom
        p = (1.0 - alpha) * poly[j] + alpha * poly[j+1]
        out.append(p)
    return np.array(out, dtype=np.float32)

# ---- video feed with PiP overlay -------------------------------------------

_VIDEO_THREADS: dict[str, "VideoThread"] = {}

class VideoThread(threading.Thread):
    def __init__(self, camera_index: int, window_name: str, overlay_path: str, fps: float = 15.0):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.window_name = window_name
        self.overlay_path = overlay_path
        self.fps = fps
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._last_frame = None

    def stop(self):
        self._stop.set()

    def run(self):
        if self.camera_index is None or self.camera_index < 0:
            return
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return
        last_overlay_mtime = None
        overlay_img = None
        delay_ms = max(1, int(1000 / max(1e-3, self.fps)))
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._last_frame =  frame.copy()
            # lazy-load overlay when file changes
            try:
                if os.path.exists(self.overlay_path):
                    mtime = os.path.getmtime(self.overlay_path)
                    if last_overlay_mtime != mtime:
                        overlay_img = cv2.imread(self.overlay_path, cv2.IMREAD_COLOR)
                        last_overlay_mtime = mtime
            except Exception:
                overlay_img = None

            if overlay_img is not None:
                # draw PiP top-right
                H, W = frame.shape[:2]
                ph = int(0.3 * H)
                pw = max(1, int(overlay_img.shape[1] * (ph / overlay_img.shape[0])))
                pip = cv2.resize(overlay_img, (pw, ph), interpolation=cv2.INTER_AREA)
                x1 = W - 12 - pw
                y1 = 12
                frame[y1:y1+ph, x1:x1+pw] = pip

            cv2.putText(frame, self.window_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow(self.window_name, frame)
            k = cv2.waitKey(delay_ms) & 0xFF
            if k == ord('q'):
                break
        cap.release()
        #try:
        #    cv2.destroyWindow(self.window_name)
        #except Exception:
        #    pass

def start_video_pip(
    camera_index: Annotated[int, "OpenCV camera index; use -1 to skip if no camera."],
    overlay_path: Annotated[str, "Path to solver overlay PNG shown as PiP."],
    window_name: Annotated[str, "Window title."] = "Maze Runtime",
    fps: Annotated[float, "Display FPS."] = 15.0
) -> Dict:
    """Start a background video thread with PiP overlay."""
    if camera_index is None or camera_index < 0:
        return {"status":"skipped","reason":"camera_index < 0"}
    # stop existing with same name
    if window_name in _VIDEO_THREADS:
        try:
            _VIDEO_THREADS[window_name].stop()
        except Exception:
            pass
    th = VideoThread(camera_index=camera_index, window_name=window_name, overlay_path=overlay_path, fps=fps)
    _VIDEO_THREADS[window_name] = th
    th.start()
    return {"status":"ok","window":window_name}

def stop_video_pip(
    window_name: Annotated[str, "Window title to close."] = "Maze Runtime"
) -> Dict:
    """Stop the background video thread."""
    th = _VIDEO_THREADS.pop(window_name, None)
    if th is None:
        return {"status":"skipped","reason":"no such window"}
    try:
        th.stop()
        th.join(timeout=1.0)           # ensure thread is finished
    except Exception as e:
        return {"status":"error","reason":f"thread stop failed: {e}"}
    try:
        cv2.destroyWindow(window_name) # main-thread teardown (no Qt warnings)
    except Exception:
        pass
    return {"status":"ok","window":window_name}

# ---- capture / copy current image ------------------------------------------

def use_existing_current_image(
    src_image_path: Annotated[str, "Existing current image to use."],
    dst_image_path: Annotated[str, "Where to store the run image (cur/frame.jpg)."]
) -> Dict:
    img = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"status":"error","reason":f"Cannot read {src_image_path}"}
    _ensure_dir(dst_image_path)
    cv2.imwrite(dst_image_path, img)
    return {"status":"ok","image_path": dst_image_path}

def capture_current_image(
    out_path: Annotated[str, "Where to save cur frame (cur/frame.jpg)."],
    camera_index: Annotated[int, "OpenCV camera index."] = 0,
    width: Annotated[int, "Width."] = 1920,
    height: Annotated[int, "Height."] = 1080,
    warmup_frames: Annotated[int, "Warmup frames to stabilize exposure."] = 10
) -> Dict:
    _ensure_dir(out_path)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"status":"error","reason":f"Cannot open camera {camera_index}"}
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(max(1, warmup_frames)):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"status":"error","reason":"Failed to read a frame"}
    cv2.imwrite(out_path, frame)
    return {"status":"ok","image_path": out_path}

# ---- corners QA (optional) --------------------------------------------------

def detect_corners_qa(
    image_path: Annotated[str, "Path to image to analyze."],
    corners_json_out: Annotated[str, "Write TL/TR/BR/BL in pixels here (JSON)."],
    overlay_out: Annotated[str, "Write overlay PNG here."] = ""
) -> Dict:
    """
    Optional QA: find four outer corners from detected points (max-area convex quad).
    Uses your outer_corners_maxarea module if available; otherwise skipped.
    """
    try:
        from outer_corners import _convex_hull, _order_tl_tr_br_bl, _max_area_quad_from_hull, shi_tomasi
    except Exception as e:
        return {"status":"skipped","reason":f"outer_corners_maxarea not available: {e}"}

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"status":"error","reason":f"Cannot read {image_path}"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)

    pts = shi_tomasi(gray_blur, max_corners=1500, quality=0.01, min_dist=500, block_size=3, subpix=True)
    if pts.shape[0] < 4:
        return {"status":"error","reason":"<4 detected points"}

    pts = np.unique(np.round(pts, 1), axis=0).astype(np.float32)
    Hh = _convex_hull(pts)
    if Hh.shape[0] < 4:
        return {"status":"error","reason":"Convex hull <4 vertices"}
    idxs = _max_area_quad_from_hull(Hh)
    quad = np.vstack([Hh[idxs[0]], Hh[idxs[1]], Hh[idxs[2]], Hh[idxs[3]]]).astype(np.float32)
    ordered = _order_tl_tr_br_bl(quad)

    # overlay
    if overlay_out:
        _ensure_dir(overlay_out)
        vis = img.copy()
        for (x,y) in pts.astype(int):
            cv2.circle(vis, (x,y), 1, (0,255,0), -1, cv2.LINE_AA)
        Q = quad.astype(int)
        for a,b in zip(Q, np.roll(Q, -1, axis=0)):
            cv2.line(vis, tuple(a), tuple(b), (0,255,255), 2, cv2.LINE_AA)
        for lab,(x,y) in zip(["TL","TR","BR","BL"], ordered.astype(int)):
            cv2.circle(vis, (x,y), 7, (0,0,255), -1)
            cv2.putText(vis, lab, (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imwrite(overlay_out, vis)

    _ensure_dir(corners_json_out)
    with open(corners_json_out, "w") as f:
        json.dump({"TL":ordered[0].tolist(),"TR":ordered[1].tolist(),"BR":ordered[2].tolist(),"BL":ordered[3].tolist()}, f, indent=2)

    return {"status":"ok","corners_json": corners_json_out, "overlay": overlay_out}






# ---- weighted Dijkstra solver (returns ACTUAL pixel path) -------------------

def solve_maze_weighted_previous(
    image_path: Annotated[str, "Path to current image to solve."],
    out_json: Annotated[str, "Where to save solver JSON (includes path_px)."],
    out_overlay: Annotated[str, "Where to save solver overlay PNG."],
    lam: Annotated[float, "lambda for weighted Dijkstra (higher favors center)."] = 6.0,
    eps: Annotated[float, "epsilon for weighted Dijkstra."] = 1.0,
    maxdim: Annotated[int, "Downscale long side to speed up (px)."] = 1200,
    blur: Annotated[int, "Gaussian blur kernel size."] = 5,
    wall_open: Annotated[int, "Morph open iterations for walls."] = 3,
    wall_close: Annotated[int, "Morph close iterations for walls."] = 9,
    prefer_lr: Annotated[bool, "Prefer left/right openings instead of top/bottom."] = False
) -> Dict:
    # Load + downscale
    gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        return {"status":"error","reason":f"Cannot read {image_path}"}
    gray, scale = d.resize_keep_aspect(gray0, maxdim)
    H0, W0 = gray0.shape[:2]

    # Binarize + crop to ROI
    corr, walls = d.binarize_and_clean(gray, blur, wall_open, wall_close)
    gray_roi, corr_roi, (y0,y1,x0,x1) = d.crop_to_walls(gray, walls, pad=3)
    walls_roi = 1 - corr_roi
    H, W = gray_roi.shape

    # Detect openings & seeds
    try:
        (s1,c1), (s2,c2), _ = d.detect_openings(walls_roi, prefer_top_bottom=not prefer_lr)
    except Exception as e:
        return {"status":"error","reason":f"Opening detection failed: {e}"}

    band = max(20, min(H, W)//25)
    Ty = int(np.argmax(walls_roi[:band, :].sum(axis=1)))
    By = H - band + int(np.argmax(walls_roi[H-band:H, :].sum(axis=1)))
    Lx = int(np.argmax(walls_roi[:, :band].sum(axis=0)))
    Rx = W - band + int(np.argmax(walls_roi[:, W-band:W].sum(axis=0)))

    def _seed(side, center):
        if side == "top":    return (Ty+1, center)
        if side == "bottom": return (By-1, center)
        if side == "left":   return (center, Lx+1)
        if side == "right":  return (center, Rx-1)
        raise ValueError(side)

    start = _seed(s1, c1); goal = _seed(s2, c2)

    # EDT on corridor
    corr_dt = corr_roi.copy()
    corr_dt[0,:]=0; corr_dt[-1,:]=0; corr_dt[:,0]=0; corr_dt[:,-1]=0
    dist = cv2.distanceTransform((corr_dt*255).astype(np.uint8), cv2.DIST_L2, 3)

    # Weighted Dijkstra
    path_rc, achieved_min, achieved_avg = d.dijkstra_weighted(dist, start, goal, lam=float(lam), eps=float(eps))
    if not path_rc:
        return {"status":"error","reason":"No path through corridor (check binarization)."}

    used_T = int(np.floor(achieved_min))
    safe = (dist >= max(1, used_T)).astype(np.uint8)
    safe[0,:]=0; safe[-1,:]=0; safe[:,0]=0; safe[:,-1]=0

    # Convert ROI path (r,c) -> original pixel path (u,v)
    path_rc = np.array(path_rc, dtype=np.float32)            # ROI y,x (downscaled)
    path_rc_full = path_rc + np.array([y0, x0], dtype=np.float32)  # to full-downscaled
    path_uv_orig = np.empty_like(path_rc_full)
    path_uv_orig[:,0] = path_rc_full[:,1] / max(scale, 1e-9)  # u=x/scale
    path_uv_orig[:,1] = path_rc_full[:,0] / max(scale, 1e-9)  # v=y/scale

    # Overlay at original size
    base_small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # draw_safe_and_path_on_canvas expects ROI path (not shifted); pass path_rc here:
    out_small = d.draw_safe_and_path_on_canvas(base_small.copy(), safe, path_rc, (y0,y1,x0,x1))
    overlay = cv2.resize(out_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
    _ensure_dir(out_overlay)
    cv2.imwrite(out_overlay, overlay)

    # Report entrance/exit location in original pixels
    if s1 in ("top","bottom"):
        entrance_pos_px = int(round((x0 + c1) / max(scale,1e-9)))
    else:
        entrance_pos_px = int(round((y0 + c1) / max(scale,1e-9)))
    if s2 in ("top","bottom"):
        exit_pos_px = int(round((x0 + c2) / max(scale,1e-9)))
    else:
        exit_pos_px = int(round((y0 + c2) / max(scale,1e-9)))

    result = {
        "status":"ok",
        "mode":"weighted",
        "entrance":{"side":s1, "pos_px": entrance_pos_px},
        "exit":{"side":s2, "pos_px": exit_pos_px},
        "solution_overlay": out_overlay,
        "scale_down": float(scale),
        "threshold_used_downscaled_px": int(used_T),
        "threshold_equiv_original_px": float(used_T / max(scale,1e-9)),
        "path_length_nodes": int(len(path_rc)),
        "achieved_min_clearance_downscaled_px": float(achieved_min),
        "achieved_min_clearance_equiv_original_px": float(achieved_min / max(scale,1e-9)),
        "weighted_lambda": float(lam),
        "weighted_eps": float(eps),
        "path_px": path_uv_orig.tolist()
    }
    _ensure_dir(out_json)
    _write_json(result, out_json)
    return result

# ---- map to robot & save CSV -----------------------------------------------

def map_path_to_robot(
    solver_json: Annotated[str, "Path to solver JSON (must contain 'path_px')."],
    H_path: Annotated[str, "Precomputed 3x3 homography npy (pixels->robot)."],
    out_csv: Annotated[str, "Write robot path CSV here."],
    z_const: Annotated[float, "Constant Z value (robot units)."] = 0.0,
    spacing: Annotated[Optional[float], "Resample spacing in robot units; <=0 disables."] = 2.0
) -> Dict:
    if not os.path.exists(solver_json):
        return {"status":"error","reason":f"Missing solver json: {solver_json}"}
    with open(solver_json, "r") as f:
        sj = json.load(f)
    if sj.get("status") != "ok" or "path_px" not in sj:
        return {"status":"error","reason":"solver json has no path_px"}
    path_px = np.array(sj["path_px"], dtype=np.float32)
    if path_px.ndim != 2 or path_px.shape[1] != 2 or path_px.shape[0] < 2:
        return {"status":"error","reason":"path_px invalid"}

    if not os.path.exists(H_path):
        return {"status":"error","reason":f"Missing homography: {H_path}"}
    H = np.load(H_path).astype(np.float32)

    xy = _apply_homography_pts(H, path_px)  # robot XY
    xy = _resample_polyline_xy(xy, spacing)
    xyz = np.hstack([xy, np.full((xy.shape[0],1), float(z_const), dtype=np.float32)])

    _ensure_dir(out_csv)
    with open(out_csv, "w") as f:
        f.write("X,Y,Z\n")
        for X, Y, Z in xyz:
            f.write(f"{X:.4f},{Y:.4f},{Z:.4f}\n")
    return {"status":"ok","csv": out_csv, "count": int(xyz.shape[0])}

# ---- execute (stub; ready for pydobotplus) ---------------------------------

def robot_execute(
    csv_path: Annotated[str, "Robot CSV with X,Y,Z columns."],
    dryrun: Annotated[bool, "If true, do not contact robot."] = True,
    speed: Annotated[float, "Driver-specific speed."] = 50.0,
    settle_ms: Annotated[int, "Optional settle after motion (ms)."] = 0
) -> Dict:
    if not os.path.exists(csv_path):
        return {"status":"error","reason":f"Missing CSV: {csv_path}"}
    pts = []
    with open(csv_path, "r") as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line: continue
            pts.append(tuple(map(float, line.split(","))))
    if len(pts) < 2:
        return {"status":"error","reason":"Too few waypoints"}

    # TODO: plug your pydobotplus calls here.
    # Example:
    if not dryrun:
         import pydobotplus as pdp
    bot = pdp.Dobot(port="/dev/ttyACM0")

    for (X,Y,Z) in pts:
        bot.move_to(X,Y,Z, 0)

    bot.move_to(215.56, -5.8, 53.62, 0)
    bot.close()



    t0 = time.time()
    time.sleep(20)
    
    if dryrun:
        time.sleep(min(0.5, 0.001 * len(pts)))
    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)
    elapsed = time.time() - t0
    
    return {"status":"ok","dryrun": bool(dryrun), "points": len(pts), "elapsed_s": round(elapsed,3)}






def solve_maze_weighted1(
    image_path: Annotated[str, "Path to current image to solve."],
    out_json: Annotated[str, "Where to save solver JSON (includes path_px)."],
    out_overlay: Annotated[str, "Where to save solver overlay PNG."],
    lam: Annotated[float, "lambda for weighted Dijkstra (higher favors center)."] = 6.0,
    eps: Annotated[float, "epsilon for weighted Dijkstra."] = 1.0,
    maxdim: Annotated[int, "Downscale long side to speed up (px)."] = 1200,
    blur: Annotated[int, "Gaussian blur kernel size."] = 5,
    wall_open: Annotated[int, "Morph open iterations for walls."] = 3,
    wall_close: Annotated[int, "Morph close iterations for walls."] = 9,
    prefer_lr: Annotated[bool, "Prefer left/right openings instead of top/bottom."] = False
) -> Dict:
    # Load + downscale
    gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        return {"status":"error","reason":f"Cannot read {image_path}"}
    gray, scale = d.resize_keep_aspect(gray0, maxdim)
    H0, W0 = gray0.shape[:2]

    # Binarize + crop to ROI
    corr, walls = d.binarize_and_clean(gray, blur, wall_open, wall_close)
    gray_roi, corr_roi, (y0,y1,x0,x1) = d.crop_to_walls(gray, walls, pad=3)
    # ensure integer ROI bounds (some OpenCV ops return np.int64 but be explicit)
    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    walls_roi = 1 - corr_roi
    H, W = gray_roi.shape

    # Detect openings & seeds
    try:
        (s1,c1), (s2,c2), _ = d.detect_openings(walls_roi, prefer_top_bottom=(not bool(prefer_lr)))
    except Exception as e:
        return {"status":"error","reason":f"Opening detection failed: {e}"}

    # centers may be np types; make sure they are ints for indexing
    c1 = int(c1); c2 = int(c2)

    band = max(20, min(H, W)//25)
    Ty = int(np.argmax(walls_roi[:band, :].sum(axis=1)))
    By = H - band + int(np.argmax(walls_roi[H-band:H, :].sum(axis=1)))
    Lx = int(np.argmax(walls_roi[:, :band].sum(axis=0)))
    Rx = W - band + int(np.argmax(walls_roi[:, W-band:W].sum(axis=0)))

    def _seed(side: str, center: int) -> Tuple[int,int]:
        if side == "top":    return (Ty+1, center)
        if side == "bottom": return (By-1, center)
        if side == "left":   return (center, Lx+1)
        if side == "right":  return (center, Rx-1)
        raise ValueError(side)

    start = _seed(s1, c1); goal = _seed(s2, c2)

    # EDT on corridor
    corr_dt = corr_roi.copy()
    corr_dt[0,:]=0; corr_dt[-1,:]=0; corr_dt[:,0]=0; corr_dt[:,-1]=0
    dist = cv2.distanceTransform((corr_dt*255).astype(np.uint8), cv2.DIST_L2, 3)

    # Weighted Dijkstra
    path_rc, achieved_min, achieved_avg = d.dijkstra_weighted(dist, start, goal, lam=float(lam), eps=float(eps))
    if not path_rc:
        return {"status":"error","reason":"No path through corridor (check binarization)."}

    # === IMPORTANT: build two versions of the path ===
    # 1) Integer ROI (r,c) for any array indexing / overlay drawing
    path_rc_int = np.asarray(path_rc, dtype=np.int32)             # (N,2) ints in ROI coords
    # 2) Float for accurate conversion back to original pixel coordinates
    path_rc_float = path_rc_int.astype(np.float32)

    used_T = int(np.floor(achieved_min))
    safe = (dist >= max(1, used_T)).astype(np.uint8)
    safe[0,:]=0; safe[-1,:]=0; safe[:,0]=0; safe[:,-1]=0

    # Convert ROI path -> original pixel path (u,v) in *original image* coordinates
    # shift to full downscaled image coords
    path_rc_full = path_rc_float + np.array([y0, x0], dtype=np.float32)
    # map to original pixels by undoing the scale (u=x/scale, v=y/scale)
    path_uv_orig = np.empty_like(path_rc_full, dtype=np.float32)
    path_uv_orig[:,0] = path_rc_full[:,1] / max(scale, 1e-9)  # u
    path_uv_orig[:,1] = path_rc_full[:,0] / max(scale, 1e-9)  # v

    # Overlay at original size (draw with INT indices!)
    base_small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    try:
        out_small = d.draw_safe_and_path_on_canvas(base_small.copy(), safe, path_rc_int, (y0,y1,x0,x1))
    except Exception as e:
        # If the drawer still fails, return path anyway and report the overlay error
        out_small = base_small
    overlay = cv2.resize(out_small, (W0, H0), interpolation=cv2.INTER_NEAREST)
    _ensure_dir(out_overlay)
    cv2.imwrite(out_overlay, overlay)

    # Entrance/exit positions in original pixels along image border
    if s1 in ("top","bottom"):
        entrance_pos_px = int(round((x0 + c1) / max(scale,1e-9)))
    else:
        entrance_pos_px = int(round((y0 + c1) / max(scale,1e-9)))
    if s2 in ("top","bottom"):
        exit_pos_px = int(round((x0 + c2) / max(scale,1e-9)))
    else:
        exit_pos_px = int(round((y0 + c2) / max(scale,1e-9)))

    result = {
        "status":"ok",
        "mode":"weighted",
        "entrance":{"side":s1, "pos_px": entrance_pos_px},
        "exit":{"side":s2, "pos_px": exit_pos_px},
        "solution_overlay": out_overlay,
        "scale_down": float(scale),
        "threshold_used_downscaled_px": int(used_T),
        "threshold_equiv_original_px": float(used_T / max(scale,1e-9)),
        "path_length_nodes": int(len(path_rc_int)),
        "achieved_min_clearance_downscaled_px": float(achieved_min),
        "achieved_min_clearance_equiv_original_px": float(achieved_min / max(scale,1e-9)),
        "weighted_lambda": float(lam),
        "weighted_eps": float(eps),
        # The ACTUAL solution path in original-image pixels (u,v)
        "path_px": path_uv_orig.tolist()
    }
    _ensure_dir(out_json)
    _write_json(result, out_json)
    return result

def snapshot_from_display(
    window_name: Annotated[str, "Window title used by start_video_pip."] = "Maze Runtime",
    out_path: Annotated[str, "Where to save the snapshot (e.g., cur/frame.jpg)."] = "cur/frame.jpg",
    timeout_s: Annotated[float, "Max seconds to wait for the first frame."] = 2.0
) -> Dict:
    th = _VIDEO_THREADS.get(window_name)
    if th is None or not th.is_alive():
        return {"status":"error","reason":f"No running display thread for window '{window_name}'"}
    t0 = time.time()
    frame = None
    while time.time() - t0 < timeout_s:
        with th._frame_lock:
            if th._last_frame is not None:
                frame = th._last_frame.copy()
                break
        time.sleep(0.01)
    if frame is None:
        return {"status":"error","reason":"Timeout waiting for a frame from display thread"}
    _ensure_dir(out_path)
    cv2.imwrite(out_path, frame)
    return {"status":"ok","image_path": out_path}







# code for composing homographies to allow for shifted maze position
def _order_tl_tr_br_bl(P: np.ndarray) -> np.ndarray:
    P = P.astype(np.float32)
    s = P.sum(axis=1)           # x+y
    d = np.diff(P, axis=1).ravel()  # x - y
    tl = P[np.argmin(s)]
    br = P[np.argmax(s)]
    tr = P[np.argmin(d)]
    bl = P[np.argmax(d)]
    return np.vstack([tl, tr, br, bl]).astype(np.float32)

def _load_corners_any(path: str) -> np.ndarray:
    """Load corners from .npy or .json; returns (4,2) TL,TR,BR,BL (float32)."""
    if path.lower().endswith(".npy"):
        C = np.load(path).astype(np.float32)
        if C.shape != (4,2):
            raise ValueError(f"{path} must be shape (4,2)")
        return _order_tl_tr_br_bl(C)
    elif path.lower().endswith(".json"):
        with open(path, "r") as f:
            D = json.load(f)
        C = np.array([D["TL"], D["TR"], D["BR"], D["BL"]], dtype=np.float32)
        return _order_tl_tr_br_bl(C)
    else:
        raise ValueError(f"Unsupported corners file type: {path}")

def compose_current_homography(
    H_ref_path: Annotated[str, "np.save path of 3x3 homography (REF pixels -> ROBOT)."],
    ref_corners_path: Annotated[str, "Reference image corners TL,TR,BR,BL (.npy or .json)."],
    cur_corners_path: Annotated[str, "Current image corners TL,TR,BR,BL (.npy or .json)."],
    out_npy: Annotated[str, "Where to save H_cur (CURRENT pixels -> ROBOT)."] = "data/cur/H_cur.npy",
    verify_image_path: Annotated[Optional[str], "Optional current image to draw verification grid."] = None,
    verify_overlay_path: Annotated[Optional[str], "Optional path to save verification overlay."] = None,
    grid_step: Annotated[float, "Grid step in robot units for overlay."] = 25.0
) -> Dict:
    """Compute H_cur = H_ref @ getPerspectiveTransform(C_cur -> C_ref)."""
    if not os.path.exists(H_ref_path):
        return {"status":"error","reason":f"Missing H_ref: {H_ref_path}"}
    try:
        H_ref = np.load(H_ref_path).astype(np.float64)
        C_ref = _load_corners_any(ref_corners_path).astype(np.float32)
        C_cur = _load_corners_any(cur_corners_path).astype(np.float32)
    except Exception as e:
        return {"status":"error","reason":f"Load failed: {e}"}

    H_cur_to_ref = cv2.getPerspectiveTransform(C_cur, C_ref)         # current px -> reference px
    H_cur = H_ref @ H_cur_to_ref                                      # current px -> robot

    _ensure_dir(out_npy)
    np.save(out_npy, H_cur)

    out = {"status":"ok","H_cur_path": out_npy}

    # optional visual QA
    if verify_image_path and verify_overlay_path and os.path.exists(verify_image_path):
        try:
            img = cv2.imread(verify_image_path, cv2.IMREAD_COLOR)
            H_inv = np.linalg.inv(H_cur)

            # project a coarse robot grid back to image
            # first estimate robot bbox by mapping the current corners
            rob_corners = cv2.perspectiveTransform(C_cur.reshape(1,-1,2), H_cur)[0]
            minX, maxX = float(rob_corners[:,0].min()), float(rob_corners[:,0].max())
            minY, maxY = float(rob_corners[:,1].min()), float(rob_corners[:,1].max())
            xs = np.arange(minX, maxX + 1e-6, grid_step)
            ys = np.arange(minY, maxY + 1e-6, grid_step)

            dbg = img.copy()
            for X in xs:
                seg = np.array([[X,minY],[X,maxY]], dtype=np.float32).reshape(1,-1,2)
                seg_px = cv2.perspectiveTransform(seg, H_inv)[0]
                p1 = tuple(np.round(seg_px[0]).astype(int)); p2 = tuple(np.round(seg_px[1]).astype(int))
                cv2.line(dbg, p1, p2, (0,255,0), 1, cv2.LINE_AA)
            for Y in ys:
                seg = np.array([[minX,Y],[maxX,Y]], dtype=np.float32).reshape(1,-1,2)
                seg_px = cv2.perspectiveTransform(seg, H_inv)[0]
                p1 = tuple(np.round(seg_px[0]).astype(int)); p2 = tuple(np.round(seg_px[1]).astype(int))
                cv2.line(dbg, p1, p2, (0,255,0), 1, cv2.LINE_AA)

            # draw current corners
            for (x,y), lab in zip(C_cur.astype(int), ["TL","TR","BR","BL"]):
                cv2.circle(dbg, (x,y), 6, (0,0,255), -1)
                cv2.putText(dbg, lab, (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            _ensure_dir(verify_overlay_path)
            cv2.imwrite(verify_overlay_path, dbg)
            out["overlay"] = verify_overlay_path
        except Exception:
            pass

    return out


# --- Entrance/Exit markers ----------------------------------------------------
# Add this to maze_tools_autogen.py

from typing import Annotated, Dict, Optional
import os, json, time
import numpy as np
import cv2

def _ensure_dir(p: str):
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)

def _largest_blob_centroid(mask: np.ndarray, min_area: int = 50):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0: 
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        if best is None or a > best[0]:
            best = (a, (cx, cy))
    return None if best is None else best[1]

def _color_mask_hsv(img_bgr: np.ndarray, color: str,
                    sat_min: int = 80, val_min: int = 60) -> np.ndarray:
    """Simple HSV threshold for 'green' or 'red' (two-peak red)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if color == "green":
        lower = np.array([35, sat_min, val_min], np.uint8)
        upper = np.array([85, 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "red":
        lower1 = np.array([0,  sat_min, val_min], np.uint8)
        upper1 = np.array([10, 255, 255], np.uint8)
        lower2 = np.array([170, sat_min, val_min], np.uint8)
        upper2 = np.array([179, 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    else:
        raise ValueError("color must be 'green' or 'red'")
    # clean gently
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    return mask

def _restrict_to_border(mask: np.ndarray, band_px: int) -> np.ndarray:
    """Keep only a border band (top/bottom/left/right) to avoid false detections inside the maze."""
    H, W = mask.shape[:2]
    band = max(1, min(band_px, min(H, W)//2))
    border = np.zeros_like(mask)
    border[:band, :] = 255
    border[-band:, :] = 255
    border[:, :band] = 255
    border[:, -band:] = 255
    return cv2.bitwise_and(mask, border)

def detect_entry_exit_markers(
    image_path: Annotated[str, "Current image path."],
    out_json: Annotated[str, "Save marker centers as JSON: {'green_px':[x,y],'red_px':[x,y]}."],
    overlay_out: Annotated[str, "Save a debug overlay image (PNG/JPG)."],
    border_band_frac: Annotated[float, "Restrict detection to this fraction of the min(H,W) near edges."] = 0.08,
    min_area: Annotated[int, "Minimum blob area (pixels)."] = 50,
    sat_min: Annotated[int, "HSV S min threshold."] = 80,
    val_min: Annotated[int, "HSV V min threshold."] = 60
) -> Dict:
    """Detect GREEN (entrance) and RED (exit) dots near the border."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"status":"error","reason":f"Cannot read {image_path}"}
    H, W = img.shape[:2]
    band_px = int(round(min(H, W) * float(border_band_frac)))

    mg = _color_mask_hsv(img, "green", sat_min, val_min)
    mr = _color_mask_hsv(img, "red",   sat_min, val_min)
    mg = _restrict_to_border(mg, band_px)
    mr = _restrict_to_border(mr, band_px)

    g = _largest_blob_centroid(mg, min_area=min_area)
    r = _largest_blob_centroid(mr, min_area=min_area)

    if g is None or r is None:
        ov = img.copy()
        if g is None:
            cv2.putText(ov, "NO GREEN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if r is None:
            cv2.putText(ov, "NO RED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        _ensure_dir(overlay_out); cv2.imwrite(overlay_out, ov)
        return {"status":"error","reason":"Missing green or red marker", "overlay": overlay_out}

    ov = img.copy()
    pG = (int(round(g[0])), int(round(g[1])))
    pR = (int(round(r[0])), int(round(r[1])))
    cv2.circle(ov, pG, 10, (0,255,0), -1, cv2.LINE_AA)  # green entrance
    cv2.circle(ov, pR, 10, (0,0,255), -1, cv2.LINE_AA)  # red exit
    _ensure_dir(overlay_out); cv2.imwrite(overlay_out, ov)

    meta = {"green_px": [float(g[0]), float(g[1])],
            "red_px":   [float(r[0]), float(r[1])]}
    _ensure_dir(out_json)
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    return {"status":"ok", "markers_json": out_json, "overlay": overlay_out, **meta}

# --- Marker-driven weighted solver ------------------------------------------
# Add to the same file where your existing solve_maze_weighted lives.

def _snap_to_corridor(corr_roi: np.ndarray, r0: int, c0: int, max_radius: int = 35):
    """Find nearest pixel (r,c) with corr_roi[r,c]==1. Returns None if not found within radius."""
    H, W = corr_roi.shape[:2]
    r0 = int(np.clip(r0, 0, H-1)); c0 = int(np.clip(c0, 0, W-1))
    if corr_roi[r0, c0] > 0:
        return (r0, c0)
    for rad in range(1, int(max_radius)+1):
        rmin = max(0, r0 - rad); rmax = min(H, r0 + rad + 1)
        cmin = max(0, c0 - rad); cmax = min(W, c0 + rad + 1)
        sub = corr_roi[rmin:rmax, cmin:cmax]
        ys, xs = np.where(sub > 0)
        if len(xs) > 0:
            # nearest in Euclidean distance
            dy = ys - (r0 - rmin)
            dx = xs - (c0 - cmin)
            idx = int(np.argmin(dx*dx + dy*dy))
            return (int(rmin + ys[idx]), int(cmin + xs[idx]))
    return None

def _roi_rc_to_image_xy(path_rc, crop, scale):
    """Convert [(r,c)...] in ROI (downscaled) to [(x,y)...] in original pixels."""
    y0, y1, x0, x1 = crop
    inv = 1.0 / max(scale, 1e-6)
    out = []
    for (r, c) in path_rc:
        x = (x0 + c) * inv
        y = (y0 + r) * inv
        out.append([float(x), float(y)])
    return out

def solve_maze_weighted(
    image_path: Annotated[str, "Current image path."],
    out_json: Annotated[str, "Save solver JSON (includes path_px)."],
    out_overlay: Annotated[str, "Save overlay PNG on the current image."],
    lam: Annotated[float, "Weight for centerline bias." ] = 6.0,
    eps: Annotated[float, "Stability (distance epsilon)."] = 1.0,
    maxdim: Annotated[int, "Downscale max dimension."] = 1200,
    blur: Annotated[int, "Gaussian blur for binarization."] = 5,
    wall_open: Annotated[int, "Morph open iterations."] = 3,
    wall_close: Annotated[int, "Morph close iterations."] = 9,
    prefer_lr: Annotated[bool, "Prefer left/right openings when auto-detecting."] = False,
    markers_json: Annotated[Optional[str], "If provided, green/red markers override auto-openings."] = None
) -> Dict:
    """
    Weighted Dijkstra centerline solver. If markers_json is given:
      - start at GREEN dot, end at RED dot;
      - both are snapped to nearest corridor pixel;
      - the path is validated to stay strictly inside the corridor.
    """
    # ---- Load and downscale ----
    gray0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        return {"status":"error","reason":f"Cannot read {image_path}"}
    # You should already have these helpers available in your codebase.
    gray, scale = resize_keep_aspect(gray0, maxdim)  # (downscaled),  scale = downscaled/original
    H0, W0 = gray0.shape[:2]

    # ---- Binarize & crop to outer walls ----
    corr, walls = binarize_and_clean(gray, blur, wall_open, wall_close)  # corr=1 in corridors
    gray_roi, corr_roi, (y0,y1,x0,x1) = crop_to_walls(gray, walls, pad=3)
    walls_roi = 1 - corr_roi
    H, W = corr_roi.shape

    # ---- Entrance/Exit seeding ----
    if markers_json:
        try:
            with open(markers_json, "r") as f:
                M = json.load(f)
            gx, gy = float(M["green_px"][0]), float(M["green_px"][1])  # original px
            rx, ry = float(M["red_px"][0]),   float(M["red_px"][1])
        except Exception as e:
            return {"status":"error","reason":f"Bad markers_json: {e}"}

        # Map markers to ROI (downscaled)
        gx_ds = gx * scale; gy_ds = gy * scale
        rx_ds = rx * scale; ry_ds = ry * scale
        g_rc = (int(round(gy_ds - y0)), int(round(gx_ds - x0)))
        r_rc = (int(round(ry_ds - y0)), int(round(rx_ds - x0)))

        # Snap to nearest corridor pixels to ensure the seeds are valid
        g_seed = _snap_to_corridor(corr_roi, *g_rc, max_radius=35)
        r_seed = _snap_to_corridor(corr_roi, *r_rc, max_radius=35)
        if g_seed is None or r_seed is None:
            return {"status":"error","reason":"Markers too far from corridor; could not snap to valid seeds."}

        start = g_seed
        goal  = r_seed
        openings_dbg = {"start_marker_rc": list(map(int, start)), "goal_marker_rc": list(map(int, goal))}
    else:
        # Fallback to your existing automatic opening detection (unchanged)
        try:
            (s1,c1), (s2,c2), dbg = detect_openings(walls_roi, prefer_top_bottom=not prefer_lr)
        except Exception as e:
            return {"status":"error","reason":f"Opening detection failed: {e}"}
        band = max(20, min(H, W)//25)
        Ty = int(np.argmax(walls_roi[:band, :].sum(axis=1)))
        By = H - band + int(np.argmax(walls_roi[H-band:H, :].sum(axis=1)))
        Lx = int(np.argmax(walls_roi[:, :band].sum(axis=0)))
        Rx = W - band + int(np.argmax(walls_roi[:, W-band:W].sum(axis=0)))
        def seed(side, center):
            if side == "top":    return (Ty+1, center)
            if side == "bottom": return (By-1, center)
            if side == "left":   return (center, Lx+1)
            if side == "right":  return (center, Rx-1)
            raise ValueError(side)
        start = seed(s1,c1); goal = seed(s2,c2)
        openings_dbg = {"start_opening": [s1,int(c1)], "goal_opening": [s2,int(c2)]}

    # ---- Distance transform inside corridor (downscaled ROI) ----
    corr_dt = corr_roi.copy()
    corr_dt[0,:]=0; corr_dt[-1,:]=0; corr_dt[:,0]=0; corr_dt[:,-1]=0
    dist = cv2.distanceTransform((corr_dt*255).astype(np.uint8), cv2.DIST_L2, 3)

    # ---- Weighted Dijkstra (green→red) ----
    path_rc, achieved_min, achieved_avg = dijkstra_weighted(dist, start, goal, lam=float(lam), eps=float(eps))
    if not path_rc:
        return {"status":"error","reason":"No path found inside corridor."}

    # ---- Validate: path must stay strictly inside corridor ----
    ok_inside = all(0 <= r < H and 0 <= c < W and corr_roi[r, c] > 0 for (r,c) in path_rc)
    if not ok_inside:
        return {"status":"error","reason":"Solver path left the corridor (invalid)."}

    # ---- Safe region (for overlay) ----
    used_T = int(np.floor(achieved_min))
    safe = (dist >= max(1, used_T)).astype(np.uint8)
    safe[0,:]=0; safe[-1,:]=0; safe[:,0]=0; safe[:,-1]=0

    # ---- Overlays at original size ----
    base_small = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out_small = draw_safe_and_path_on_canvas(base_small.copy(), safe, path_rc, (y0,y1,x0,x1))
    overlay = cv2.resize(out_small, (W0, H0), interpolation=cv2.INTER_NEAREST)

    # Draw green/red endpoints on the overlay for clarity
    path_px = _roi_rc_to_image_xy(path_rc, (y0,y1,x0,x1), scale)  # (x,y) in original px
    pG = tuple(np.int32(np.round(path_px[0])))
    pR = tuple(np.int32(np.round(path_px[-1])))
    cv2.circle(overlay, pG, 10, (0,255,0), -1, cv2.LINE_AA)
    cv2.circle(overlay, pR, 10, (0,0,255), -1, cv2.LINE_AA)

    _ensure_dir(out_overlay); cv2.imwrite(out_overlay, overlay)

    # ---- Report (including actual pixel polyline) ----
    out = {
        "status":"ok",
        "path_px": np.round(np.array(path_px, dtype=np.float32), 2).tolist(),
        "path_points": int(len(path_px)),
        "solution_overlay": out_overlay,
        "scale": float(scale),
        "achieved_min_clearance_downscaled_px": float(achieved_min),
        "achieved_min_clearance_equiv_original_px": float(achieved_min / max(scale,1e-6)),
        "weighted_lambda": float(lam),
        "weighted_eps": float(eps),
    }
    out.update(openings_dbg)

    _ensure_dir(out_json)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    return out

# Optional tiny wrapper so you can keep your old system prompt unchanged:
def solve_maze_weighted_from_markers(
    image_path: Annotated[str, "Current image path."],
    markers_json: Annotated[str, "Markers JSON with green_px/red_px in original pixels."],
    out_json: Annotated[str, "Save solver JSON."],
    out_overlay: Annotated[str, "Save overlay PNG."],
    lam: float = 6.0, eps: float = 1.0, maxdim: int = 1200, blur: int = 5, wall_open: int = 3, wall_close: int = 9,
    prefer_lr: bool = False
) -> Dict:
    return solve_maze_weighted(
        image_path=image_path, out_json=out_json, out_overlay=out_overlay,
        lam=lam, eps=eps, maxdim=maxdim, blur=blur, wall_open=wall_open, wall_close=wall_close,
        prefer_lr=prefer_lr, markers_json=markers_json
    )
