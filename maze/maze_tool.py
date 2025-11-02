# maze_tool.py
from __future__ import annotations
from typing import Annotated, Literal, Optional, TypedDict, List
from types import SimpleNamespace

from d import solve_centerline  # <- your solver module

class MazeResult(TypedDict, total=False):
    status: str
    mode: Literal["widest", "weighted", "shortest"]
    solution_image: str
    scale: float
    threshold_used_downscaled_px: int
    threshold_equiv_original_px: float
    path_length_px: int
    entrance: dict
    exit: dict
    # full paths:
    path_roi_downscaled: List[List[int]]
    path_downscaled: List[List[int]]
    path_original: List[List[int]]
    waypoints_original: List[List[int]]
    moves: List[str]
    # optional extras depending on mode:
    achieved_min_clearance_downscaled_px: float
    achieved_min_clearance_equiv_original_px: float
    achieved_avg_clearance_downscaled_px: float
    achieved_avg_clearance_equiv_original_px: float
    weighted_lambda: float
    weighted_eps: float
    min_clearance_downscaled_px: int
    min_clearance_equiv_original_px: float

def solve_maze(
    image_path: Annotated[str, "Local path to the maze image (e.g., 'the_maze.jpg')."],
    mode: Annotated[Literal["widest", "weighted", "shortest"], "Centerline strategy."] = "widest",
    # shortest-mode minimum clearance (optional for widest/weighted):
    min_clearance: Annotated[Optional[int], "Minimum clearance (original pixels) for mode='shortest'."] = None,
    clearance_ref: Annotated[Literal["original", "downscaled"], "Units for min_clearance."] = "original",
    min_clearance_mm: Annotated[Optional[float], "Minimum clearance in mm (requires maze_width_mm)."] = None,
    maze_width_mm: Annotated[Optional[float], "Physical width of the maze drawing, in mm."] = None,
    # weighted-mode knobs:
    lam: Annotated[float, "Center preference strength for mode='weighted'."] = 6.0,
    eps: Annotated[float, "Stability term in cost 1 + lam/(eps + dist)."] = 1.0,
    # preprocessing & behavior:
    maxdim: Annotated[int, "Downscale long side for speed (0 = no downscale)."] = 1200,
    blur: Annotated[int, "Gaussian blur kernel (odd >=3); 0 disables."] = 5,
    wall_open: Annotated[int, "Morphology OPEN for walls (noise removal); 0 disables."] = 3,
    wall_close: Annotated[int, "Morphology CLOSE for walls (seal tiny gaps); 0 disables."] = 9,
    prefer_lr: Annotated[bool, "Prefer left/right openings if top–bottom absent."] = False,
    include_gate: Annotated[bool, "Widest: include entrance/exit slit in clearance score."] = False,
    # optional exports:
    save_csv: Annotated[Optional[str], "Optional CSV path (original coords)."] = None,
    save_json: Annotated[Optional[str], "Optional JSON path (full result)."] = None,
) -> MazeResult:
    """
    Solve a maze image using your centerline_pixel solver and return the result dict.
    This function is registered as a tool for the agent.
    """
    # Bridge from tool signature => your solver's args object
    args = SimpleNamespace(
        mode=mode,
        min_clearance=min_clearance,
        clearance_ref=clearance_ref,
        min_clearance_mm=min_clearance_mm,
        maze_width_mm=maze_width_mm,
        lam=lam,
        eps=eps,
        maxdim=maxdim,
        blur=blur,
        open=wall_open,        # note: your solver expects 'open'/'close'
        close=wall_close,
        prefer_lr=prefer_lr,
        include_gate=include_gate,
        save_csv=save_csv,
        save_json=save_json,
    )
    res = solve_centerline(image_path, args)  # returns a dict
    # Return as-is; the agent will surface it to the user.
    return res  # type: ignore[return-value]
