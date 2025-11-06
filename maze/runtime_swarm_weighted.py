#!/usr/bin/env python3



# Todo
# 1 fix video feed
# 2 move to waypoint at the beginning
# 3 Take image
# 3b Compose homographies 
# 4 Allow any position for maze
# 5 Display Pip
# 6 Show overlay of solved maze
# 7 Incorporate user
"""
runtime_swarm_weighted.py
AutoGen Swarm runtime that:
- (optional) starts video feed with PiP
- captures current image
- (optional) corners QA overlay
- solves maze (weighted Dijkstra; no BFS)
- maps to robot using precomputed H
- executes robot (stub by default)
- (optional) stops video feed

Requires:
  pip install autogen_ext autogen-core pillow opencv-python-headless (or opencv-python)

Usage
-----
# Use existing image; display disabled; dry-run execution:
python runtime_swarm_weighted.py \
  --current-image the_maze.jpg \
  --href ref/H_ref.npy \
  --save-dir runs/2025-11-04a \
  --lam 6.0 --eps 1.0 --maxdim 1200 --blur 5 --wall-open 3 --wall-close 9 \
  --z 1.5 --spacing 2.0 \
  --dryrun

# Capture from camera 0; start display with PiP overlay; dry-run:
python runtime_swarm_weighted.py \
  --camera 0 \
  --href ref/H_ref.npy \
  --save-dir runs/2025-11-04b \
  --display \
  --lam 6.0 --eps 1.0 --maxdim 1200 --blur 5 --wall-open 3 --wall-close 9 \
  --z 1.5 --spacing 2.0 \
  --dryrun
"""

from __future__ import annotations
import os, argparse, json, asyncio
from typing import Optional

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import HandoffTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Tools
from maze_tools_autogen import (
    start_video_pip, stop_video_pip,
    use_existing_current_image, capture_current_image,
    detect_corners_qa,
    solve_maze_weighted,
    map_path_to_robot,
    robot_execute,
    snapshot_from_display,
    compose_current_homography
)

# ---------- System prompts for each agent ------------------------------------

SYSTEM_DISPLAY_START = """You are DisplayStartAgent.
- If a camera is available, start a live video window with PiP overlay.
- Then HANDOFF to CaptureAgent but keep the video going.
Call:
  start_video_pip(camera_index=<cam_index_or_-1>, overlay_path='<solver_overlay>', window_name='Maze Runtime', fps=15.0)
"""

SYSTEM_DISPLAY_STOP = """You are DisplayStopAgent.
- Stop the live video window if it is running.
- Then HANDOFF to "user".
Call:
  stop_video_pip(window_name='Maze Runtime')
"""

SYSTEM_CAPTURE = """You are CaptureAgent.
- Obtain the CURRENT image and save it to the path given by the user.
- Then HANDOFF to CornersAgent.

  If the live display is running (window 'Maze Runtime'), DO NOT open the camera.
  Call snapshot_from_display(window_name='Maze Runtime', out_path='<save>/cur/frame.jpg').
"""

SYSTEM_CORNERS = """You are CornersAgent.
- Optional QA step: detect four outer corners (TL,TR,BR,BL) and draw an overlay.
- Then HANDOFF to SolveAgent.
Call:
  detect_corners_qa(image_path, corners_json_out, overlay_out)
"""

SYSTEM_HOMOGRAPHY = """You are HomographyAgent.
- Compute CURRENT pixels->robot homography using calibration artifacts.
- Call exactly once:
  compose_current_homography(
    H_ref_path='<href>',
    ref_corners_path='<ref_corners>',
    cur_corners_path='<save>/cur/maze_corners.json',
    out_npy='<save>/cur/H_cur.npy',
    verify_image_path='<save>/cur/frame.jpg',
    verify_overlay_path='<save>/cur/hverify_current.png',
    grid_step=25.0)
- Do NOT ask for confirmation.
- Then HANDOFF to SolveAgent.
"""

SYSTEM_SOLVE = """You are SolveAgent.
- Solve the maze using WEIGHTED DIJKSTRA (no BFS).
- Return the ACTUAL pixel path in solver JSON ('path_px').
- Then HANDOFF to MapAgent.
Call:
  solve_maze_weighted(image_path, out_json, out_overlay, lam=<lam>, eps=<eps>, maxdim=<maxdim>, blur=<blur>, wall_open=<wall_open>, wall_close=<wall_close>, prefer_lr=<prefer_lr>)
"""

SYSTEM_MAP = """You are MapAgent.
- Map pixel path to robot XY (and add constant Z), resample, and save CSV.
- Then HANDOFF to ExecuteAgent.
Call:
  map_path_to_robot(solver_json='<save>/solver/result.json', H_cur_path='<save>/cur/H_cur.npy', out_csv, z_const=<z_const>, spacing=<spacing>)
"""

SYSTEM_EXECUTE = """You are ExecuteAgent.
- Execute the robot CSV (dry-run by default).
- Respond with a short JSON: {"status":"ok","dryrun":true,"csv":"..."} and HANDOFF to DisplayStopAgent (or user if no display).
Call:
  robot_execute(csv_path, dryrun=<dryrun>, speed=<speed>, settle_ms=0)
"""

# ---------- Build model ------------------------------------------------------


load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

def build_model(model_name: Optional[str]):
    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
        parallel_tool_calls=False,  # important for deterministic tool calling
    )

# ---------- Build team -------------------------------------------------------

def build_runtime_team(*,
                       model_name: Optional[str],
                       save_dir: str,
                       display: bool,
                       lam: float, eps: float, maxdim: int, blur: int, wall_open: int, wall_close: int, prefer_lr: bool,
                       z_const: float, spacing: Optional[float],
                       dryrun: bool, speed: float,
                       href: str,
                       ref_corners: str):

    model_client = build_model(model_name)

    agents = []

    if display:
        display_start = AssistantAgent(
            name="DisplayStartAgent",
            description="Starts live video feed with PiP.",
            model_client=model_client,
            tools=[start_video_pip],
            system_message=SYSTEM_DISPLAY_START,
            handoffs=["CaptureAgent"]
        )
        agents.append(display_start)

    capture_agent = AssistantAgent(
        name="CaptureAgent",
        description="Captures or loads the current frame.",
        model_client=model_client,
        tools=[snapshot_from_display, use_existing_current_image, capture_current_image],
        system_message=SYSTEM_CAPTURE,
        handoffs=["CornersAgent"]
    )
    agents.append(capture_agent)



    homography_agent = AssistantAgent(
        name="HomographyAgent",
        description="Compose CURRENT pixels->robot homography from calibration + detected corners.",
        model_client=model_client,
        tools=[compose_current_homography],
        system_message=SYSTEM_HOMOGRAPHY
            .replace("<href>", str(href))                    # pass from CLI
            .replace("<ref_corners>", str(ref_corners)),     # pass from CLI
        handoffs=["SolveAgent"]
    )
    agents.append(homography_agent)
    



    corners_agent = AssistantAgent(
        name="CornersAgent",
        description="Optional QA: find four outer corners for overlay.",
        model_client=model_client,
        tools=[detect_corners_qa],
        system_message=SYSTEM_CORNERS,
        handoffs=["HomographyAgent"]
    )
    agents.append(corners_agent)

    solve_agent = AssistantAgent(
        name="SolveAgent",
        description="Solve maze with weighted Dijkstra; output path_px.",
        model_client=model_client,
        tools=[solve_maze_weighted],
        system_message=SYSTEM_SOLVE
            .replace("<lam>", str(lam))
            .replace("<eps>", str(eps))
            .replace("<maxdim>", str(maxdim))
            .replace("<blur>", str(blur))
            .replace("<wall_open>", str(wall_open))
            .replace("<wall_close>", str(wall_close))
            .replace("<prefer_lr>", "true" if prefer_lr else "false"),
        handoffs=["MapAgent"]
    )
    agents.append(solve_agent)

    map_agent = AssistantAgent(
        name="MapAgent",
        description="Map pixel path to robot CSV.",
        model_client=model_client,
        tools=[map_path_to_robot],
        system_message=SYSTEM_MAP
            .replace("<z_const>", str(z_const))
            .replace("<spacing>", "null" if (spacing is None or spacing <= 0) else str(spacing))
            .replace("<save>", save_dir),
        handoffs=["ExecuteAgent"]
    )
    agents.append(map_agent)

    execute_agent = AssistantAgent(
        name="ExecuteAgent",
        description="Execute robot CSV (dry-run by default).",
        model_client=model_client,
        tools=[robot_execute],
        system_message=SYSTEM_EXECUTE
            .replace("<dryrun>", "true" if dryrun else "false")
            .replace("<speed>", str(speed)),
        handoffs=["DisplayStopAgent"]
    )
    agents.append(execute_agent)

    if display:
        display_stop = AssistantAgent(
            name="DisplayStopAgent",
            description="Stops live video feed.",
            model_client=model_client,
            tools=[stop_video_pip],
            system_message=SYSTEM_DISPLAY_STOP
        )
        agents.append(display_stop)

    team = Swarm(
        participants=agents,
        termination_condition=HandoffTermination(target="user") | MaxMessageTermination(max_messages=24)
    )
    return team

# ---------- CLI --------------------------------------------------------------

async def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, help="Capture from this camera index.")
    g.add_argument("--current-image", type=str, help="Use existing image at this path.")
    ap.add_argument("--href", default="data/calibration/homography.npy", help="Precomputed homography .npy (pixels->robot).")
    ap.add_argument("--save-dir", default="data", help="Run directory for artifacts.")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model (default env or gpt-4o-mini).")
    ap.add_argument("--display", action="store_true", help="Start a live camera window with PiP overlay.")

    # Solver params
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--maxdim", type=int, default=1200)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--wall-open", type=int, default=3)
    ap.add_argument("--wall-close", type=int, default=9)
    ap.add_argument("--prefer-lr", action="store_true", help="Prefer left/right openings instead of top/bottom.")

    ap.add_argument("--ref-corners", type=str, default="data/calibration/maze_corners.json", help="Corners from the reference image")
    # Mapping/robot
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--speed", type=float, default=10.0)

    args = ap.parse_args()

    # Prepare directories and canonical paths
    os.makedirs(os.path.join(args.save_dir, "cur"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "solver"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "robot"), exist_ok=True)

    frame_path        = os.path.join(args.save_dir, "cur", "frame.jpg")
    corners_json_path = os.path.join(args.save_dir, "cur", "maze_corners.json")
    corners_overlay   = os.path.join(args.save_dir, "cur", "corners_overlay.png")

    H_cur_path          = os.path.join(args.save_dir, "cur",    "H_cur.npy")
    hverify_png         = os.path.join(args.save_dir, "cur",    "hverify_current.png")

    solver_json_path  = os.path.join(args.save_dir, "solver", "result.json")
    solver_overlay    = os.path.join(args.save_dir, "solver", "overlay.png")
    robot_csv_path    = os.path.join(args.save_dir, "robot", "path.csv")

    team = build_runtime_team(
        model_name=args.model,
        save_dir=args.save_dir,
        display=bool(args.display),
        lam=args.lam, eps=args.eps, maxdim=args.maxdim, blur=args.blur,
        wall_open=args.wall_open, wall_close=args.wall_close, prefer_lr=bool(args.prefer_lr),
        z_const=args.z, spacing=(None if args.spacing is None or args.spacing <= 0 else float(args.spacing)),
        dryrun=bool(args.dryrun), speed=args.speed,
        href=args.href,
        ref_corners=args.ref_corners
    )

    # Build a single deterministic "plan" message with exact calls & paths.
    plan_lines = []
    if args.display:
        cam_index = args.camera if args.camera is not None else -1
        plan_lines.append(
            f"- DisplayStartAgent: start_video_pip(camera_index={cam_index}, overlay_path='{solver_overlay}', window_name='Maze Runtime', fps=15.0)"
        )

    if args.current_image:
        plan_lines.append(
            f"- CaptureAgent: use_existing_current_image(src_image_path='{args.current_image}', dst_image_path='{frame_path}')"
        )
    else:
        plan_lines.append(
            f"- CaptureAgent: capture_current_image(out_path='{frame_path}', camera_index={args.camera}, width=1920, height=1080, warmup_frames=10)"
        )

    plan_lines.append(
        f"- CornersAgent: detect_corners_qa(image_path='{frame_path}', corners_json_out='{corners_json_path}', overlay_out='{corners_overlay}')"
    )


    plan_lines.append(
        f"- HomographyAgent: compose_current_homography("
        f"H_ref_path='{args.href}', "
        f"ref_corners_path='{args.ref_corners}', "
        f"cur_corners_path='{corners_json_path}', "
        f"out_npy='{H_cur_path}', "
        f"verify_image_path='{frame_path}', "
        f"verify_overlay_path='{hverify_png}', grid_step=25.0)"
    )



    plan_lines.append(
        f"- SolveAgent: solve_maze_weighted(image_path='{frame_path}', out_json='{solver_json_path}', out_overlay='{solver_overlay}', lam={args.lam}, eps={args.eps}, maxdim={args.maxdim}, blur={args.blur}, wall_open={args.wall_open}, wall_close={args.wall_close}, prefer_lr={'True' if args.prefer_lr else 'False'})"
    )
    plan_lines.append(
        f"- MapAgent: map_path_to_robot(solver_json='{solver_json_path}', H_cur_path='{H_cur_path}', out_csv='{robot_csv_path}', z_const={args.z}, spacing={('None' if (args.spacing is None or args.spacing<=0) else args.spacing)})"
    )
    plan_lines.append(
        f"- ExecuteAgent: robot_execute(csv_path='{robot_csv_path}', dryrun={'True' if args.dryrun else 'False'}, speed={args.speed})"
    )
    if args.display:
        plan_lines.append(
            f"- DisplayStopAgent: stop_video_pip(window_name='Maze Runtime')"
        )

    seed = "Run the following steps in order, calling exactly one tool each time and handing off to the next agent:\n" + "\n".join(plan_lines)

    # Choose first agent
    first = "DisplayStartAgent" if args.display else "CaptureAgent"
    result = await team.run(task=TextMessage(content=seed, source="user", target=first))

    # Final assistant content (ExecuteAgent or DisplayStopAgent should handoff to user)
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
