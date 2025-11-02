#!/usr/bin/env python3
"""
runtime_swarm.py
Runtime Swarm team: Capture -> Register -> Solve -> Map -> Preview -> Execute
Solves the current maze image, updates homography, maps to robot, and (optionally) executes.

Usage examples
--------------
# Using camera index 0 (recommended):
python runtime_swarm.py \
  --camera 0 \
  --ref-corners ref/mymaze_ref_corners.npy \
  --href ref/mymaze_H_ref.npy \
  --save-dir runs/2025-10-29_001 \
  --model gpt-4o-mini \
  --z 1.5 \
  --spacing 2.0 \
  --dryrun

# Using an existing current image file:
python runtime_swarm.py \
  --current-image samples/maze_now.jpg \
  --ref-corners ref/mymaze_ref_corners.npy \
  --href ref/mymaze_H_ref.npy \
  --save-dir runs/2025-10-29_002 \
  --dryrun
"""

from __future__ import annotations
import os, argparse, json
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import HandoffTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Tools (deterministic functions)
from runtime_tools import (
    capture_current_image, use_existing_current_image, detect_current_corners,
    compose_current_homography, solve_maze_current, map_path_to_robot,
    preview_overlay, robot_execute
)

# -------------------- system messages (tight, explicit) --------------------

SYSTEM_CAPTURE = """You are CaptureAgent.
- Capture or load the CURRENT image and save it to the given path.
- Then HANDOFF to RegisterAgent.
Call exactly one of:
  - use_existing_current_image(src_image_path, dst_image_path)
  - capture_current_image(out_path, camera_index, width, height, warmup_frames)
"""

SYSTEM_REGISTER = """You are RegisterAgent.
- Detect outer corners in the CURRENT image and compose the CURRENT pixels->robot homography.
- If detection fails, HANDOFF back to CaptureAgent (suggest adjusting capture).
- Otherwise, HANDOFF to SolveAgent.
Steps:
  1) detect_current_corners(image_path, out_npy='<save>/cur/cur_corners.npy', overlay_path='<save>/cur/corners_overlay.png')
  2) compose_current_homography(H_ref_path, ref_corners_path, cur_corners_path='<save>/cur/cur_corners.npy', out_npy='<save>/cur/H_cur.npy')
"""

SYSTEM_SOLVE = """You are SolveAgent.
- Solve the maze on the CURRENT image (not cached).
- If solver returns error, you MAY retry once lowering constraints (e.g., mode='widest').
- On success, HANDOFF to MapAgent.
Call:
  solve_maze_current(image_path, save_json='<save>/solver/result.json', mode='<mode>', min_clearance=<min_clearance or None>,
                     lam=<lam>, eps=<eps>, maxdim=<maxdim>, blur=<blur>, wall_open=<wall_open>, wall_close=<wall_close>)
"""

SYSTEM_MAP = """You are MapAgent.
- Map the pixel path to robot coordinates using H_cur, resample, and save CSV.
- If out of bounds/safety were to be checked, this tool would reject (not implemented here).
- HANDOFF to PreviewAgent.
Call:
  map_path_to_robot(solver_json='<save>/solver/result.json', H_cur_path='<save>/cur/H_cur.npy',
                    z_const=<z>, spacing=<spacing>, out_csv='<save>/robot/path.csv')
"""

SYSTEM_PREVIEW = """You are PreviewAgent.
- Draw a preview overlay of the solution on the CURRENT image for operator confirmation.
- HANDOFF to ExecuteAgent.
Call:
  preview_overlay(image_path='<save>/cur/frame.jpg', solver_json='<save>/solver/result.json',
                  H_cur_path='<save>/cur/H_cur.npy', out_path='<save>/cur/preview_overlay.png',
                  draw_backproject=True)
"""

SYSTEM_EXECUTE = """You are ExecuteAgent.
- Execute the robot path (dry-run by default).
- Respond with a short JSON: {"status":"ok","dryrun":true,"csv":"..."} then HANDOFF to "user".
Call:
  robot_execute(robot_csv='<save>/robot/path.csv', dryrun=<dryrun>, driver=<driver or None>, speed=<speed>)
"""

# -------------------- build model/client --------------------

def build_model(model_name: Optional[str]):
    return OpenAIChatCompletionClient(
        model=model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        parallel_tool_calls=False,   # IMPORTANT for tool/team stability
    )

# -------------------- team builder --------------------

def build_runtime_team(save_dir: str, *, model_name: Optional[str]=None,
                       mode: str="widest", min_clearance: Optional[int]=None,
                       lam: float=6.0, eps: float=1.0,
                       maxdim: int=1200, blur: int=5, wall_open: int=3, wall_close: int=9,
                       z: float=0.0, spacing: float=2.0, dryrun: bool=True, driver: Optional[str]=None, speed: float=50.0):

    model_client = build_model(model_name)

    capture_agent = AssistantAgent(
        name="CaptureAgent",
        description="Gets the current frame (camera or file).",
        model_client=model_client,
        tools=[use_existing_current_image, capture_current_image],
        system_message=SYSTEM_CAPTURE
    )

    register_agent = AssistantAgent(
        name="RegisterAgent",
        description="Detect corners and compose current pixels->robot homography.",
        model_client=model_client,
        tools=[detect_current_corners, compose_current_homography],
        system_message=SYSTEM_REGISTER.replace("<save>", save_dir)
    )

    solve_agent = AssistantAgent(
        name="SolveAgent",
        description="Solve the maze on the current image.",
        model_client=model_client,
        tools=[solve_maze_current],
        system_message=SYSTEM_SOLVE.replace("<save>", save_dir)
                                     .replace("<mode>", mode)
                                     .replace("<lam>", str(lam))
                                     .replace("<eps>", str(eps))
                                     .replace("<maxdim>", str(maxdim))
                                     .replace("<blur>", str(blur))
                                     .replace("<wall_open>", str(wall_open))
                                     .replace("<wall_close>", str(wall_close))
                                     .replace("<min_clearance>", "None" if min_clearance is None else str(min_clearance))
    )

    map_agent = AssistantAgent(
        name="MapAgent",
        description="Map pixel path to robot CSV.",
        model_client=model_client,
        tools=[map_path_to_robot],
        system_message=SYSTEM_MAP.replace("<save>", save_dir)
                                 .replace("<z>", str(z))
                                 .replace("<spacing>", "null" if spacing is None else str(spacing))
    )

    preview_agent = AssistantAgent(
        name="PreviewAgent",
        description="Render preview overlay on current image.",
        model_client=model_client,
        tools=[preview_overlay],
        system_message=SYSTEM_PREVIEW.replace("<save>", save_dir)
    )

    execute_agent = AssistantAgent(
        name="ExecuteAgent",
        description="Dry-run or execute on the robot.",
        model_client=model_client,
        tools=[robot_execute],
        system_message=SYSTEM_EXECUTE.replace("<save>", save_dir)
                                     .replace("<dryrun>", "true" if dryrun else "false")
                                     .replace("<driver or None>", "null" if driver is None else f"'{driver}'")
                                     .replace("<speed>", str(speed))
    )

    team = Swarm(
        participants=[capture_agent, register_agent, solve_agent, map_agent, preview_agent, execute_agent],
        termination=[
            HandoffTermination(target="user"),
            MaxMessageTermination(max_messages=24),
        ],
    )

    return team

# -------------------- CLI driver --------------------

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, help="Capture from this camera index.")
    g.add_argument("--current-image", type=str, help="Use this existing current image.")

    ap.add_argument("--ref-corners", required=True, help="Path to ref corners .npy (TL,TR,BR,BL) from calibration.")
    ap.add_argument("--href", required=True, help="Path to H_ref.npy (pixels(ref)->robot).")

    ap.add_argument("--save-dir", required=True, help="Run directory (will be created).")
    ap.add_argument("--model", type=str, default=None, help="OpenAI model (default from env or gpt-4o-mini)")

    # Solver params
    ap.add_argument("--mode", choices=["widest","weighted","shortest"], default="widest")
    ap.add_argument("--min-clearance", type=int, default=None)
    ap.add_argument("--lam", type=float, default=6.0)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--maxdim", type=int, default=1200)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--wall-open", type=int, default=3)
    ap.add_argument("--wall-close", type=int, default=9)

    # Mapping / execution
    ap.add_argument("--z", type=float, default=0.0, help="Constant Z (robot units).")
    ap.add_argument("--spacing", type=float, default=2.0, help="Resample spacing (robot units). Use 0 to disable.")
    ap.add_argument("--dryrun", action="store_true", help="Do not contact robot; just return success.")
    ap.add_argument("--driver", type=str, default=None, help="Robot driver id (if implemented).")
    ap.add_argument("--speed", type=float, default=50.0, help="Robot motion speed (driver-specific).")

    args = ap.parse_args()

    # Make run dirs
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "cur"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "solver"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "robot"), exist_ok=True)

    # Build team
    team = build_runtime_team(
        save_dir=args.save_dir,
        model_name=args.model,
        mode=args.mode,
        min_clearance=args.min_clearance,
        lam=args.lam, eps=args.eps,
        maxdim=args.maxdim, blur=args.blur, wall_open=args.wall_open, wall_close=args.wall_close,
        z=args.z, spacing=(None if (args.spacing is None or args.spacing<=0) else args.spacing),
        dryrun=bool(args.dryrun), driver=args.driver, speed=args.speed
    )

    # Seed to CaptureAgent with explicit file targets so tools are called deterministically
    frame_path = os.path.join(args.save_dir, "cur", "frame.jpg")
    if args.current_image:
        seed = (
            "Use EXISTING current image.\n"
            f"- Call use_existing_current_image(src_image_path='{args.current_image}', dst_image_path='{frame_path}')\n"
            "Then handoff to RegisterAgent."
        )
    else:
        seed = (
            "CAPTURE a current image from camera.\n"
            f"- Call capture_current_image(out_path='{frame_path}', camera_index={args.camera})\n"
            "Then handoff to RegisterAgent."
        )

    # Also provide the calibration artifacts to the RegisterAgent via the task context
    seed += (
        f"\nCalibration artifacts:\n"
        f"- H_ref_path='{args.href}'\n"
        f"- ref_corners_path='{args.ref_corners}'\n"
    )

    # Run swarm from CaptureAgent
    result = team.run(task=TextMessage(content=seed, source="user"))

    # After execution, prompt RegisterAgent explicitly to compose H_cur with exact paths (deterministic)
    resume_reg = HandoffMessage(
        content=(
            "Proceed with registration and mapping steps using these exact paths:\n"
            f"- image_path='{frame_path}'\n"
            f"- H_ref_path='{args.href}'\n"
            f"- ref_corners_path='{args.ref_corners}'\n"
            f"- cur_corners_path='{os.path.join(args.save_dir, 'cur', 'cur_corners.npy')}'\n"
            f"- H_cur_out='{os.path.join(args.save_dir, 'cur', 'H_cur.npy')}'\n"
            f"- solver_json='{os.path.join(args.save_dir, 'solver', 'result.json')}'\n"
            f"- robot_csv='{os.path.join(args.save_dir, 'robot', 'path.csv')}'\n"
            f"- preview_png='{os.path.join(args.save_dir, 'cur', 'preview_overlay.png')}'\n"
            "Follow the step-by-step calls described in your system prompt, handing off after each."
        ),
        source="user",
        target="RegisterAgent"
    )
    result = team.run(task=resume_reg)

    # Print final assistant message (ExecuteAgent should hand off to user)
    print(result.messages[-1].content)

if __name__ == "__main__":
    main()
