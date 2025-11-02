#!/usr/bin/env python3
"""
calibrate_swarm.py
AutoGen Swarm calibration team:
CornerAgent -> CorrespondenceAgent -> FitAgent -> VerifyAgent

Design:
- Each agent calls exactly one tool and then issues a HandoffMessage to the next agent.
- The final agent (VerifyAgent) hands off to "user" with a succinct JSON summary.
- Team termination: HandoffTermination(target="user").

Usage
-----
# A) Using an existing reference image and a correspondences file:
python calibrate_swarm.py \
  --ref-image path/to/maze_ref.jpg \
  --corresp path/to/corresp.txt \
  --save-prefix ref/mymaze \
  --model gpt-4o

# B) Capture reference from camera index 0, then use a correspondences file:
python calibrate_swarm.py \
  --camera 0 \
  --corresp path/to/corresp.txt \
  --save-prefix ref/mymaze
"""

from __future__ import annotations
import os, argparse, json
from typing import Optional
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import HandoffTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from dotenv import load_dotenv

# Our deterministic tools:
from calib_tools import (
    capture_reference_image, use_existing_reference_image,
    detect_reference_corners, load_correspondences,
    fit_homography_pixels_to_robot, verify_homography_on_reference
)


load_dotenv()

SYSTEM_CORNER = """You are CornerAgent.
- Your ONLY job is to obtain a reference image and detect its four outer corners.
- Then write a one-line JSON summary and HANDOFF to CorrespondenceAgent.
Call exactly one of:
  - use_existing_reference_image(src_image_path, dst_image_path)  # if ref-image path is provided in the task
  - capture_reference_image(out_path, camera_index, width, height, warmup_frames)  # if camera is provided
Then call detect_reference_corners(image_path=ref/maze_ref.jpg, out_npy=<save_prefix>_ref_corners.npy, overlay_path=<save_prefix>_corners_overlay.png)
"""

SYSTEM_CORR = """You are CorrespondenceAgent.
- Load pixel↔robot correspondences (u v Xr Yr lines; >=4).
- If a file path is missing, HANDOFF to user with a clear request (do not guess).
- Otherwise, call load_correspondences(file_path) and HANDOFF to FitAgent with the loaded arrays.
"""

SYSTEM_FIT = """You are FitAgent.
- Fit a pixels->robot homography (H_ref) using RANSAC with the loaded correspondences.
- Save it to <save_prefix>_H_ref.npy.
- Include the fit stats (n_points, n_inliers, rms, median) in your message.
- HANDOFF to VerifyAgent.
"""

SYSTEM_VERIFY = """You are VerifyAgent.
- Verify H_ref visually by drawing a projected robot-grid onto the reference image.
- Save overlay at <save_prefix>_hverify_overlay.png.
- Reply with a short JSON containing:
  { "status":"ok", "H_ref":"<path>", "ref_corners":"<path>", "rms": <float>, "median": <float>, "overlays":{...} }
- Then HANDOFF to "user" to finish the team.
"""

api_key = os.environ.get("OPENAI_API_KEY")

def build_model(model_name: Optional[str]):
    return OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=api_key,
        parallel_tool_calls=False,   # IMPORTANT for team/tool stability
    )

def build_team(save_prefix: str, *, model_name: Optional[str]=None):
    model_client = build_model(model_name)

    # Agents with tools they are allowed to call and explicit handoffs
    corner_agent = AssistantAgent(
        name="CornerAgent",
        model_client=model_client,
        system_message=SYSTEM_CORNER + f"\nUse save_prefix='{save_prefix}'.\n",
        tools=[use_existing_reference_image, capture_reference_image, detect_reference_corners],
        handoffs=["CorrespondenceAgent", "user"],  # can handoff to corr or user
        description="Captures or loads the reference image and detects corners."
    )

    corr_agent = AssistantAgent(
        name="CorrespondenceAgent",
        model_client=model_client,
        system_message=SYSTEM_CORR + f"\nUse save_prefix='{save_prefix}'.\n",
        tools=[load_correspondences],
        handoffs=["FitAgent", "user"],
        description="Loads the pixel↔robot pairs file."
    )

    fit_agent = AssistantAgent(
        name="FitAgent",
        model_client=model_client,
        system_message=SYSTEM_FIT + f"\nUse save_prefix='{save_prefix}'.\n",
        tools=[fit_homography_pixels_to_robot],
        handoffs=["VerifyAgent", "user"],
        description="Fits H_ref using RANSAC."
    )

    verify_agent = AssistantAgent(
        name="VerifyAgent",
        model_client=model_client,
        system_message=SYSTEM_VERIFY + f"\nUse save_prefix='{save_prefix}'.\n",
        tools=[verify_homography_on_reference],
        handoffs=["user"],
        description="Verifies calibration and finalizes."
    )

    team = Swarm(
        participants=[corner_agent, corr_agent, fit_agent, verify_agent],
        termination_condition=HandoffTermination(target="user") | MaxMessageTermination(max_messages=16)
    )
    return team

async def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref-image", type=str, help="Use an existing reference image at this path.")
    g.add_argument("--camera", type=int, help="Capture a new reference frame from this camera index.")

    ap.add_argument("--corresp", type=str, default=None, help="Text file with rows: u v Xr Yr (>=4 lines).")
    ap.add_argument("--save-prefix", type=str, required=True, help="Prefix for saving artifacts, e.g., ref/mymaze")
    ap.add_argument("--model", type=str, default=None, help="OpenAI model name (default from env or gpt-4o)")

    args = ap.parse_args()

    # Ensure ref/ folder exists and set canonical paths based on save_prefix
    ref_img_dst = f"{args.save_prefix}_maze_ref.jpg"
    corners_npy = f"{args.save_prefix}_ref_corners.npy"
    corners_overlay = f"{args.save_prefix}_corners_overlay.png"
    href_npy = f"{args.save_prefix}_H_ref.npy"
    hverify_png = f"{args.save_prefix}_hverify_overlay.png"

    # Build the team
    team = build_team(args.save_prefix, model_name=args.model)

    # Seed task for the first agent (CornerAgent). We pass concrete paths so it can call tools deterministically.
    if args.ref_image:
        seed = (
            "Calibrate using an EXISTING reference image.\n"
            f"- Call use_existing_reference_image(src_image_path='{args.ref_image}', dst_image_path='{ref_img_dst}')\n"
            f"- Then call detect_reference_corners(image_path='{ref_img_dst}', out_npy='{corners_npy}', overlay_path='{corners_overlay}')\n"
            f"- After detection, handoff to CorrespondenceAgent."
        )
    else:
        seed = (
            "Calibrate by CAPTURING a reference image from camera.\n"
            f"- Call capture_reference_image(out_path='{ref_img_dst}', camera_index={args.camera})\n"
            f"- Then call detect_reference_corners(image_path='{ref_img_dst}', out_npy='{corners_npy}', overlay_path='{corners_overlay}')\n"
            f"- After detection, handoff to CorrespondenceAgent."
        )

    # Run Swarm from CornerAgent
    result = await team.run(task=TextMessage(content=seed, source="user"))

    # If correspondences were not provided, CorrespondenceAgent should have handed off to user.
    # If you DID provide --corresp, continue by resuming the team with a HandoffMessage to CorrespondenceAgent
    if args.corresp:
        # Resume: provide the file path deterministically; this avoids a human round-trip.
        resume_msg = HandoffMessage(
            content=(
                "Use this correspondences file now:\n"
                f"- load_correspondences(file_path='{args.corresp}')\n"
                f"- Then FitAgent should call fit_homography_pixels_to_robot(\n"
                f"    pix_uv=<from loader>, rob_xy=<from loader>, out_npy='{href_npy}')\n"
                f"- Finally VerifyAgent should call verify_homography_on_reference(\n"
                f"    ref_image_path='{ref_img_dst}', H_ref_path='{href_npy}', ref_corners_path='{corners_npy}',\n"
                f"    overlay_path='{hverify_png}') and HANDOFF to user."
            ),
            source="user",
            target="CorrespondenceAgent",
        )
        result = await team.run(task=resume_msg)

    # Print the last assistant content (should include the JSON summary)
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())