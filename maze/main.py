from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage

#

from autogen_ext.models.openai import OpenAIChatCompletionClient

from PIL import Image

from dotenv import load_dotenv

import os
import cv2
import asyncio

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_client = OpenAIChatCompletionClient(
    model="gpt-5",
    api_key=api_key
)

def camera_tool(camera_index: int, output_path: str) -> str:
    """
    Capture an image from a given camera index and save it to the specified output path.
    Args:
        camera_index (int): The camera index (0 for default camera).
        output_path (str): The path to save the captured image(maze.jpg for default path).
    Returns:
        dict: A dictionary containing the saved image path.
    """
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Camera failed to capture image.")
    cv2.imwrite(output_path, frame)
    return {"image_path": output_path}

camera_agent = AssistantAgent(
    name="CameraAgent",
    description="Captures a maze image from the AI Kit camera.",
    model_client=model_client,
    tools=[camera_tool],
    system_message="You are an autonomous robot vision assistant." 
        "When the user asks you to capture an image, call `camera_tool(camera_index, output_path)` "
        "directly to take the picture. Do not just describe the action — execute it."
)

async def test_camera_agent():
    task = "Use camera_tool with camera_index=0 and output_path='maze.jpg' to capture an image."
    result = await camera_agent.run(task=task)
    print(result.messages[-1].content)


#===================================================================================================
#                     maze solver agent
#===================================================================================================

from autogen_core import Image as AGImage


MAZE_SYSTEM = """
You are MazeVision, a rigorous maze solver.

You will receive ONE image of a maze that should be treated as a 4x4 grid of cells.
Infer which edges of each cell are open (no wall) or closed (wall), using 4‑neighbor connectivity only.
The entrance/exit must be border cells whose outer edge is OPEN.

RESPONSE FORMAT (JSON only; no extra text):
{
  "grid_size": {"rows": 4, "cols": 4},
  "cells": [
    [ {"N":bool,"E":bool,"S":bool,"W":bool}, {"N":...}, {"N":...}, {"N":...} ],
    [ ... 4 per row ],
    [ ... ],
    [ ... ]
  ],
  "entrance": {"cell":[row,col], "side":"top|bottom|left|right"},
  "exit": {"cell":[row,col], "side":"top|bottom|left|right"},
  "path_cells": [[r,c], [r,c], ...],   // start cell first, exit cell last, 4‑neighbor steps only
  "moves": "U,R,R,D,L,...",            // one letter per single‑cell step; must match path_cells
  "confidence": 0.0
}

HARD RULES:
- Coordinates are 0‑indexed; (0,0) is top‑left. Rows increase downward, cols to the right.
- A move between adjacent cells is legal only if BOTH cells are open toward each other (e.g., A.E==true AND B.W==true).
- The first element of path_cells MUST equal entrance.cell; the last MUST equal exit.cell.
- If the image is ambiguous (low contrast, broken lines), answer with:
  {"ambiguous": true, "reason": "...", "confidence": 0.0}
- Reply with JSON ONLY. Do not include explanations or markdown.
"""


maze_solver = AssistantAgent(
    name="MazeVision",
    description="You are a visual reasoning expert that solves mazes from images using logical deduction.",
    model_client=model_client,
    system_message=MAZE_SYSTEM
)

# ---------------- FEW-SHOT EXAMPLES ---------------- #
def create_example(unsolved_path, solved_path, json_path):

    unsolved_img = AGImage(Image.open(unsolved_path))
    solved_img = AGImage(Image.open(solved_path))

    with open(json_path, "r") as f:
        json_solution = f.read().strip()

    reasoning_text = f"""
Example Maze:
The first image is the unsolved maze (black walls on white).
The second image shows the correct path in red.
Below is the corresponding structured JSON representation of the maze and its solution:

{json_solution}

From this pattern, learn how to infer open edges, entrance/exit, and path sequence.
"""
    return MultiModalMessage(
        content=[
            reasoning_text,
            unsolved_img,
            solved_img
        ],
        source="User"
    )

example1 = create_example(
    "examples/maze.png",
    "examples/solved_maze.png",
    "examples/maze.json"
)

maze_img = AGImage(Image.open("examples/maze11.png"))

maze_prompt = MultiModalMessage(
    content=[
        "Now solve this new maze. "
        "Think carefully step-by-step about the entrance, exit, and possible turns. "
        "Output reasoning first, then final JSON only.",
        maze_img
    ],
    source="User"
)

conversation = [example1, maze_prompt]

async def test_maze_solver_agent():

    print(maze_prompt.model_dump_json(indent=2)[:800])
    result = await maze_solver.run(task=conversation)
    print(result.messages[-1].content)


#===================================================================================================
#                                     main
#===================================================================================================



async def main():
    #await test_camera_agent()
    await test_maze_solver_agent()


if __name__=="__main__":
    asyncio.run(main())