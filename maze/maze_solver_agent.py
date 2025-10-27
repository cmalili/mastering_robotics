# maze_tool.py
from __future__ import annotations
from typing import Annotated, Literal, Optional, TypedDict, List
from types import SimpleNamespace

from centerline_pixel import solve_centerline  # <- your solver module

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

# maze_agent.py
import os
import asyncio
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_core import Image as AGImage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from maze_tool import solve_maze  # the tool we defined above

SYSTEM_MESSAGE = """You are MazeToolAgent.
- When a user provides a maze image (or a local file path), DO NOT try to solve it by vision.
- You MUST call the tool `solve_maze` with the provided local `image_path` and reasonable defaults.
- After the tool returns, summarize key fields (entrance, exit, used threshold, achieved clearance if present, path length),
  and include the returned `solution_image` path. If the tool returns error, surface its `reason`.
- If the user asks for a different mode or parameters (e.g., 'weighted', 'min_clearance'), pass them through.
- Keep answers concise and put the raw JSON from the tool in a fenced code block titled `result` at the end.
"""

def build_agent(model: Optional[str] = None) -> AssistantAgent:
    """
    Create an AssistantAgent that can call and execute the `solve_maze` tool.
    Note: parallel_tool_calls is disabled to avoid concurrency issues during file I/O.
    """
    client = OpenAIChatCompletionClient(
        model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
        # api_key can be picked up from OPENAI_API_KEY env var automatically
        parallel_tool_calls=False,
    )
    # Pass the tool directly via `tools=[...]` so the agent can call & execute it.
    agent = AssistantAgent(
        name="MazeToolAgent",
        description="Agent that solves mazes by calling a local Python tool.",
        system_message=SYSTEM_MESSAGE,
        model_client=client,
        tools=[solve_maze],           # <-- this exposes and executes the tool
        reflect_on_tool_use=True,     # brief final summary after tool calls
        max_tool_iterations=3,        # safety bound
    )
    return agent

async def run_with_path(image_path: str, *, mode="widest"):
    """Simplest way: give a path in plain text so the model can fill the tool args."""
    agent = build_agent()
    task = TextMessage(
        content=f"Solve the maze at local path '{image_path}' using mode='{mode}'. "
                f"Return path_original, waypoints_original, moves and the overlay path.",
        source="User",
    )
    result = await agent.run(task=task)
    print(result.messages[-1].content)  # assistant's final reply

async def run_with_image(image_path: str, *, mode="widest"):
    """
    Multimodal example: attach the image for context AND include the local path as text.
    (The tool needs the filesystem path; AutoGen's multimodal content does not expose
     a local file path to the tool automatically.)
    """
    from PIL import Image
    pil_image = Image.open(image_path)
    ag_img = AGImage(pil_image)
    agent = build_agent()
    mm = MultiModalMessage(
        content=[
            f"Please call `solve_maze` with image_path='{image_path}', mode='{mode}'.",
            ag_img,
        ],
        source="User",
    )
    result = await agent.run(task=mm)
    print(result.messages[-1].content)

if __name__ == "__main__":
    # Example:
    #   export OPENAI_API_KEY=sk-...
    #   python maze_agent.py
    img = os.environ.get("MAZE_IMAGE", "the_maze.jpg")
    asyncio.run(run_with_path(img, mode="widest"))
