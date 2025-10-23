from autogen_agentchat.agents import AssistantAgent

from autogen_ext.models.openai import OpenAIChatCompletionClient

from dotenv import load_dotenv

import os
import cv2
import asyncio

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
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

async def main():
    await test_camera_agent()

if __name__=="__main__":
    asyncio.run(main())