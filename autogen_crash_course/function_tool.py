import asyncio

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

from autogen_core.tools import FunctionTool

from autogen_ext.models.openai import OpenAIChatCompletionClient
#from autogen_ext.ui import Console
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=api_key
)

def reverse_string(text: str) -> str:
    """Reverse the given text
        input : str
        output : str
        The reversed string is returned
    """
    return text[::-1]

# registering function as a tool
reverse_tool = FunctionTool(
    reverse_string,
    description="A too to reverse a string"
)

agent = AssistantAgent(
    name="Assistant",
    description="You are a helpful assistant who can reverse a given string using reverse_tool.",
    model_client=model_client,
    system_message="You are a helpful assistant who can reverse a given string using reverse_tool tool."\
    "Give the result with the summary.",
    tools=[reverse_tool],
    reflect_on_tool_use=True
)

task = "Reverse the text 'Hello, how are you?'"

async def main():
    result = await agent.run(task=task)

    print(f"Agent Response: {result.messages[-1].content}")


if __name__=="__main__":
    asyncio.run(main())