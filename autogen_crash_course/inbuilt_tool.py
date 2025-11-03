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

