import asyncio

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

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

assistant = AssistantAgent(
    name="assistant",
    description="You are a great assistant",
    model_client=model_client,
    system_message="You are a really helpful assistant who helps on the task given."
)

user_agent = UserProxyAgent(
    name = "UserProxy",
    description="A proxy agent that represents the user",
    input_func=input
)

termination_condition = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat(
    participants=[assistant, user_agent],
    termination_condition=termination_condition
)

stream = team.run_stream(task="Write a four line poem about Malawi")

async def main():
    await Console(stream)

if __name__=="__main__":
    asyncio.run(main())