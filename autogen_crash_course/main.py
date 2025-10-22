from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_core import Image as AGImage
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.ui import Console



from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
import asyncio

import requests

from PIL import Image
from io import BytesIO

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)

async def web_search(query: str) -> str:
    '''find information on the web'''
    return "The Labrador Retriever or simply Labrador is a British breed of retriever gun dog."

agent = AssistantAgent(
    name='assistant', 
    model_client=model_client,
    description="A basic first agent",
    tools=[web_search],
    system_message="use tools to solve tasks"
)

agent2 = AssistantAgent(
    name='text_agent',
    model_client=model_client,
    system_message="You are a helpful agent answer all queries accurately"
)

async def test_multimodal():
    response = requests.get("https://picsum.photos/id/237/200/300")
    pil_image = Image.open(BytesIO(response.content))
    ag_image = AGImage(pil_image)

    multi_modal_message = MultiModalMessage(
        content=['What is in the image?', ag_image],
        source="User"
    )

    result = await agent2.run(task=multi_modal_message)
    print(result.messages[-1].content)

async def assistant_run()-> None:
    response = await agent.on_messages(
        messages = [TextMessage(content="Find information about the Labrador Retriever", source='User')],
        cancellation_token=CancellationToken()
    )

    print(response.inner_messages)
    print("\n\n\n\n")
    print(response.chat_message)

async def assistant_run_stream() -> None:
    await Console(
        agent.on_messages_stream(
            messages=[TextMessage(content="Find information about the Labrador Retriever", source="User")],
            cancellation_token=CancellationToken()
        )
    ),
    output_stats=True


#------------------------------------------------------------------------------------------------------------
#                STRUCTURED OUTPUT
#-------------------------------------------------------------------------------------------------------------

from pydantic import BaseModel

class PlanetInfo(BaseModel):
    name: str
    color: str
    distance_miles: int

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=api_key,
    response_format=PlanetInfo
)

unstructured_model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=api_key,
    #response_format=PlanetInfo
)

agent3 = AssistantAgent(
    name="planet_agent",
    model_client=unstructured_model_client,
    system_message="You are a helpful assistant that provides information about planets in the structure JSON" \
    "{name: str" \
    "age: int" \
    "}"
)

agent4 = AssistantAgent(
    name="planet_agent",
    model_client=model_client,
    system_message="You are a helpful assistant that provides information about planets. in the structure JSON"
)

async def test_structured_output():
    task= TextMessage(content="Please provide information about Mars.", source="User")
    result = await agent4.run(task=task)
    structured_response = result.messages[-1].content
    print(structured_response)

#-------------------------------------------------------------------------------------------------------------------
#                           TEAMS   (MULTIPLE AGENTS)
#-------------------------------------------------------------------------------------------------------------------

plot_agent = AssistantAgent(
    name = "plot_writer",
    model_client=unstructured_model_client,
    system_message="You create engaging plots for stories. Focus on the Pokemon's journey."
)

character_agent = AssistantAgent(
    name = "character_writer",
    model_client=unstructured_model_client,
    system_message="You develop characters. Describe the Pokemon and villain in detail, including their motivations and background."
)

ending_agent = AssistantAgent(
    name = "ending_writer",
    model_client=unstructured_model_client,
    system_message="You write engaging endings. Conclude the story with a twist."
)

#------------------------------- Round Robin Configuration of Teams -------------------------------------------------

from autogen_agentchat.teams import RoundRobinGroupChat

team = RoundRobinGroupChat(
    participants=[plot_agent, character_agent, ending_agent],
    max_turns=3
)

async def test_team():
    task = TextMessage(
        content="Write a short story about a brave boy and his Pokemon. Keep it less than 50 words.",
        source="User"
    )
    result = await team.run(task=task)

    for each_agent_message in result.messages:
        print(f"**{each_agent_message.source}**: {each_agent_message.content}\n\n")



from autogen_agentchat.base import TaskResult

#-----------------------------------------------------------------------------------------------------
#                              MULTIPLE AGENTS WITH TERMINATING CONDITION
#-----------------------------------------------------------------------------------------------------

from autogen_agentchat.conditions import MaxMessageTermination

termination_condition = MaxMessageTermination(1)

team2 = RoundRobinGroupChat(
    participants=[plot_agent, character_agent, ending_agent],
    termination_condition=termination_condition
)

#--------------------------- CHAINING TERMINATING CONDITIONS ----------------------------------------

from autogen_agentchat.conditions import TextMentionTermination

termination_condition2 = MaxMessageTermination(10) | TextMentionTermination("APPROVE")

review_agent = AssistantAgent(
    name="review_writer",
    model_client=model_client,
    system_message="You have to say 'APPROVE' once the story is complete and you like the full flow,"\
    "provide your feedback"
)


team3 = RoundRobinGroupChat(
    participants=[plot_agent, character_agent, ending_agent, review_agent],
    termination_condition=termination_condition2
)

async def main():
    #result = await agent.run(task="Find information about the Labrador Retriever.")
    #print(result.messages[-1].content)
    #await assistant_run_stream()
    #await test_structured_output()
    #await test_team()
    await team3.reset()

    async for message in team3.run_stream(task="Write a short poem about the fall season."):
        if isinstance(message, TaskResult):
            print("Stop Reason", message.stop_reason)
        else:
            print(message.content)


if __name__=="__main__":
    asyncio.run(main())