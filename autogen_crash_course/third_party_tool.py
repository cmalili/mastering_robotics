import asyncio

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

from autogen_core.tools import FunctionTool

from autogen_ext.models.openai import OpenAIChatCompletionClient

from langchain_community.utilities import GoogleSerperAPIWrapper

#from autogen_ext.ui import Console
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=api_key
)

serper_api_key = os.getenv("SERPER_API_KEY")

search_tool_wrapper = GoogleSerperAPIWrapper(type="news")

def search_web(query: str) -> str:
    """Search web using Serper API"""
    try:
        return search_tool_wrapper.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"
    
search_agent = AssistantAgent(
    name="SearchAgent",
    description="You are a helpful assistant that can search the web to find current information",
    model_client=model_client,
    system_message="You are a helpful search agent that can search the web."\
    "When asked a question, use web_search tool to find relevant information "\
    "and provide a comprehensive answer based on the search results.",
    tools=[search_web],
    reflect_on_tool_use=True
)

async def demonstrate_search():
    """Demonstrate the search functionality"""
    print("==== Autogen Third Party Tools Demonstration ====\n")

    test_queries = [
        "Who won the last IPL tournament in cricket in 2025?"
    ]

    for query in test_queries:
        print(f"Query: {query}\n")
        print("-"*50)

        try:
            result = await search_agent.run(task=query)
            print(f"Response: {result.messages[-1].content}")

        except Exception as e:
            print(f"Error: {str(e)}")

        print("\n" + "="*70 + "\n")

async def main():
    await demonstrate_search()


if __name__=="__main__":
    asyncio.run(main())

