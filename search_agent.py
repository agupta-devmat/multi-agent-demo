import logging
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException
from strands import Agent, tool
from strands.models import BedrockModel
from strands.multiagent.a2a import A2AServer

logging.getLogger("strands").setLevel(logging.INFO)

from dotenv import load_dotenv

load_dotenv(override=True)  # take environment variables

@tool
def websearch(
    keywords: str, region: str = "us-en", max_results: int | None = None
) -> str:
    """Search the web to get updated information.
    Args:
        keywords (str): The search query keywords.
        region (str): The search region: wt-wt, us-en, uk-en, ru-ru, etc..
        max_results (int | None): The maximum number of results to return.
    Returns:
        List of dictionaries with search results.
    """
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)
        return results if results else "No results found."
    except RatelimitException:
        return "RatelimitException: Please try again after a short delay."
    except DDGSException as d:
        return f"DuckDuckGoSearchException: {d}"
    except Exception as e:
        return f"Exception: {e}"


bedrock_model = BedrockModel(
    model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0"
)

search_agent = Agent(
    system_prompt="""You are a helpful assistant.
    Help users to find information based on their queries and answer questions.
    Use the websearch tool to find information.""",
    description="An agent that can search the web to answer questions.",
    tools=[websearch],
    model=bedrock_model,
)

a2a_server = A2AServer(
    agent=search_agent,
    http_url="http://localhost:5001"
)
a2a_server.serve(host="0.0.0.0", port=5001)