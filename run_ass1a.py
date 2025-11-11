import os
import asyncio
from dotenv import load_dotenv

# import ADK components
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

async def main():
    """
    Main async function to run our agent
    """

    # --- 1. Load Configuration ---
    print("Loading API Key...")
    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    
    # This was in the notebook, so we'll keep it
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("Config loaded successfully.")


    # ----2. Define agent ------
    root_agent = Agent(
        name="helpful_assistant",
        model="gemini-2.5-flash-lite",
        description="A simple agent that can answer general questions",
        instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
        tools=[google_search],
    )
    print("Agent defined successfully.")

    # ----3. run agent ------
    runner = InMemoryRunner(agent=root_agent)
    print("Starting agent run...")

    # first question
    response = await runner.run_debug(
        "What is Agent Development Kit from Google? What languages is the SDK available in?"
    )
    print("Response to first question:")
    print(response)

    # second question
    print("\nAsking second question... ")
    response_weather = await runner.run_debug(
        "What is the weather in New York City today?"
    )
    print("Response to second question:")
    print(response_weather)

if __name__ == "__main__":
    asyncio.run(main())



