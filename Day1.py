import os
import asyncio
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

GOOGLE_API_KEY = "AIzaSyD-Sf0VgI4Mmgi9GRD0OVb0HNS2FN4ycU4"

# Seting up the key for the SDK
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("✅ Gemini API key setup completed.")

print("✅ ADK components imported successfully.")

# Creating the root agent
root_agent = Agent(
    name="helpful_assistant",
    model="gemini-2.5-flash-lite",
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

print("✅ Root Agent defined.")

async def main():
    """Asynchronously runs the agent and prints the response."""
    runner = InMemoryRunner(agent=root_agent)
    print("✅ Runner created.")
    response = await runner.run_debug(
        "What is Agent Development Kit from Google? What languages is the SDK available in?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())

#https://www.kaggle.com/code/kaggle5daysofai/day-1a-from-prompt-to-action