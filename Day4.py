import os
import logging
import asyncio
from typing import List

# --- Imports from ADK ---
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

# --- 1. Set up Google API Key ---
GOOGLE_API_KEY = "GOOGLE_API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
    print("⚠️ WARNING: Please paste your GOOGLE_API_KEY into the script.")
else:
    print("✅ API Key configured.")

# --- 2. Configure Logging ---
logging.basicConfig(
    filename="logger.log", 
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
    filemode="w",  
)
print("✅ Logging configured. Output will be in 'logger.log'")

# --- 3. Custom Plugin (from your snippet) ---
class CountInvocationPlugin(BasePlugin):
    """A custom plugin that counts agent and tool invocations."""

    def __init__(self) -> None:
        """Initialize the plugin with counters."""
        super().__init__(name="count_invocation")
        self.agent_count: int = 0
        self.tool_count: int = 0
        self.llm_request_count: int = 0

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Count agent runs."""
        self.agent_count += 1
        logging.info(f"[Plugin] Agent run count: {self.agent_count}")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Count LLM requests."""
        self.llm_request_count += 1
        logging.info(f"[Plugin] LLM request count: {self.llm_request_count}")


# --- 4. Define Agent Logic ---

# Define retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Define a custom tool
def count_papers(papers: List[str]):
    """
    This function counts the number of papers in a list of strings.
    Args:
      papers: A list of strings, where each string is a research paper.
    Returns:
      The number of papers in the list.
    """
    print(f"Tool 'count_papers' received {len(papers)} items.")
    logging.info(f"Tool 'count_papers' received {len(papers)} items.")
    return len(papers)


# Google search sub-agent
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for information using Google search",
    instruction="Use the google_search tool to find information on the given topic. Return the raw search results.",
    tools=[google_search],
)

# Root agent
research_agent_with_plugin = LlmAgent(
    name="research_paper_finder_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Your task is to find research papers and count them. 
   
   You must follow these steps:
   1) Find research papers on the user provided topic using the 'google_search_agent'. 
   2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
   3) Return both the list of research papers and the total number of papers.
   """,
    tools=[AgentTool(agent=google_search_agent), count_papers],
)

print("✅ Agents and tools defined.")


# --- 5. Main function to configure and run the agent ---
async def main():
    # Configure the runner with the root agent and plugins
    runner = InMemoryRunner(
        agent=research_agent_with_plugin,
        plugins=[
            LoggingPlugin(),  # Handles standard Observability logging
            CountInvocationPlugin(),  
        ],
    )
    print("✅ Runner configured with LoggingPlugin and CountInvocationPlugin.")

    print("\nRunning agent... (This may take a moment)")
    print("--------------------------------------------------")

    response = await runner.run_debug("Find recent papers on quantum computing")

    print("--------------------------------------------------")
    print("✅ Agent run complete.")
    print("\nFinal Response from Agent:")
    print(response)

    print("\nCheck the 'logger.log' file for detailed DEBUG and plugin logs.")


# --- 6. Running the main async function ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        logging.error(f"An error occurred: {e}", exc_info=True)