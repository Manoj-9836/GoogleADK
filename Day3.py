import os
import sqlite3
import asyncio
from typing import Any, Dict

# -----------------------------
# STEP 1: Hardcode your Gemini API Key
# -----------------------------
# WARNING: It's safer to use environment variables in production
# (e.g., os.environ.get("GOOGLE_API_KEY"))
# But for this demo script, we'll use the hardcoded key.
GOOGLE_API_KEY = "GEMINI_API_KEY_HERE" 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("✅ Gemini API key setup complete.")

# -----------------------------
# STEP 2: Import Google ADK modules
# -----------------------------
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

print("✅ ADK components imported successfully.")

# -----------------------------
# STEP 3: Define helper function to run sessions
# -----------------------------
async def run_session_async(runner_instance: Runner, user_queries, session_name="default", session_service=None):
    if session_service is None:
        raise ValueError("session_service must be provided")

    async def inner(user_queries_local):
        print(f"\n### Session: {session_name}")
        app_name = runner_instance.app_name

        # Create or retrieve session
        try:
            session = await session_service.create_session(
                app_name=app_name, user_id=USER_ID, session_id=session_name
            )
        except Exception:
            # Catch a generic exception if session already exists
            session = await session_service.get_session(
                app_name=app_name, user_id=USER_ID, session_id=session_name
            )

        # Handle queries
        if isinstance(user_queries_local, str):
            user_queries_local = [user_queries_local]

        for query in user_queries_local:
            print(f"\nUser > {query}")
            query = types.Content(role="user", parts=[types.Part(text=query)])
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text and text != "None":
                        print(f"{MODEL_NAME} > {text}")

    # Await the inner function
    await inner(user_queries)

print("✅ Helper functions defined.")

# -----------------------------
# STEP 4: Setup configurations
# -----------------------------
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

APP_NAME = "default"
USER_ID = "default"
SESSION = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

# -----------------------------
# STEP 9 (Moved up): Tool Definitions
# -----------------------------
def save_userinfo(tool_context: ToolContext, user_name: str, country: str) -> Dict[str, Any]:
    """Saves the user's name and country to the session state."""
    print(f"\n🤖 TOOL: Saving user_name='{user_name}', country='{country}' to state.")
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country
    return {"status": "success", "user_name": user_name, "country": country}

def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """Retrieves the user's name and country from the session state."""
    print("\n🤖 TOOL: Retrieving user info from state.")
    user_name = tool_context.state.get("user:name", "Username not found")
    country = tool_context.state.get("user:country", "Country not found")
    return {"status": "success", "user_name": user_name, "country": country}

print("✅ Tools created.")


# -----------------------------
# STEP 7 (Moved up): Check DB Function
# -----------------------------
def check_data_in_db():
    print("\n--- Checking data in my_agent_data.db ---")
    if not os.path.exists("my_agent_data.db"):
        print("Database file not found.")
        return
        
    with sqlite3.connect("my_agent_data.db") as connection:
        cursor = connection.cursor()
        try:
            result = cursor.execute("select app_name, session_id, author, content from events")
            print([_[0] for _ in result.description])
            for each in result.fetchall():
                print(each)
        except sqlite3.OperationalError as e:
            print(f"Error querying database: {e}")
    print("------------------------------------------")


# ==========================================================
# MAIN ASYNC FUNCTION
# ==========================================================
# This function wraps all the runnable steps
async def main():
    
    # -----------------------------
    # STEP 5: Create a basic stateful agent
    # -----------------------------
    root_agent_step5 = Agent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="text_chat_bot",
        description="A simple text chatbot"
    )
    # Use an in-memory service for this step
    session_service_step5 = InMemorySessionService()
    runner_step5 = Runner(agent=root_agent_step5, app_name=APP_NAME, session_service=session_service_step5)
    print("✅ Stateful agent initialized!")

    await run_session_async(
        runner_step5,
        ["Hi, I am Manoj! What is the capital of India?", "Hello! What is my name?"],
        "stateful-agentic-session",
        session_service_step5
    )
    await run_session_async(
        runner_step5,
        ["What did I ask you about earlier?", "And remind me, what's my name?"],
        "stateful-agentic-session",
        session_service_step5
    )

    # -----------------------------
    # STEP 6: Persistent Sessions using SQLite
    # -----------------------------
    chatbot_agent_step6 = LlmAgent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot with persistent memory"
    )
    db_url = "sqlite:///my_agent_data.db"
    session_service_step6 = DatabaseSessionService(db_url=db_url)
    runner_step6 = Runner(agent=chatbot_agent_step6, app_name=APP_NAME, session_service=session_service_step6)
    print("✅ Upgraded to persistent sessions!")

    await run_session_async(
        runner_step6,
        ["Hi, I am Manoj! What is the capital of India?", "Hello! What is my name?"],
        "test-db-session-01",
        session_service_step6
    )
    await run_session_async(
        runner_step6,
        ["What is the capital of India?", "Hello! What is my name?"],
        "test-db-session-01",
        session_service_step6
    )
    await run_session_async(runner_step6, ["Hello! What is my name?"], "test-db-session-02", session_service_step6)

    # -----------------------------
    # STEP 7: Check data in SQLite
    # -----------------------------
    # Call the synchronous function
    check_data_in_db()
    print("✅ Database check complete.")

    # -----------------------------
    # STEP 10: Agent with Session State Tools
    # -----------------------------
    root_agent_step10 = LlmAgent(
        model=Gemini(model=MODEL_NAME, retry_options=retry_config),
        name="text_chat_bot",
        description="""A text chatbot with memory tools.
        Tools:
        * save_userinfo(user_name, country)
        * retrieve_userinfo()
        """,
        tools=[save_userinfo, retrieve_userinfo],
    )
    # Use a new in-memory service for this step
    session_service_step10 = InMemorySessionService()
    runner_step10 = Runner(agent=root_agent_step10, session_service=session_service_step10, app_name="default")
    print("✅ Agent with session state tools initialized!")

    await run_session_async(
        runner_step10,
        [
            "Hi there, how are you doing today? What is my name?",
            "My name is Manoj. I'm from India.",
            "What is my name? Which country am I from?",
        ],
        "state-demo-session",
        session_service_step10
    )
    
    async def show_state():
        session = await session_service_step10.get_session(app_name=APP_NAME, user_id=USER_ID, session_id="state-demo-session")
        print("\n--- Session State Contents (state-demo-session) ---")
        print(session.state)
        print("----------------------------------------------------")
    
    # Await the inner async function
    await show_state()

    await run_session_async(runner_step10, ["Hi there, how are you doing today? What is my name?"], "new-isolated-session", session_service_step10)

    async def show_new_state():
        session = await session_service_step10.get_session(app_name=APP_NAME, user_id=USER_ID, session_id="new-isolated-session")
        print("\n--- New Session State (new-isolated-session) ---")
        print(session.state)
        print("------------------------------------------------")
    
    # Await the inner async function
    await show_new_state()

    print("\n✅ Demo script finished.")


# ==========================================================
# SCRIPT ENTRY POINT
# ==========================================================
# This runs the 'main' async function one time.
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # Handle a specific shutdown error that can occur in Windows
        if "Event loop is closed" in str(e) and os.name == 'nt':
            pass
        else:
            raise
