import os
import json
import requests
import subprocess
import time
import uuid
import warnings
import tempfile

from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

warnings.filterwarnings("ignore")

# ============================
# 1. HARDCODED API KEY
# ============================
GOOGLE_API_KEY = "AIzaSyD-Sf0VgI4Mmgi9GRD0OVb0HNS2FN4ycU4"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("✅ GOOGLE_API_KEY loaded successfully.")

# ============================
# 2. RETRY SETTINGS
# ============================
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# ============================
# 3. PRODUCT LOOKUP TOOL
# ============================
def get_product_info(product_name: str) -> str:
    product_catalog = {
        "iphone 15 pro": "iPhone 15 Pro, $999, Low Stock (8 units), 128GB, Titanium finish",
        "samsung galaxy s24": "Samsung Galaxy S24, $799, In Stock (31 units), 256GB, Phantom Black",
        "dell xps 15": 'Dell XPS 15, $1,299, In Stock (45 units), 15.6" display, 16GB RAM, 512GB SSD',
        "macbook pro 14": 'MacBook Pro 14", $1,999, In Stock (22 units), M3 Pro chip, 18GB RAM, 512GB SSD',
        "sony wh-1000xm5": "Sony WH-1000XM5 Headphones, $399, In Stock (67 units), Noise-canceling, 30hr battery",
        "ipad air": 'iPad Air, $599, In Stock (28 units), 10.9" display, 64GB',
        "lg ultrawide 34": 'LG UltraWide 34" Monitor, $499, Out of Stock, Expected: Next week',
    }

    key = product_name.lower().strip()
    if key in product_catalog:
        return f"Product: {product_catalog[key]}"
    else:
        available = ", ".join([p.title() for p in product_catalog.keys()])
        return f"Sorry, no info for {product_name}. Available products: {available}"


print("✅ Product Info Tool Loaded.")

# ============================
# 4. PRODUCT CATALOG AGENT
# ============================
product_catalog_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="product_catalog_agent",
    description="Provides product info from vendor catalog.",
    instruction="Use get_product_info tool to fetch product data.",
    tools=[get_product_info],
)

print("✅ Product Catalog Agent Created.")

# ============================
# 5. CREATE REMOTE SERVER FILE
# ============================
temp_dir = tempfile.gettempdir()
server_path = os.path.join(temp_dir, "product_catalog_server.py")

server_code = f"""
import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

os.environ['GOOGLE_API_KEY'] = '{GOOGLE_API_KEY}'

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

def get_product_info(product_name: str) -> str:
    product_catalog = {{
        "iphone 15 pro": "iPhone 15 Pro, $999, Low Stock (8 units), 128GB, Titanium finish",
        "samsung galaxy s24": "Samsung Galaxy S24, $799, In Stock (31 units), 256GB, Phantom Black",
        "dell xps 15": "Dell XPS 15, $1,299, In Stock (45 units), 15.6\\" display, 16GB RAM, 512GB SSD",
        "macbook pro 14": "MacBook Pro 14\\", $1,999, In Stock (22 units), M3 Pro chip, 18GB RAM, 512GB SSD",
        "sony wh-1000xm5": "Sony WH-1000XM5 Headphones, $399, In Stock (67 units), Noise-canceling, 30hr battery",
        "ipad air": "iPad Air, $599, In Stock (28 units), 10.9\\" display, 64GB",
        "lg ultrawide 34": "LG UltraWide 34\\" Monitor, $499, Out of Stock, Expected: Next week",
    }}

    key = product_name.lower().strip()
    if key in product_catalog:
        return f"Product: {{product_catalog[key]}}"
    else:
        available = ", ".join([p.title() for p in product_catalog.keys()])
        return f"Sorry, no info for {{product_name}}. Available: {{available}}"

product_catalog_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="product_catalog_agent",
    instruction="Use get_product_info tool.",
    tools=[get_product_info],
)

app = to_a2a(product_catalog_agent, port=8001)
"""

with open(server_path, "w") as f:
    f.write(server_code)

print(f"📝 Server file created at: {server_path}")

# ============================
# 6. START UVICORN SERVER
# ============================
print("🚀 Starting Product Catalog A2A Server...")

server_process = subprocess.Popen(
    ["uvicorn", "product_catalog_server:app", "--host", "localhost", "--port", "8001"],
    cwd=temp_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=os.environ,
)

# Wait for readiness
for _ in range(30):
    try:
        r = requests.get(f"http://localhost:8001/.well-known/agent-card.json", timeout=2)
        if r.status_code == 200:
            print("✅ Product Catalog Server Running!")
            break
    except:
        time.sleep(1)

# ============================
# 7. CONNECT REMOTE AGENT
# ============================
remote_product_catalog_agent = RemoteA2aAgent(
    name="product_catalog_agent",
    description="Remote product catalog agent",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("✅ Remote Agent Connected!")

# ============================
# 8. CUSTOMER SUPPORT AGENT
# ============================
customer_support_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="customer_support_agent",
    description="Customer support assistant",
    instruction="Use product_catalog_agent to fetch product info.",
    sub_agents=[remote_product_catalog_agent],
)

print("✅ Customer Support Agent Ready.")

# ============================
# 9. TEST FUNCTION (UPDATED)
# ============================
async def test_a2a_communication(query: str):
    session_service = InMemorySessionService()
    session_id = f"session_{uuid.uuid4().hex[:6]}"

    await session_service.create_session(
        app_name="support_app",
        user_id="demo_user",
        session_id=session_id
    )

    runner = Runner(
        agent=customer_support_agent,
        app_name="support_app",
        session_service=session_service
    )

    print(f"\n👤 User: {query}")
    print("🤖 Support Agent:")

    msg = types.Content(parts=[types.Part(text=query)])

    async for event in runner.run_async(
        user_id="demo_user",
        session_id=session_id,
        new_message=msg
    ):
        if event.is_final_response():

            for part in event.content.parts:

                # 1️⃣ Print final AI text
                if hasattr(part, "text") and part.text:
                    print(part.text)

                # 2️⃣ Print tool call request
                if hasattr(part, "function_call") and part.function_call:
                    print("\n🔧 Function Call:")
                    print(f"  Name: {part.function_call.name}")
                    print(f"  Args: {part.function_call.args}")

                # 3️⃣ Print tool call result
                if hasattr(part, "function_response") and part.function_response:
                    print("\n📦 Sub-Agent Response:")
                    print(part.function_response.response)

# ============================
# 10. RUN TESTS
# ============================
import asyncio

print("\n🧪 Running Tests...\n")
asyncio.run(test_a2a_communication("Tell me about the iPhone 15 Pro"))
asyncio.run(test_a2a_communication("Compare Dell XPS 15 and MacBook Pro 14"))
asyncio.run(test_a2a_communication("Do you have Sony WH-1000XM5?"))
