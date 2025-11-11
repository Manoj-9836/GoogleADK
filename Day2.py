import os
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.adk.code_executors import BuiltInCodeExecutor

# ---------------------- GOOGLE API KEY ----------------------
GOOGLE_API_KEY = "GOOGLE_API_KEY" 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("Google API Key setup completed.")

# ---------------------- HELPER FUNCTION ----------------------
def show_python_code_and_result(response):
    """Helper to show Python code and results from LLM responses."""
    for i in range(len(response)):
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])

print("Helper functions defined.")

# ---------------------- RETRY CONFIG ----------------------
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# ---------------------- TOOL 1: PAYMENT FEE LOOKUP ----------------------
def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method."""
    fee_database = {
        "platinum credit card": 0.02,  
        "gold debit card": 0.035,      
        "bank transfer": 0.01,         
    }

    fee = fee_database.get(method.lower())
    if fee is not None:
        return {"status": "success", "fee_percentage": fee}
    else:
        return {"status": "error", "error_message": f"Payment method '{method}' not found"}

print("-------------Fee lookup function created-------------")
print(f"Test: {get_fee_for_payment_method('platinum credit card')}")

# ---------------------- TOOL 2: EXCHANGE RATE LOOKUP ----------------------
def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies."""
    rate_database = {
        "usd": {"eur": 0.93, "jpy": 157.50, "inr": 83.58}
    }

    base = base_currency.lower()
    target = target_currency.lower()
    rate = rate_database.get(base, {}).get(target)
    if rate is not None:
        return {"status": "success", "rate": rate}
    else:
        return {
            "status": "error",
            "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}",
        }

print("-------------Exchange rate function created-------------")
print(f"Test: {get_exchange_rate('USD', 'EUR')}")

# ---------------------- CURRENCY AGENT ----------------------
currency_agent = LlmAgent(
    name="currency_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a smart currency conversion assistant.

    For currency conversion requests:
    1. Use `get_fee_for_payment_method()` to find transaction fees
    2. Use `get_exchange_rate()` to get currency conversion rates
    3. Check the "status" field in each tool's response for errors
    4. Calculate the final amount after fees based on the output
       from the tools and provide a clear breakdown.
    5. First, state the final converted amount.
       Then explain how you got that result by showing intermediate amounts:
       fee percentage, value in original currency, amount after fee,
       and exchange rate used.
    """,
    tools=[get_fee_for_payment_method, get_exchange_rate],
)

print("Currency agent created")

# ---------------------- RUNNER TEST ----------------------
currency_runner = InMemoryRunner(agent=currency_agent)

async def test_currency_agent():
    await currency_runner.run_debug(
        "I want to convert 500 US Dollars to Euros using my Platinum Credit Card. How much will I receive?"
    )

# ---------------------- CALCULATION AGENT ----------------------
calculation_agent = LlmAgent(
    name="CalculationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a specialized calculator that ONLY responds with Python code.
    Do NOT write explanations — only return Python code that calculates the result
    and prints it.
    """,
    code_executor=BuiltInCodeExecutor(),
)

# ---------------------- ENHANCED CURRENCY AGENT ----------------------
enhanced_currency_agent = LlmAgent(
    name="enhanced_currency_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a smart currency conversion assistant. Follow these steps:

    1. Use get_fee_for_payment_method() to find transaction fee.
    2. Use get_exchange_rate() to find exchange rate.
    3. Check the 'status' of each response. If 'error', stop and explain.
    4. You are NOT allowed to do arithmetic yourself.
       Use the calculation_agent tool to generate Python code that performs all math.
    5. Provide detailed breakdown:
       - Final converted amount
       - Fee percentage and amount
       - Amount after fee
       - Exchange rate applied
    """,
    tools=[
        get_fee_for_payment_method,
        get_exchange_rate,
        AgentTool(agent=calculation_agent),
    ],
)

print("Enhanced currency agent created")

# ---------------------- FINAL RUNNER ----------------------
enhanced_runner = InMemoryRunner(agent=enhanced_currency_agent)

async def main():
    print("Running Currency Conversion Example...\n")
    response = await enhanced_runner.run_debug(
        "Convert 1,250 USD to INR using a Bank Transfer. Show me the precise calculation."
    )
    show_python_code_and_result(response)

# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
#https://www.kaggle.com/code/manoj0956/day-2a-agent-tools/edit
