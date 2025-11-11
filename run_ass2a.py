import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.genai import types

# --- 1. API Key and Retry Configuration ---

# Check if the API key is set
# if "GOOGLE_API_KEY" not in os.environ:
#     print(
#         "🔑 Authentication Error: Please set the 'GOOGLE_API_KEY' environment variable."
#     )
#     exit()

# Configure retry options (from notebook cell [5])
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --- 2. Helper Function (from notebook cell [4]) ---


def show_python_code_and_result(response):
    """Prints the generated Python code and results from the code executor."""
    print("\n--- 🕵️ Agent's Calculation ---")
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
                        "Generated Python Code >> \n",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])
    print("---------------------------------")


# --- 3. Custom Tool Definitions (from notebook cells [6] & [7]) ---


def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method.

    This tool simulates looking up a company's internal fee structure based on
    the name of the payment method provided by the user.

    Args:
        method: The name of the payment method. It should be descriptive,
            e.g., "platinum credit card" or "bank transfer".

    Returns:
        Dictionary with status and fee information.
        Success: {"status": "success", "fee_percentage": 0.02}
        Error: {"status": "error", "error_message": "Payment method not found"}
    """
    fee_database = {
        "platinum credit card": 0.02,  # 2%
        "gold debit card": 0.035,  # 3.5%
        "bank transfer": 0.01,  # 1%
    }
    fee = fee_database.get(method.lower())
    if fee is not None:
        return {"status": "success", "fee_percentage": fee}
    else:
        return {
            "status": "error",
            "error_message": f"Payment method '{method}' not found",
        }


def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies.

    Args:
        base_currency: The ISO 4217 currency code of the currency you
            are converting from (e.g., "USD").
        target_currency: The ISO 4217 currency code of the currency you
            are converting to (e.g., "EUR").

    Returns:
        Dictionary with status and rate information.
        Success: {"status": "success", "rate": 0.93}
        Error: {"status": "error", "error_message": "Unsupported currency pair"}
    """
    rate_database = {
        "usd": {
            "eur": 0.93,  # Euro
            "jpy": 157.50,  # Japanese Yen
            "inr": 83.58,  # Indian Rupee
        }
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


# --- 4. Agent Definitions (from notebook cells [10] & [11]) ---

# The specialist agent for reliable math
calculation_agent = LlmAgent(
    name="CalculationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a specialized calculator that ONLY responds with Python code.
    You are forbidden from providing any text, explanations, or conversational responses.
    Your task is to take a request for a calculation and translate it into a single
    block of Python code that calculates the answer.
    **RULES:**
    1.  Your output MUST be ONLY a Python code block.
    2.  Do NOT write any text before or after the code block.
    3.  The Python code MUST calculate the result.
    4.  The Python code MUST print the final result to stdout.
    5.  You are PROHIBITED from performing the calculation yourself.
        Your only job is to generate the code that will perform the calculation.
    Failure to follow these rules will result in an error.
    """,
    code_executor=BuiltInCodeExecutor(),
)

# The main "manager" agent
enhanced_currency_agent = LlmAgent(
    name="enhanced_currency_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a smart currency conversion assistant.
    You must strictly follow these steps and use the available tools.
    For any currency conversion request:
    1. Get Transaction Fee: Use the get_fee_for_payment_method() tool.
    2. Get Exchange Rate: Use the get_exchange_rate() tool.
    3. Error Check: After each tool call, you must check the "status" field.
       If "error", stop and explain the issue.
    4. Calculate Final Amount (CRITICAL): You are strictly prohibited from
       performing any arithmetic yourself. You must use the
       calculation_agent tool to generate Python code that calculates the
       final converted amount.
    5. Provide Detailed Breakdown: State the final amount and explain how
       it was calculated, including the fee, the amount after fee,
       and the exchange rate.
    """,
    tools=[
        get_fee_for_payment_method,
        get_exchange_rate,
        AgentTool(agent=calculation_agent),  # Using the other agent as a tool!
    ],
)

print("✅ Agents and tools defined.")

# --- 5. Main function to run the agent ---


async def main():
    """Defines the main asynchronous function to run our agent."""

    print("Loading API key from .env file...")
    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("Authentication Error: 'GOOGLE_API_KEY' not found")
        print("please make sure you have a .env file with your key")
        return
    print("Config loaded.")

    print("🚀 Initializing agent runner...")
    enhanced_runner = InMemoryRunner(agent=enhanced_currency_agent)

    # The prompt from the notebook (cell [13])
    user_prompt = "Convert 1,250 USD to INR using a Bank Transfer. Show me the precise calculation."
    print(f"\n💬 User > {user_prompt}\n")

    # Run the agent and get the full debug response
    response = await enhanced_runner.run_debug(user_prompt)

    # The final answer is the last message in the response
    final_answer = response[-1].content.parts[0].text
    print(f"🤖 Agent > {final_answer}")

    # Use the helper function to show the code (from cell [14])
    show_python_code_and_result(response)


# --- 6. Standard Python entry point ---

if __name__ == "__main__":
    asyncio.run(main())
