import asyncio
import os
import uuid

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from mcp import StdioServerParameters

# --- 1. Setup (API Key Check & Retry) ---

# if "GOOGLE_API_KEY" not in os.environ:
#     print("🔑 Authentication Error: Please set 'GOOGLE_API_KEY' environment variable.")
#     exit()

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

MODEL = "gemini-2.5-flash-lite"

print("✅ Setup and authentication complete.")

# --- 2. Tool Definitions (The Exercise) ---

# TOOL 1: The MCP Server (from Section 2)
# This tool provides the actual image generation (getTinyImage)
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)
print("✅ MCP Tool defined (will launch npx server on first use).")


# TOOL 2: The LRO Approval Tool (from Section 3, adapted for the exercise)
# This tool *only* handles the approval logic.
def request_bulk_approval(num_images: int, tool_context: ToolContext) -> dict:
    """
    Checks if a bulk image request requires approval.
    Auto-approves 1 image, but pauses for >1.

    Args:
        num_images: The number of images requested.
    Returns:
        Dictionary with approval status.
    """
    # SCENARIO 1: Small order (1 image) auto-approves
    if num_images <= 1:
        return {
            "status": "approved",
            "message": f"Auto-approved for {num_images} image.",
        }

    # SCENARIO 2: Large order (>1) - FIRST CALL (needs approval)
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Bulk Order: {num_images} images requested. Approve cost?",
            payload={"num_images": num_images},
        )
        return {
            "status": "pending",
            "message": f"Order for {num_images} images requires approval.",
        }

    # SCENARIO 3: Large order - RESUMED CALL (handle human decision)
    if tool_context.tool_confirmation.confirmed:
        return {"status": "approved", "message": f"Human approved {num_images} images."}
    else:
        return {"status": "rejected", "message": f"Human rejected {num_images} images."}


print("✅ LRO Approval Tool defined.")

# --- 3. The Agent (The Exercise) ---

exercise_agent = LlmAgent(
    name="image_agent",
    model=Gemini(model=MODEL, retry_options=retry_config),
    instruction="""You are an image generation assistant.
    You have two tools: `getTinyImage` (from mcp_image_server) and `request_bulk_approval`.

    **CRITICAL WORKFLOW:**
    1.  When the user asks for images, you MUST determine the `num_images`.
    2.  If `num_images` == 1, you MUST call `getTinyImage` directly.
    3.  If `num_images` > 1, you MUST FIRST call `request_bulk_approval` with the `num_images`.
    4.  **NEVER** call `getTinyImage` for a bulk order unless `request_bulk_approval` has returned status 'approved'.
    5.  If 'pending', inform the user.
    6.  If 'rejected', inform the user.
    7.  If 'approved', you may THEN call `getTinyImage`.
    8.  When `getTinyImage` succeeds, just tell the user "Here is your image." (don't show the base64 data).
    """,
    tools=[
        mcp_image_server,  # The MCP tool
        FunctionTool(func=request_bulk_approval),  # The LRO tool
    ],
)

# --- 4. The Resumable App & Runner (from Section 3) ---
# This is required for the LRO tool to pause and resume state.

exercise_app = App(
    name="image_approver_app",
    root_agent=exercise_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

session_service = InMemorySessionService()
exercise_runner = Runner(
    app=exercise_app,  # Pass the app, not the agent
    session_service=session_service,
)
print("✅ Resumable App and Runner created.")

# --- 5. The Workflow Code (from Section 4) ---
# These functions handle detecting the "pause" event and resuming.


def check_for_approval(events):
    """Check if events contain an approval request."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None


def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"🤖 Agent > {part.text}")


def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )


async def run_image_workflow(query: str, auto_approve: bool = True):
    """Runs the full image workflow, simulating a human decision."""
    print(f"\n{'=' * 60}")
    print(f"👤 User > {query}\n")
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name="image_approver_app", user_id="local_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    # STEP 1: Send initial request
    async for event in exercise_runner.run_async(
        user_id="local_user", session_id=session_id, new_message=query_content
    ):
        events.append(event)

    # STEP 2: Check if the agent paused for approval
    approval_info = check_for_approval(events)

    # STEP 3: Handle the result
    if approval_info:
        # PATH A: Agent paused, needs human input
        print(f"⏸️  Workflow Paused: Agent is waiting for human approval.")
        decision = "APPROVE ✅" if auto_approve else "REJECT ❌"
        print(f"🤔 Human Decision (Simulated): {decision}\n")

        # Resume the agent, sending the decision
        async for event in exercise_runner.run_async(
            user_id="local_user",
            session_id=session_id,
            new_message=create_approval_response(approval_info, auto_approve),
            invocation_id=approval_info["invocation_id"],  # CRITICAL: Resumes
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"🤖 Agent > {part.text}")
    else:
        # PATH B: No approval needed, print final response
        print_agent_response(events)

    print(f"{'=' * 60}")


# --- 6. Main function to run the demos ---


async def main():
    print("🚀 Starting Exercise Demos...")

    print("Loading API key from .env file...")
    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("Authentication Error: 'GOOGLE_API_KEY' not found")
        print("please make sure you have a .env file with your key")
        return

    # Demo 1: Small order (1 image) -> Should auto-approve and call getTinyImage
    await run_image_workflow("Please get me 1 tiny image of a cat.")

    # Demo 2: Large order (5 images) -> Should pause, then APPROVE
    await run_image_workflow(
        "I need a bulk order of 5 tiny images for my project.", auto_approve=True
    )

    # Demo 3: Large order (10 images) -> Should pause, then REJECT
    await run_image_workflow("Generate 10 tiny images for me.", auto_approve=False)

    print("\n✅ All demos complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
