"""CopilotAgent — autonomous task execution with streaming.

Demonstrates:
 - Basic run (non-streaming)
 - Streaming with CopilotEvent routing
 - Custom tools via @copilot_tool
 - Scratchpad tools for visible reasoning
 - Permission modes
 - Multi-turn sessions
 - BYOK provider configuration
"""

from __future__ import annotations

import asyncio
import os

from ai_infra import (
    CopilotAgent,
    CopilotEvent,
    CopilotResult,
    PermissionMode,
    copilot_tool,
)
from ai_infra.llm.agents.copilot import (
    SCRATCHPAD_TOOL_NAMES,
    create_scratchpad_tools,
)


# ---------------------------------------------------------------------------
# Custom tool
# ---------------------------------------------------------------------------
@copilot_tool
async def lookup_ticket(ticket_id: str) -> str:
    """Fetch details for a support ticket."""
    return f"Ticket {ticket_id}: customer reports login timeout after 30s"


# ---------------------------------------------------------------------------
# 1. Basic run (non-streaming)
# ---------------------------------------------------------------------------
async def basic_run() -> None:
    agent = CopilotAgent(
        cwd=".",
        tools=[lookup_ticket],
        permissions=PermissionMode.READ_ONLY,
    )
    result: CopilotResult = await agent.run("Summarise the codebase structure in this directory")
    print(result.content)
    print(f"Tools called: {result.tools_called}, took {result.duration_ms:.0f}ms")


# ---------------------------------------------------------------------------
# 2. Streaming with event routing
# ---------------------------------------------------------------------------
async def streaming_demo() -> None:
    scratchpad_tools = create_scratchpad_tools()
    agent = CopilotAgent(
        cwd=".",
        tools=[lookup_ticket, *scratchpad_tools],
    )

    async for event in agent.stream("Fix the bug described in ticket TK-42"):
        _route_event(event)


def _route_event(event: CopilotEvent) -> None:
    """Route CopilotEvent to the appropriate UI handler."""
    if event.type == "token":
        # Final answer text — display in the response area.
        print(event.content, end="", flush=True)

    elif event.type in ("reasoning", "reasoning_delta"):
        # Model's chain-of-thought — show in a collapsible scratchpad.
        print(f"  [reasoning] {event.content}")

    elif event.type == "intent":
        # Agent describes what it is about to do.
        print(f"\n  Intent: {event.content}")

    elif event.type == "tool_start":
        if event.tool in SCRATCHPAD_TOOL_NAMES:
            # Scratchpad tools are internal thinking — route to scratchpad.
            print(f"  [scratchpad:{event.tool}] {event.arguments}")
        else:
            print(f"\n  -> {event.tool}({event.arguments})")

    elif event.type == "tool_output":
        print(f"     {event.content[:120]}")

    elif event.type == "tool_end":
        if event.tool not in SCRATCHPAD_TOOL_NAMES:
            print(f"     [{event.latency_ms:.0f}ms]")

    elif event.type == "todo":
        for item in event.todo_items:
            status = item.get("status", "not-started")
            print(f"  [{status}] {item.get('title', '')}")

    elif event.type == "subagent":
        print(f"  [sub-agent:{event.subagent_phase}] {event.subagent_name}")

    elif event.type == "task_complete":
        print(f"\n  Task complete: {event.content}")

    elif event.type == "error":
        print(f"\n  ERROR: {event.error}")

    elif event.type == "done":
        print("\n--- done ---")


# ---------------------------------------------------------------------------
# 3. Multi-turn session
# ---------------------------------------------------------------------------
async def multi_turn() -> None:
    agent = CopilotAgent(cwd=".")
    r1 = await agent.run("Explore the src/ directory and list all modules")
    r2 = await agent.run(
        "Now add type hints to the auth module",
        session_id=r1.session_id,
    )
    print(r2.content)


# ---------------------------------------------------------------------------
# 4. BYOK with non-GitHub model
# ---------------------------------------------------------------------------
async def byok_demo() -> None:
    agent = CopilotAgent(
        model="gpt-4.1",
        provider={
            "type": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": os.environ["OPENAI_API_KEY"],
        },
        cwd=".",
    )
    result = await agent.run("What files are in this project?")
    print(result.content)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(streaming_demo())
