"""Scratchpad tools for CopilotAgent reasoning visibility.

Provides lightweight in-memory tools that let the agent explicitly record its
thinking, planning, and reflection during task execution. When a tool whose
name starts with ``"scratchpad_"`` fires, the host application can surface the
content as "reasoning" in the UI rather than as a regular tool call.

Usage::

    from ai_infra.llm.agents.copilot import create_scratchpad_tools

    tools = create_scratchpad_tools()
    agent = CopilotAgent(tools=tools + my_other_tools, ...)

The returned list contains three tools:

``scratchpad_think``
    Record free-form reasoning, analysis, or observations.
``scratchpad_plan``
    Lay out a numbered plan or next steps.
``scratchpad_reflect``
    Look back at progress, note corrections, or reassess approach.

All three share a single in-memory buffer. The agent can also read the
full scratchpad to recall earlier reasoning via ``scratchpad_read``.

The ``SCRATCHPAD_TOOL_NAMES`` set is exported so callers can check whether
an event's tool name belongs to the scratchpad system.
"""

from __future__ import annotations

from ai_infra.llm.agents.copilot._tools import _CopilotTool

SCRATCHPAD_TOOL_NAMES: frozenset[str] = frozenset(
    {"scratchpad_think", "scratchpad_plan", "scratchpad_reflect", "scratchpad_read"}
)


def create_scratchpad_tools() -> list[_CopilotTool]:
    """Return a list of scratchpad tools with shared in-memory state.

    Each call creates a fresh buffer, so concurrent agent sessions get
    independent scratchpads.
    """
    buffer: list[str] = []

    def scratchpad_think(thought: str) -> str:
        """Record your reasoning, analysis, or observations about the current task."""
        buffer.append(f"[think] {thought}")
        return "Recorded."

    def scratchpad_plan(plan: str) -> str:
        """Lay out your plan or next steps for completing the task."""
        buffer.append(f"[plan] {plan}")
        return "Plan recorded."

    def scratchpad_reflect(reflection: str) -> str:
        """Reflect on progress so far — note corrections or reassess approach."""
        buffer.append(f"[reflect] {reflection}")
        return "Reflection recorded."

    def scratchpad_read() -> str:
        """Read back all scratchpad entries recorded during this session."""
        if not buffer:
            return "(scratchpad is empty)"
        return "\n---\n".join(buffer)

    return [
        _CopilotTool(
            fn=scratchpad_think,
            name="scratchpad_think",
            description=(
                "Record your reasoning and analysis about the current task. "
                "Use this to think through problems step by step before acting. "
                "Content is visible to the user as your thinking process."
            ),
            skip_permission=True,
        ),
        _CopilotTool(
            fn=scratchpad_plan,
            name="scratchpad_plan",
            description=(
                "Write out your plan or next steps for completing the task. "
                "Use this before starting complex multi-step work. "
                "Content is visible to the user."
            ),
            skip_permission=True,
        ),
        _CopilotTool(
            fn=scratchpad_reflect,
            name="scratchpad_reflect",
            description=(
                "Reflect on your progress so far. Note any corrections, "
                "reassess your approach, or summarize findings. "
                "Content is visible to the user."
            ),
            skip_permission=True,
        ),
        _CopilotTool(
            fn=scratchpad_read,
            name="scratchpad_read",
            description=(
                "Read back all your earlier scratchpad entries from this session. "
                "Use this to recall your previous reasoning and plans."
            ),
            skip_permission=True,
        ),
    ]
