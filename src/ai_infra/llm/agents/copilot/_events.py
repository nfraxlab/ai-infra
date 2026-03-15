"""Typed event and result types emitted by CopilotAgent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CopilotEvent:
    """A typed event emitted by ``CopilotAgent.stream()``.

    Designed to be structurally compatible with ai-infra's ``StreamEvent``
    so existing streaming consumers can handle both ``Agent.astream()`` and
    ``CopilotAgent.stream()`` with minimal branching.

    Event types:

    ``token``
        Text chunk from the assistant response. Read ``content``.
    ``tool_start``
        A tool has started executing. Read ``tool`` and ``arguments``.
    ``tool_output``
        Partial stdout from a running tool (e.g. bash streaming output).
        Read ``tool`` and ``content``.
    ``tool_end``
        Tool execution completed. Read ``tool``, ``result``, and ``latency_ms``.
    ``intent``
        Short description of what the agent is currently doing.
        Read ``content`` (e.g. ``"Exploring codebase"``).
    ``context``
        Working directory or git context changed. Read ``cwd`` and ``branch``.
    ``reasoning``
        Complete chain-of-thought block from a model that supports reasoning
        (e.g. Claude with extended thinking). Read ``content`` and ``reasoning_id``.
    ``reasoning_delta``
        Streaming reasoning chunk (incremental). Read ``content`` and
        ``reasoning_id``. Accumulate across events to build the full block.
    ``subagent``
        Sub-agent lifecycle event when ``custom_agents`` are configured.
        Read ``subagent_name`` and ``subagent_phase``
        (one of ``"selected"``, ``"started"``, ``"completed"``,
        ``"failed"``, ``"deselected"``).
    ``compaction``
        Context compaction event (fired when infinite sessions compress the
        context window). Read ``compaction_phase`` (``"start"`` or
        ``"complete"``) and ``tokens_removed`` / ``content`` (summary).
    ``usage``
        Per-LLM-call token accounting. Read ``input_tokens``,
        ``output_tokens``, ``cost``, and ``content`` (model identifier).
    ``turn_start``
        The agent has begun a new reasoning turn. Read ``turn_id``.
    ``turn_end``
        The agent has finished a reasoning turn. Read ``turn_id``.
    ``task_complete``
        The agent considers its assigned task fully complete. Read
        ``content`` for the task summary.
    ``done``
        Session turn fully complete (all tools finished, agent idle).
        ``content`` holds the actual session ID assigned by the CLI.
    ``error``
        An error occurred during the run. Read ``error``.

    Attributes:
        type: Event discriminator (see above).
        content: Token text (``token``), partial output (``tool_output``),
            intent string (``intent``), reasoning text (``reasoning`` /
            ``reasoning_delta``), compaction summary (``compaction``),
            or session ID (``done``).
        tool: Tool name for tool-related events.
        arguments: Tool arguments dict (``tool_start``).
        result: Tool result string (``tool_end``).
        cwd: Current working directory (``context``).
        branch: Git branch name (``context``).
        error: Error message (``error``).
        latency_ms: Execution time in milliseconds (``tool_end``).
        timestamp: Unix epoch when the event was emitted.
        reasoning_id: Reasoning block identifier (``reasoning`` /
            ``reasoning_delta``).
        subagent_name: Sub-agent name (``subagent`` events).
        subagent_phase: Sub-agent lifecycle phase (``subagent`` events).
        tokens_removed: Tokens removed by compaction (``compaction``
            with ``compaction_phase="complete"``).
        compaction_phase: ``"start"`` or ``"complete"`` (``compaction`` events).
        input_tokens: Input tokens consumed (``usage`` events).
        output_tokens: Output tokens produced (``usage`` events).
        cost: Model multiplier cost (``usage`` events from billing).
        turn_id: Turn identifier (``turn_start`` / ``turn_end`` events).
    """

    type: Literal[
        "token",
        "tool_start",
        "tool_output",
        "tool_end",
        "intent",
        "context",
        "done",
        "error",
        "reasoning",
        "reasoning_delta",
        "subagent",
        "compaction",
        "usage",
        "turn_start",
        "turn_end",
        "task_complete",
    ]
    content: str = ""
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    cwd: str = ""
    branch: str = ""
    error: str = ""
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    # reasoning events
    reasoning_id: str = ""
    # subagent events
    subagent_name: str = ""
    subagent_phase: str = ""
    # compaction events
    tokens_removed: int = 0
    compaction_phase: str = ""
    # usage events (per-LLM-call token accounting)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    # turn events
    turn_id: str = ""


@dataclass
class CopilotResult:
    """Result returned by ``CopilotAgent.run()``.

    Attributes:
        content: The assistant's final text response.
        session_id: Session identifier. Pass this to a subsequent ``run()``
            or ``stream()`` call to resume the conversation.
        tools_called: Number of tool invocations during the run.
        duration_ms: Total wall-clock time in milliseconds.
    """

    content: str
    session_id: str
    tools_called: int = 0
    duration_ms: float = 0.0


__all__ = [
    "CopilotEvent",
    "CopilotResult",
]
