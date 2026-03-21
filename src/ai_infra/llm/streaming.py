"""Streaming utilities for Agent.astream().

This module provides typed streaming events and configuration for
normalized agent streaming. Apps receive clean, typed events instead
of raw LangChain message chunks.

Example:
    from ai_infra import Agent

    agent = Agent(tools=[search_docs])
    async for event in agent.astream("What is the refund policy?"):
        if event.type == "token":
            print(event.content, end="", flush=True)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal


@dataclass
class StreamEvent:
    """Typed streaming event from Agent.astream().

    Framework-agnostic - apps decide how to serialize (SSE, WebSocket, etc.)

    Event types:
        - thinking: Agent started processing (emitted once at start)
        - reasoning: Model's internal reasoning/narration before or between tool calls.
            Text the model emits *before* its first tool call or *between*
            consecutive tool calls is classified as reasoning.  This gives
            consumers a "thinking out loud" stream they can show in a
            collapsed panel.  Reasoning is NOT part of the final answer.
        - token: Text content chunk from the LLM response (the final answer)
        - tool_start: Tool execution started (name, arguments based on visibility)
        - tool_end: Tool execution completed (with timing, optional results)
        - done: Streaming completed (with summary stats)
        - error: Error occurred during streaming

    Attributes:
        type: Event type identifier
        content: Token content (for "token" events)
        tool: Tool name (for "tool_start", "tool_end" events)
        tool_id: Tool call ID for correlation
        arguments: Tool arguments dict (visibility=detailed+)
        result: Tool result (visibility=detailed+). Can be:
                - str: Formatted text for LLM consumption (default)
                - dict: Structured data when result_structured=True
                Use this when you need to parse tool outputs (e.g., extract
                package names, paths, create clickable links).
        result_structured: True if result is a structured dict from
                          create_retriever_tool(structured=True). When True,
                          to_dict() outputs 'structured_result' key instead of
                          'result' for frontend JSON parsing.
        preview: Truncated tool result (visibility=debug only).
                 For UI display/debugging, not parsing. Max 500 chars by default.
                 Use this for quick visual inspection during development.
        latency_ms: Tool execution time in milliseconds
        model: Model name (for "thinking" event)
        tools_called: Total tools called (for "done" event)
        error: Error message (for "error" event)
        timestamp: Event timestamp (Unix epoch)

    Visibility levels and tool results:
        - minimal: No tool events
        - standard: Tool names + timing only
        - detailed: + tool arguments + FULL tool results (result field)
        - debug: + truncated preview (preview field, max 500 chars)

    Example:
        # Token event
        StreamEvent(type="token", content="Hello")

        # Reasoning event (model narration before tool call)
        StreamEvent(type="reasoning", content="Let me search the docs for that.")

        # Tool start event
        StreamEvent(type="tool_start", tool="search_docs", tool_id="call_abc123")

        # Tool end event with text result (detailed visibility)
        StreamEvent(
            type="tool_end",
            tool="search_docs",
            tool_id="call_abc123",
            latency_ms=234.5,
            result="### Result 1 (svc-infra: auth.md)\\n...\\n### Result 2..."
        )

        # Tool end event with structured result (for frontend parsing)
        StreamEvent(
            type="tool_end",
            tool="search_docs",
            tool_id="call_abc123",
            latency_ms=234.5,
            result={"results": [...], "query": "auth", "count": 3},
            result_structured=True,
        )

        # Done event
        StreamEvent(type="done", tools_called=2)
    """

    type: Literal["thinking", "reasoning", "token", "tool_start", "tool_end", "done", "error"]

    # Type-specific data
    content: str | None = None  # token content
    tool: str | None = None  # tool name
    tool_id: str | None = None  # tool call ID
    arguments: dict[str, Any] | None = None  # tool arguments (detailed+)
    result: str | dict[str, Any] | None = None  # tool result (detailed+)
    result_structured: bool = False  # True if result is structured dict (from retriever tool)
    preview: str | None = None  # truncated result preview (debug only, for UI)
    latency_ms: float | None = None  # tool execution time
    model: str | None = None  # model name (thinking event)
    tools_called: int | None = None  # total tools (done event)
    error: str | None = None  # error message

    # Metadata
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization (excludes None values).

        For tool_end events with structured results, the result is included
        directly under 'structured_result' key for frontend parsing.

        Returns:
            Dict with type and all non-None fields.

        Example:
            event = StreamEvent(type="token", content="Hello")
            event.to_dict()  # {"type": "token", "content": "Hello"}

            # Structured tool result
            event = StreamEvent(
                type="tool_end",
                tool="search",
                result={"results": [...], "query": "...", "count": 3},
                result_structured=True,
            )
            event.to_dict()  # {"type": "tool_end", "tool": "search", "structured_result": {...}}
        """
        d: dict[str, Any] = {"type": self.type}

        # Handle structured tool results specially
        if self.type == "tool_end" and self.result_structured and self.result is not None:
            # Include structured result directly for frontend parsing
            d["structured_result"] = self.result
            d["result_structured"] = True
        elif self.result is not None:
            # Include text result normally
            d["result"] = self.result

        # Include other non-None fields
        for field_name in [
            "content",
            "tool",
            "tool_id",
            "arguments",
            "preview",
            "latency_ms",
            "model",
            "tools_called",
            "error",
        ]:
            val = getattr(self, field_name)
            if val is not None:
                d[field_name] = val
        return d

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        parts = [f"type={self.type!r}"]
        if self.content is not None:
            # Truncate long content
            content = self.content[:50] + "..." if len(self.content) > 50 else self.content
            parts.append(f"content={content!r}")
        if self.tool is not None:
            parts.append(f"tool={self.tool!r}")
        if self.error is not None:
            parts.append(f"error={self.error!r}")
        return f"StreamEvent({', '.join(parts)})"


@dataclass
class StreamConfig:
    """Configuration for Agent.astream() behavior.

    Most users won't need this - use direct parameters on astream() instead.
    This is for advanced customization scenarios.

    Attributes:
        visibility: Event detail level (minimal/standard/detailed/debug)
        include_thinking: Whether to emit "thinking" event at start
        include_reasoning: Whether to classify pre/inter-tool tokens as
            "reasoning" events instead of "token" events. When True,
            text the model emits before its first tool call or between
            consecutive tool calls is yielded as StreamEvent(type="reasoning").
            Defaults to True.
        include_tool_events: Whether to emit tool_start/tool_end events
        reasoning_token_limit: Maximum chars to buffer as reasoning before
            flushing as answer tokens (default 300).  If the model emits
            more than this many chars without calling a tool, the text is
            re-classified as a direct answer (token events).
        tool_result_preview_length: Max chars for tool result preview (debug)
        deduplicate_tool_starts: Prevent duplicate tool_start for same call

    Example:
        # Custom config for debugging
        config = StreamConfig(
            visibility="debug",
            tool_result_preview_length=1000,
        )
        async for event in agent.astream(prompt, stream_config=config):
            ...
    """

    # Visibility level (controls what events/data are emitted)
    visibility: Literal["minimal", "standard", "detailed", "debug"] = "standard"

    # Event filtering
    include_thinking: bool = True
    include_reasoning: bool = True
    include_tool_events: bool = True

    # Reasoning classifier
    reasoning_token_limit: int = 300

    # Tool handling
    tool_result_preview_length: int = 500
    deduplicate_tool_starts: bool = True


# Visibility level numeric values for comparison
_VISIBILITY_LEVELS: dict[str, int] = {
    "minimal": 0,
    "standard": 1,
    "detailed": 2,
    "debug": 3,
}

# Minimum visibility required for each event type
_EVENT_VISIBILITY_REQUIREMENTS: dict[str, int] = {
    "token": 0,  # Always emit tokens
    "thinking": 1,  # standard+
    "reasoning": 1,  # standard+
    "tool_start": 1,  # standard+
    "tool_end": 1,  # standard+
    "done": 1,  # standard+
    "error": 0,  # Always emit errors
}


def should_emit_event(event_type: str, visibility: str) -> bool:
    """Determine if event should be emitted based on visibility level.

    Args:
        event_type: The event type to check
        visibility: Current visibility setting

    Returns:
        True if event should be emitted
    """
    level = _VISIBILITY_LEVELS.get(visibility, 1)
    required = _EVENT_VISIBILITY_REQUIREMENTS.get(event_type, 0)
    return level >= required


def filter_event_for_visibility(event: StreamEvent, visibility: str) -> StreamEvent:
    """Filter event data based on visibility level.

    Removes sensitive or verbose data that shouldn't be exposed at the
    current visibility level.

    Visibility behavior for tool results:
        - minimal: No tool events (filtered by should_emit_event)
        - standard: Tool names + timing only (no arguments, result, or preview)
        - detailed: + arguments + FULL result (no preview)
        - debug: + arguments + result + truncated preview (everything)

    Args:
        event: The event to filter
        visibility: Current visibility setting

    Returns:
        Filtered event (may be the same instance if no changes needed)
    """
    if visibility == "minimal":
        # minimal: tokens and errors only, already filtered by should_emit_event
        return event

    if visibility == "standard":
        # standard: Remove tool arguments, result, and preview
        changes: dict[str, Any] = {}
        if event.arguments is not None:
            changes["arguments"] = None
        if event.result is not None:
            changes["result"] = None
        if event.preview is not None:
            changes["preview"] = None
        return replace(event, **changes) if changes else event

    if visibility == "detailed":
        # detailed: Include arguments and FULL result, exclude preview
        if event.preview is not None:
            return replace(event, preview=None)
        return event

    # debug: include everything (arguments + result + preview)
    return event


__all__ = [
    "StreamConfig",
    "StreamEvent",
    "filter_event_for_visibility",
    "should_emit_event",
]
