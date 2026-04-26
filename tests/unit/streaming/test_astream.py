"""Tests for Agent.astream normalized streaming events."""

from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from ai_infra.llm.agent import Agent
from ai_infra.llm.streaming import StreamConfig


class DummyAgent(Agent):
    """Agent with a stubbed token stream for astream()."""

    def __init__(self, token_stream: Iterable[tuple[Any, Any]], **model_kwargs: Any):
        super().__init__(**model_kwargs)
        self._token_stream = list(token_stream)

    async def astream_agent_tokens(self, *args, **kwargs):
        for token in self._token_stream:
            yield token


class RecordingDummyAgent(DummyAgent):
    """Dummy agent that records forwarded kwargs for assertions."""

    def __init__(self, token_stream: Iterable[tuple[Any, Any]], **model_kwargs: Any):
        super().__init__(token_stream, **model_kwargs)
        self.last_kwargs: dict[str, Any] | None = None

    async def astream_agent_tokens(self, *args, **kwargs):
        self.last_kwargs = kwargs
        async for token in super().astream_agent_tokens(*args, **kwargs):
            yield token


@pytest.mark.asyncio
async def test_astream_emits_tool_events_and_done():
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call-1",
                "name": "search_docs",
                "args": '{"query": "pricing"}',
            }
        ],
    )
    tool_result = ToolMessage(content="result", tool_call_id="call-1", name="search_docs")
    final_chunk = AIMessageChunk(content="all done")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {}), (final_chunk, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        )
    ]

    assert [event.type for event in events] == [
        "thinking",
        "turn_start",
        "tool_start",
        "tool_end",
        "turn_end",
        "turn_start",
        "token",
        "turn_end",
        "done",
    ]
    tool_start = events[2]
    assert tool_start.tool == "search_docs"
    assert tool_start.arguments == {"query": "pricing"}
    tool_end = events[3]
    assert tool_end.tool_id == "call-1"
    assert tool_end.preview == "result"
    assert events[-1].tools_called == 1


@pytest.mark.asyncio
async def test_astream_standard_hides_tool_arguments():
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call-2",
                "name": "search_docs",
                "args": '{"query": "pricing"}',
            }
        ],
    )
    tool_result = ToolMessage(content="result", tool_call_id="call-2", name="search_docs")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="standard"),
        )
    ]

    tool_start_events = [event for event in events if event.type == "tool_start"]
    assert len(tool_start_events) == 1
    assert tool_start_events[0].arguments is None


@pytest.mark.asyncio
async def test_astream_minimal_visibility_only_tokens():
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"index": 0, "id": "call-3", "name": "search_docs", "args": "{}"}],
    )
    tool_result = ToolMessage(content="done", tool_call_id="call-3", name="search_docs")
    token_chunk = AIMessageChunk(content="final")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {}), (token_chunk, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="minimal"),
        )
    ]

    assert [event.type for event in events] == ["token"]
    assert events[0].content == "final"


@pytest.mark.asyncio
async def test_astream_deduplicates_tool_starts():
    first_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"index": 0, "id": "call-4", "name": "search_docs", "args": "{}"}],
    )
    duplicate_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"index": 0, "id": "call-4", "name": "search_docs", "args": "{}"}],
    )

    agent = DummyAgent([(first_chunk, {}), (duplicate_chunk, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="standard"),
        )
    ]

    tool_start_events = [event for event in events if event.type == "tool_start"]
    assert len(tool_start_events) == 1


@pytest.mark.asyncio
async def test_astream_result_field_at_detailed_visibility():
    """Test that result field contains full tool output at detailed visibility."""
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call-5",
                "name": "search_docs",
                "args": '{"query": "auth"}',
            }
        ],
    )
    full_result = "### Result 1 (svc-infra: auth.md)\nAuthentication docs...\n---\n### Result 2 (ai-infra: core/llm.md)\nLLM usage..."
    tool_result = ToolMessage(content=full_result, tool_call_id="call-5", name="search_docs")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="detailed"),
        )
    ]

    tool_end_events = [event for event in events if event.type == "tool_end"]
    assert len(tool_end_events) == 1
    tool_end = tool_end_events[0]

    # Result field should contain full output at detailed visibility
    assert tool_end.result == full_result
    assert tool_end.result is not None
    assert len(tool_end.result) > 100  # Full result, not truncated

    # Preview should NOT be included at detailed visibility
    assert tool_end.preview is None


@pytest.mark.asyncio
async def test_astream_result_field_at_debug_visibility():
    """Test that both result and preview fields are populated at debug visibility."""
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call-6",
                "name": "search_code",
                "args": '{"query": "Agent"}',
            }
        ],
    )
    full_result = "def Agent():\n    pass\n" * 50  # Long result
    tool_result = ToolMessage(content=full_result, tool_call_id="call-6", name="search_code")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        )
    ]

    tool_end_events = [event for event in events if event.type == "tool_end"]
    assert len(tool_end_events) == 1
    tool_end = tool_end_events[0]

    # Both result and preview should be populated at debug visibility
    assert tool_end.result == full_result
    assert tool_end.preview is not None
    assert len(tool_end.preview) <= 503  # Preview is truncated (includes "...")
    assert len(tool_end.result) > len(tool_end.preview)  # Result is full


@pytest.mark.asyncio
async def test_astream_no_result_at_standard_visibility():
    """Test that result field is None at standard visibility."""
    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call-7",
                "name": "search_docs",
                "args": '{"query": "test"}',
            }
        ],
    )
    tool_result = ToolMessage(content="some result", tool_call_id="call-7", name="search_docs")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="standard"),
        )
    ]

    tool_end_events = [event for event in events if event.type == "tool_end"]
    assert len(tool_end_events) == 1
    tool_end = tool_end_events[0]

    # Neither result nor preview should be included at standard visibility
    assert tool_end.result is None
    assert tool_end.preview is None


@pytest.mark.asyncio
async def test_astream_parse_mcp_tool_results():
    """Test parsing structured data from MCP tool results for UI features."""
    import re

    tool_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call-8",
                "name": "search_docs",
                "args": '{"query": "billing"}',
            }
        ],
    )
    # Realistic MCP search_docs output format
    mcp_result = """### Result 1 (svc-infra: billing.md)
Stripe integration for payments...
---
### Result 2 (fin-infra: payments.md)
Payment processing architecture...
---
### Result 3 (ai-infra: tools/billing.md)
Billing tool for agents..."""
    tool_result = ToolMessage(content=mcp_result, tool_call_id="call-8", name="search_docs")

    agent = DummyAgent([(tool_call_chunk, {}), (tool_result, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            stream_config=StreamConfig(visibility="detailed"),
        )
    ]

    tool_end_events = [event for event in events if event.type == "tool_end"]
    assert len(tool_end_events) == 1
    tool_end = tool_end_events[0]

    # Parse the result field to extract structured data
    assert tool_end.result is not None
    pattern = r"### Result \d+ \((.+?): (.+?)\)"
    matches = list(re.finditer(pattern, tool_end.result))

    assert len(matches) == 3

    # Verify we can extract package and path for clickable links
    docs = [{"package": m.group(1), "path": m.group(2)} for m in matches]
    assert docs == [
        {"package": "svc-infra", "path": "billing.md"},
        {"package": "fin-infra", "path": "payments.md"},
        {"package": "ai-infra", "path": "tools/billing.md"},
    ]


@pytest.mark.asyncio
async def test_astream_forwards_explicit_tool_controls_outside_model_kwargs():
    final_chunk = AIMessageChunk(content="final")
    agent = RecordingDummyAgent([(final_chunk, {})])

    events = [
        event
        async for event in agent.astream(
            "hello",
            provider="openai",
            tool_controls={"tool_choice": {"name": "calculator"}, "force_once": True},
            temperature=0.1,
        )
    ]

    assert [event.type for event in events] == [
        "thinking",
        "turn_start",
        "token",
        "turn_end",
        "done",
    ]
    assert agent.last_kwargs is not None
    assert agent.last_kwargs["tool_controls"] == {
        "tool_choice": {"name": "calculator"},
        "force_once": True,
    }
    assert agent.last_kwargs["model_kwargs"] == {"temperature": 0.1}


@pytest.mark.asyncio
async def test_astream_hoists_default_tool_controls_out_of_model_kwargs():
    final_chunk = AIMessageChunk(content="final")
    agent = RecordingDummyAgent(
        [(final_chunk, {})],
        tool_controls={"tool_choice": {"name": "search_docs"}},
        temperature=0.1,
    )

    events = [event async for event in agent.astream("hello", provider="openai")]

    assert events[-1].type == "done"
    assert agent.last_kwargs is not None
    assert agent.last_kwargs["tool_controls"] == {"tool_choice": {"name": "search_docs"}}
    assert agent.last_kwargs["model_kwargs"] == {"temperature": 0.1}
