"""Tests for streaming callback invocation order.

Tests cover:
- Event emission order (thinking -> tools -> tokens -> done)
- Tool event ordering (start before end)
- Callback invocation timing
- Multiple tool ordering
- Nested tool call order
- Event timestamp ordering
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from ai_infra.llm.agent import Agent
from ai_infra.llm.streaming import (
    StreamConfig,
    StreamEvent,
)


class OrderTrackingAgent(Agent):
    """Agent that yields tokens in a specific order for testing."""

    def __init__(self, token_stream: list[tuple[Any, Any]]) -> None:
        """Initialize with a token stream."""
        super().__init__()
        self._token_stream = token_stream

    async def astream_agent_tokens(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[tuple[Any, Any]]:
        """Yield tokens in order."""
        for token in self._token_stream:
            yield token


@pytest.mark.asyncio
class TestEventOrderBasics:
    """Tests for basic event ordering."""

    async def test_thinking_event_emitted_first(self) -> None:
        """Test that thinking event is emitted before other events."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream("test", provider="openai"):
            events.append(event)

        assert len(events) >= 1
        assert events[0].type == "thinking"

    async def test_done_event_emitted_last(self) -> None:
        """Test that done event is emitted as the last event."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
            (AIMessageChunk(content=" World"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream("test", provider="openai"):
            events.append(event)

        # Done should be last
        assert events[-1].type == "done"

    async def test_tokens_before_done(self) -> None:
        """Test that token events come before done event."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream("test", provider="openai"):
            events.append(event)

        event_types = [e.type for e in events]
        token_index = next(
            (i for i, t in enumerate(event_types) if t == "token"),
            -1,
        )
        done_index = next(
            (i for i, t in enumerate(event_types) if t == "done"),
            -1,
        )

        if token_index != -1 and done_index != -1:
            assert token_index < done_index

    async def test_basic_event_sequence(self) -> None:
        """Test the basic event sequence: thinking -> tokens -> done."""
        tokens = [
            (AIMessageChunk(content="Response"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream("test", provider="openai"):
            events.append(event)

        event_types = [e.type for e in events]
        assert event_types == ["thinking", "turn_start", "token", "turn_end", "done"]


@pytest.mark.asyncio
class TestToolEventOrdering:
    """Tests for tool-related event ordering."""

    async def test_tool_start_before_tool_end(self) -> None:
        """Test that tool_start always comes before corresponding tool_end."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "search",
                    "args": '{"query": "test"}',
                }
            ],
        )
        tool_result = ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="search",
        )
        final = AIMessageChunk(content="done")

        tokens = [(tool_call, {}), (tool_result, {}), (final, {})]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        ):
            events.append(event)

        # Find tool_start and tool_end indices
        tool_start_idx = next(
            (i for i, e in enumerate(events) if e.type == "tool_start"),
            -1,
        )
        tool_end_idx = next(
            (i for i, e in enumerate(events) if e.type == "tool_end"),
            -1,
        )

        assert tool_start_idx != -1
        assert tool_end_idx != -1
        assert tool_start_idx < tool_end_idx

    async def test_multiple_tools_maintain_order(self) -> None:
        """Test that multiple tool calls maintain proper order."""
        tool1_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "tool_a",
                    "args": "{}",
                }
            ],
        )
        tool1_result = ToolMessage(
            content="result1",
            tool_call_id="call-1",
            name="tool_a",
        )
        tool2_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-2",
                    "name": "tool_b",
                    "args": "{}",
                }
            ],
        )
        tool2_result = ToolMessage(
            content="result2",
            tool_call_id="call-2",
            name="tool_b",
        )
        final = AIMessageChunk(content="final")

        tokens = [
            (tool1_call, {}),
            (tool1_result, {}),
            (tool2_call, {}),
            (tool2_result, {}),
            (final, {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        ):
            events.append(event)

        tool_events = [e for e in events if e.type in ("tool_start", "tool_end")]

        # Verify order: start1, end1, start2, end2
        assert len(tool_events) == 4
        assert tool_events[0].type == "tool_start"
        assert tool_events[0].tool == "tool_a"
        assert tool_events[1].type == "tool_end"
        assert tool_events[1].tool == "tool_a"
        assert tool_events[2].type == "tool_start"
        assert tool_events[2].tool == "tool_b"
        assert tool_events[3].type == "tool_end"
        assert tool_events[3].tool == "tool_b"

    async def test_tool_events_after_thinking(self) -> None:
        """Test that tool events come after thinking event."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "search",
                    "args": "{}",
                }
            ],
        )
        tool_result = ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="search",
        )

        tokens = [(tool_call, {}), (tool_result, {})]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        ):
            events.append(event)

        thinking_idx = next(
            (i for i, e in enumerate(events) if e.type == "thinking"),
            -1,
        )
        first_tool_idx = next(
            (i for i, e in enumerate(events) if e.type == "tool_start"),
            -1,
        )

        assert thinking_idx < first_tool_idx


@pytest.mark.asyncio
class TestTimestampOrdering:
    """Tests for event timestamp ordering."""

    async def test_timestamps_monotonically_increasing(self) -> None:
        """Test that event timestamps are monotonically increasing."""
        tokens = [
            (AIMessageChunk(content="A"), {}),
            (AIMessageChunk(content="B"), {}),
            (AIMessageChunk(content="C"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream("test", provider="openai"):
            events.append(event)

        for i in range(1, len(events)):
            # Timestamps should be non-decreasing (might be equal for fast events)
            assert events[i].timestamp >= events[i - 1].timestamp

    async def test_tool_end_timestamp_after_tool_start(self) -> None:
        """Test that tool_end timestamp is after or equal to tool_start."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "search",
                    "args": "{}",
                }
            ],
        )
        tool_result = ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="search",
        )

        tokens = [(tool_call, {}), (tool_result, {})]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        ):
            events.append(event)

        tool_start = next((e for e in events if e.type == "tool_start"), None)
        tool_end = next((e for e in events if e.type == "tool_end"), None)

        if tool_start and tool_end:
            assert tool_end.timestamp >= tool_start.timestamp


@pytest.mark.asyncio
class TestVisibilityFilterOrder:
    """Tests for event order with different visibility levels."""

    async def test_minimal_visibility_preserves_token_order(self) -> None:
        """Test that minimal visibility preserves token order."""
        tokens = [
            (AIMessageChunk(content="First"), {}),
            (AIMessageChunk(content="Second"), {}),
            (AIMessageChunk(content="Third"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="minimal"),
        ):
            events.append(event)

        token_contents = [e.content for e in events if e.type == "token"]
        assert token_contents == ["First", "Second", "Third"]

    async def test_standard_visibility_event_order(self) -> None:
        """Test event order at standard visibility."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "search",
                    "args": "{}",
                }
            ],
        )
        tool_result = ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="search",
        )
        final = AIMessageChunk(content="done")

        tokens = [(tool_call, {}), (tool_result, {}), (final, {})]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="standard"),
        ):
            events.append(event)

        event_types = [e.type for e in events]
        # Standard: thinking, tool_start, tool_end, token, done
        assert event_types[0] == "thinking"
        assert event_types[-1] == "done"


@pytest.mark.asyncio
class TestCallbackInvocation:
    """Tests for callback-style event handling patterns."""

    async def test_event_callback_receives_all_events(self) -> None:
        """Test that a callback receives all events in order."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        callback_events: list[StreamEvent] = []

        async def on_event(event: StreamEvent) -> None:
            """Callback for each event."""
            callback_events.append(event)

        async for event in agent.astream("test", provider="openai"):
            await on_event(event)

        assert len(callback_events) >= 3  # thinking, token, done

    async def test_separate_callbacks_per_event_type(self) -> None:
        """Test separate callbacks for different event types."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "search",
                    "args": "{}",
                }
            ],
        )
        tool_result = ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="search",
        )
        final = AIMessageChunk(content="done")

        tokens = [(tool_call, {}), (tool_result, {}), (final, {})]
        agent = OrderTrackingAgent(tokens)

        thinking_calls: list[StreamEvent] = []
        tool_start_calls: list[StreamEvent] = []
        tool_end_calls: list[StreamEvent] = []
        token_calls: list[StreamEvent] = []
        done_calls: list[StreamEvent] = []

        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        ):
            if event.type == "thinking":
                thinking_calls.append(event)
            elif event.type == "tool_start":
                tool_start_calls.append(event)
            elif event.type == "tool_end":
                tool_end_calls.append(event)
            elif event.type == "token":
                token_calls.append(event)
            elif event.type == "done":
                done_calls.append(event)

        assert len(thinking_calls) == 1
        assert len(tool_start_calls) == 1
        assert len(tool_end_calls) == 1
        assert len(token_calls) >= 1
        assert len(done_calls) == 1

    async def test_async_callback_with_delay_maintains_order(self) -> None:
        """Test that async callbacks with delays maintain event order."""
        import asyncio

        tokens = [
            (AIMessageChunk(content="A"), {}),
            (AIMessageChunk(content="B"), {}),
            (AIMessageChunk(content="C"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        processed_order: list[str] = []

        async def slow_callback(event: StreamEvent) -> None:
            """Slow callback that processes events."""
            await asyncio.sleep(0.01)  # Small delay
            if event.type == "token" and event.content:
                processed_order.append(event.content)

        async for event in agent.astream("test", provider="openai"):
            await slow_callback(event)

        # Order should be preserved
        assert processed_order == ["A", "B", "C"]


@pytest.mark.asyncio
class TestEventDeduplication:
    """Tests for event deduplication behavior."""

    async def test_duplicate_tool_starts_deduplicated(self) -> None:
        """Test that duplicate tool_start events are deduplicated."""
        # Simulate tool call chunks arriving in multiple pieces
        first_chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "search",
                    "args": '{"q',
                }
            ],
        )
        second_chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "search",
                    "args": 'uery": "test"}',
                }
            ],
        )
        tool_result = ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="search",
        )

        tokens = [(first_chunk, {}), (second_chunk, {}), (tool_result, {})]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="debug", deduplicate_tool_starts=True),
        ):
            events.append(event)

        tool_starts = [e for e in events if e.type == "tool_start"]
        # Should only have one tool_start despite multiple chunks
        assert len(tool_starts) == 1


@pytest.mark.asyncio
class TestDoneEventMetadata:
    """Tests for done event metadata."""

    async def test_done_event_includes_tools_called_count(self) -> None:
        """Test that done event includes correct tools_called count."""
        tool1_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "tool_a",
                    "args": "{}",
                }
            ],
        )
        tool1_result = ToolMessage(
            content="result1",
            tool_call_id="call-1",
            name="tool_a",
        )
        tool2_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-2",
                    "name": "tool_b",
                    "args": "{}",
                }
            ],
        )
        tool2_result = ToolMessage(
            content="result2",
            tool_call_id="call-2",
            name="tool_b",
        )
        final = AIMessageChunk(content="done")

        tokens = [
            (tool1_call, {}),
            (tool1_result, {}),
            (tool2_call, {}),
            (tool2_result, {}),
            (final, {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        ):
            events.append(event)

        done_event = next((e for e in events if e.type == "done"), None)
        assert done_event is not None
        assert done_event.tools_called == 2

    async def test_done_event_zero_tools_when_no_tools(self) -> None:
        """Test that done event has tools_called=0 when no tools used."""
        tokens = [
            (AIMessageChunk(content="Just text"), {}),
        ]
        agent = OrderTrackingAgent(tokens)

        events: list[StreamEvent] = []
        async for event in agent.astream("test", provider="openai"):
            events.append(event)

        done_event = next((e for e in events if e.type == "done"), None)
        assert done_event is not None
        assert done_event.tools_called == 0
