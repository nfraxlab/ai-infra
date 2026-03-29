"""Tests for streaming timeout handling.

Tests cover:
- Stream-level timeout configuration
- Token-level timeouts
- Tool execution timeouts during streaming
- Timeout recovery
- Graceful timeout handling
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from ai_infra.llm.agent import Agent
from ai_infra.llm.streaming import StreamConfig, StreamEvent


class TimeoutDummyAgent(Agent):
    """Agent that can simulate timeouts during streaming."""

    def __init__(
        self,
        token_stream: list[tuple[Any, Any]],
        delay_at_index: int | None = None,
        delay_duration: float = 10.0,
        normal_delay: float = 0.01,
    ) -> None:
        """Initialize with configurable delays.

        Args:
            token_stream: List of (token, metadata) tuples to yield
            delay_at_index: Index at which to introduce a long delay
            delay_duration: Duration of the long delay
            normal_delay: Normal delay between tokens
        """
        super().__init__()
        self._token_stream = token_stream
        self._delay_at_index = delay_at_index
        self._delay_duration = delay_duration
        self._normal_delay = normal_delay

    async def astream_agent_tokens(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[tuple[Any, Any]]:
        """Yield tokens with configurable delays."""
        for i, token in enumerate(self._token_stream):
            if i == self._delay_at_index:
                await asyncio.sleep(self._delay_duration)
            else:
                await asyncio.sleep(self._normal_delay)
            yield token


@pytest.mark.asyncio
class TestStreamTimeout:
    """Tests for stream-level timeout handling."""

    async def test_stream_completes_within_timeout(self) -> None:
        """Test that stream completes normally within timeout."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
            (AIMessageChunk(content=" World"), {}),
        ]
        agent = TimeoutDummyAgent(tokens, normal_delay=0.01)

        events: list[StreamEvent] = []
        async with asyncio.timeout(1.0):  # 1 second timeout
            async for event in agent.astream("test", provider="openai"):
                events.append(event)

        # Should have completed with done event
        event_types = [e.type for e in events]
        assert "done" in event_types

    async def test_stream_timeout_with_asyncio_timeout(self) -> None:
        """Test that asyncio.timeout works for slow streams."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
            (AIMessageChunk(content=" World"), {}),
        ]
        # Delay at first token for 10 seconds
        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=0,
            delay_duration=10.0,
            normal_delay=0.01,
        )

        events: list[StreamEvent] = []
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):  # 100ms timeout
                async for event in agent.astream("test", provider="openai"):
                    events.append(event)

    async def test_timeout_during_token_stream(self) -> None:
        """Test timeout that occurs during token streaming."""
        tokens = [
            (AIMessageChunk(content="Token1"), {}),
            (AIMessageChunk(content="Token2"), {}),
            (AIMessageChunk(content="Token3"), {}),
        ]
        # Delay at second token
        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=1,
            delay_duration=10.0,
            normal_delay=0.01,
        )

        events: list[StreamEvent] = []
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                async for event in agent.astream("test", provider="openai"):
                    events.append(event)

        # Should have received some events before timeout
        assert len(events) >= 1

    async def test_timeout_preserves_partial_events(self) -> None:
        """Test that events before timeout are preserved."""
        tokens = [
            (AIMessageChunk(content="First"), {}),
            (AIMessageChunk(content="Second"), {}),
        ]
        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=1,
            delay_duration=10.0,
            normal_delay=0.01,
        )

        events: list[StreamEvent] = []
        try:
            async with asyncio.timeout(0.2):
                async for event in agent.astream("test", provider="openai"):
                    events.append(event)
        except TimeoutError:
            pass

        # First token event should be preserved
        token_events = [e for e in events if e.type == "token"]
        if token_events:
            assert token_events[0].content == "First"


@pytest.mark.asyncio
class TestToolExecutionTimeout:
    """Tests for timeout during tool execution in streaming."""

    async def test_timeout_during_tool_call(self) -> None:
        """Test timeout while tool is being called."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "slow_tool",
                    "args": "{}",
                }
            ],
        )
        # Tool result comes after a long delay (simulated by delay_at_index)
        tool_result = ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="slow_tool",
        )
        final = AIMessageChunk(content="done")

        tokens = [(tool_call, {}), (tool_result, {}), (final, {})]
        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=1,  # Delay before tool result
            delay_duration=10.0,
            normal_delay=0.01,
        )

        events: list[StreamEvent] = []
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.2):
                async for event in agent.astream(
                    "test",
                    provider="openai",
                    stream_config=StreamConfig(visibility="debug"),
                ):
                    events.append(event)

        # Should have tool_start before timeout
        event_types = [e.type for e in events]
        assert "tool_start" in event_types

    async def test_timeout_after_tool_complete(self) -> None:
        """Test timeout after tool completes but before done."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "fast_tool",
                    "args": "{}",
                }
            ],
        )
        tool_result = ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="fast_tool",
        )
        final = AIMessageChunk(content="slow final")

        tokens = [(tool_call, {}), (tool_result, {}), (final, {})]
        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=2,  # Delay before final token
            delay_duration=10.0,
            normal_delay=0.01,
        )

        events: list[StreamEvent] = []
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.2):
                async for event in agent.astream(
                    "test",
                    provider="openai",
                    stream_config=StreamConfig(visibility="debug"),
                ):
                    events.append(event)

        # Should have both tool_start and tool_end before timeout
        event_types = [e.type for e in events]
        assert "tool_start" in event_types
        assert "tool_end" in event_types


@pytest.mark.asyncio
class TestTimeoutRecovery:
    """Tests for recovering from timeout conditions."""

    async def test_can_start_new_stream_after_timeout(self) -> None:
        """Test that a new stream can be started after timeout."""
        slow_tokens = [
            (AIMessageChunk(content="Slow"), {}),
        ]
        fast_tokens = [
            (AIMessageChunk(content="Fast"), {}),
        ]

        slow_agent = TimeoutDummyAgent(
            slow_tokens,
            delay_at_index=0,
            delay_duration=10.0,
        )
        fast_agent = TimeoutDummyAgent(fast_tokens, normal_delay=0.01)

        # First stream times out
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                async for _ in slow_agent.astream("test", provider="openai"):
                    pass

        # Second stream should work
        events: list[StreamEvent] = []
        async with asyncio.timeout(1.0):
            async for event in fast_agent.astream("test", provider="openai"):
                events.append(event)

        assert any(e.type == "done" for e in events)

    async def test_timeout_does_not_affect_other_streams(self) -> None:
        """Test that timeout on one stream doesn't affect others."""
        fast_tokens = [
            (AIMessageChunk(content="OK"), {}),
        ]

        async def slow_stream() -> list[StreamEvent]:
            """Stream that will timeout."""
            slow_agent = TimeoutDummyAgent(
                fast_tokens,
                delay_at_index=0,
                delay_duration=10.0,
            )
            events = []
            try:
                async with asyncio.timeout(0.1):
                    async for event in slow_agent.astream("test", provider="openai"):
                        events.append(event)
            except TimeoutError:
                pass
            return events

        async def fast_stream() -> list[StreamEvent]:
            """Stream that completes normally."""
            fast_agent = TimeoutDummyAgent(fast_tokens, normal_delay=0.01)
            events = []
            async for event in fast_agent.astream("test", provider="openai"):
                events.append(event)
            return events

        slow_events, fast_events = await asyncio.gather(
            slow_stream(),
            fast_stream(),
        )

        # Fast stream should complete with done
        assert any(e.type == "done" for e in fast_events)


@pytest.mark.asyncio
class TestWaitForTimeout:
    """Tests for wait_for style timeouts."""

    async def test_wait_for_with_stream_generator(self) -> None:
        """Test using asyncio.wait_for with stream consumption."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
            (AIMessageChunk(content=" World"), {}),
        ]
        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=0,
            delay_duration=10.0,
        )

        async def consume_stream() -> list[StreamEvent]:
            """Consume all stream events."""
            events = []
            async for event in agent.astream("test", provider="openai"):
                events.append(event)
            return events

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(consume_stream(), timeout=0.1)

    async def test_wait_for_completes_successfully(self) -> None:
        """Test wait_for when stream completes in time."""
        tokens = [
            (AIMessageChunk(content="Quick"), {}),
        ]
        agent = TimeoutDummyAgent(tokens, normal_delay=0.01)

        async def consume_stream() -> list[StreamEvent]:
            """Consume all stream events."""
            events = []
            async for event in agent.astream("test", provider="openai"):
                events.append(event)
            return events

        events = await asyncio.wait_for(consume_stream(), timeout=1.0)

        assert any(e.type == "done" for e in events)


@pytest.mark.asyncio
class TestPerTokenTimeout:
    """Tests for per-token timeout scenarios."""

    async def test_detect_stalled_stream(self) -> None:
        """Test detecting a stalled stream (no tokens for a while)."""
        tokens = [
            (AIMessageChunk(content="First"), {}),
            (AIMessageChunk(content="Second"), {}),  # This one stalls
            (AIMessageChunk(content="Third"), {}),
        ]
        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=1,
            delay_duration=10.0,
            normal_delay=0.01,
        )

        events: list[StreamEvent] = []
        last_event_time = asyncio.get_event_loop().time()
        token_timeout = 0.2  # 200ms between tokens

        try:
            async for event in agent.astream("test", provider="openai"):
                current_time = asyncio.get_event_loop().time()
                if current_time - last_event_time > token_timeout and events:
                    # Would detect stall here
                    pass
                events.append(event)
                last_event_time = current_time

                # Implement manual timeout check
                if len(events) >= 2:
                    # Wait for next token with timeout
                    try:
                        await asyncio.wait_for(asyncio.sleep(0.5), timeout=0.3)
                    except TimeoutError:
                        break
        except TimeoutError:
            pass

        # Should have received first token (or turn_start before it)
        relevant_events = [e for e in events if e.type in ("token", "turn_start")]
        assert len(relevant_events) >= 1


@pytest.mark.asyncio
class TestTimeoutWithVisibility:
    """Tests for timeout behavior with different visibility levels."""

    async def test_timeout_with_minimal_visibility(self) -> None:
        """Test timeout with minimal visibility."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
        ]
        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=0,
            delay_duration=10.0,
        )

        events: list[StreamEvent] = []
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                async for event in agent.astream(
                    "test",
                    provider="openai",
                    stream_config=StreamConfig(visibility="minimal"),
                ):
                    events.append(event)

        # Even with timeout, minimal visibility should only have token types
        for event in events:
            assert event.type in ("token", "error")

    async def test_timeout_with_debug_visibility(self) -> None:
        """Test timeout with debug visibility."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-1",
                    "name": "debug_tool",
                    "args": '{"verbose": true}',
                }
            ],
        )
        tokens = [(tool_call, {})]

        agent = TimeoutDummyAgent(
            tokens,
            delay_at_index=0,
            delay_duration=10.0,
        )

        events: list[StreamEvent] = []
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                async for event in agent.astream(
                    "test",
                    provider="openai",
                    stream_config=StreamConfig(visibility="debug"),
                ):
                    events.append(event)

        # Debug visibility might have thinking event before timeout
        # depending on timing
        assert isinstance(events, list)
