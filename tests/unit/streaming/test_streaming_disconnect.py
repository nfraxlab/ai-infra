"""Tests for streaming client disconnect scenarios.

Tests cover:
- Client disconnect mid-stream
- Generator cleanup on disconnect
- Partial event handling
- Error propagation on disconnect
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from ai_infra.llm.agent import Agent
from ai_infra.llm.streaming import StreamConfig, StreamEvent


class SlowDummyAgent(Agent):
    """Agent that yields tokens slowly to simulate streaming."""

    def __init__(
        self,
        token_stream: list[tuple[Any, Any]],
        delay: float = 0.1,
    ) -> None:
        """Initialize with a token stream and delay between yields."""
        super().__init__()
        self._token_stream = token_stream
        self._delay = delay
        self._cancelled = False
        self._tokens_yielded = 0

    async def astream_agent_tokens(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[tuple[Any, Any]]:
        """Yield tokens with delays to simulate real streaming."""
        for token in self._token_stream:
            if self._cancelled:
                break
            await asyncio.sleep(self._delay)
            self._tokens_yielded += 1
            yield token


class DisconnectingConsumer:
    """Consumer that disconnects after a specified number of events."""

    def __init__(self, disconnect_after: int) -> None:
        """Initialize with number of events before disconnect."""
        self.disconnect_after = disconnect_after
        self.events_consumed = 0
        self.events: list[StreamEvent] = []

    async def consume(self, stream: AsyncIterator[StreamEvent]) -> None:
        """Consume events until disconnect."""
        async for event in stream:
            self.events_consumed += 1
            self.events.append(event)
            if self.events_consumed >= self.disconnect_after:
                # Simulate client disconnect by breaking
                break


@pytest.mark.asyncio
class TestStreamDisconnect:
    """Tests for client disconnect during streaming."""

    async def test_consumer_can_break_stream_early(self) -> None:
        """Test that consumer can break out of stream without error."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
            (AIMessageChunk(content=" World"), {}),
            (AIMessageChunk(content="!"), {}),
        ]
        agent = SlowDummyAgent(tokens, delay=0.01)

        consumer = DisconnectingConsumer(disconnect_after=1)
        await consumer.consume(agent.astream("test", provider="openai"))

        # Should have only consumed 1 event (the first one received)
        assert consumer.events_consumed == 1

    async def test_partial_stream_events_are_valid(self) -> None:
        """Test that partially consumed events are still valid."""
        tokens = [
            (AIMessageChunk(content="Part1"), {}),
            (AIMessageChunk(content="Part2"), {}),
            (AIMessageChunk(content="Part3"), {}),
        ]
        agent = SlowDummyAgent(tokens, delay=0.01)

        events: list[StreamEvent] = []
        async for event in agent.astream("test", provider="openai"):
            events.append(event)
            if len(events) >= 2:
                break

        # Verify collected events are valid
        for event in events:
            assert isinstance(event, StreamEvent)
            assert event.type in ("thinking", "token", "tool_start", "tool_end", "done", "error")

    async def test_stream_generator_cleanup_on_break(self) -> None:
        """Test that generator resources are cleaned up on break."""
        tokens = [
            (AIMessageChunk(content="A"), {}),
            (AIMessageChunk(content="B"), {}),
            (AIMessageChunk(content="C"), {}),
        ]
        agent = SlowDummyAgent(tokens, delay=0.01)

        stream = agent.astream("test", provider="openai")
        count = 0

        async for _ in stream:
            count += 1
            if count >= 1:
                break

        # Generator should be closeable after break
        await stream.aclose()  # Should not raise

    async def test_disconnect_during_tool_call(self) -> None:
        """Test disconnect during tool execution."""
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
        agent = SlowDummyAgent(tokens, delay=0.01)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="debug"),
        ):
            events.append(event)
            # Disconnect after tool_start
            if event.type == "tool_start":
                break

        # Should have thinking + tool_start
        event_types = [e.type for e in events]
        assert "thinking" in event_types
        assert "tool_start" in event_types

    async def test_disconnect_preserves_collected_events(self) -> None:
        """Test that events collected before disconnect are preserved."""
        tokens = [
            (AIMessageChunk(content="Hello"), {}),
            (AIMessageChunk(content=" World"), {}),
        ]
        agent = SlowDummyAgent(tokens, delay=0.01)

        collected: list[str] = []
        async for event in agent.astream("test", provider="openai"):
            if event.type == "token" and event.content:
                collected.append(event.content)
            if len(collected) >= 1:
                break

        # First token should be preserved
        assert len(collected) >= 1
        assert collected[0] == "Hello"


@pytest.mark.asyncio
class TestStreamCancellation:
    """Tests for stream cancellation scenarios."""

    async def test_asyncio_cancel_during_stream(self) -> None:
        """Test that asyncio.CancelledError is handled gracefully."""
        tokens = [
            (AIMessageChunk(content="A"), {}),
            (AIMessageChunk(content="B"), {}),
            (AIMessageChunk(content="C"), {}),
        ]
        agent = SlowDummyAgent(tokens, delay=0.1)

        async def consume_and_cancel() -> list[StreamEvent]:
            """Consume stream and cancel after first event."""
            events: list[StreamEvent] = []
            task = asyncio.current_task()
            async for event in agent.astream("test", provider="openai"):
                events.append(event)
                if len(events) >= 2:
                    if task:
                        task.cancel()
                    await asyncio.sleep(0)  # yield so CancelledError is raised
            return events

        with pytest.raises(asyncio.CancelledError):
            await consume_and_cancel()

    async def test_multiple_concurrent_consumers_disconnect(self) -> None:
        """Test multiple consumers disconnecting at different times."""
        tokens = [(AIMessageChunk(content=f"Token{i}"), {}) for i in range(10)]

        async def consume_n(agent: Agent, n: int) -> int:
            """Consume n events and return count."""
            count = 0
            async for _ in agent.astream("test", provider="openai"):
                count += 1
                if count >= n:
                    break
            return count

        agent1 = SlowDummyAgent(tokens.copy(), delay=0.01)
        agent2 = SlowDummyAgent(tokens.copy(), delay=0.01)

        results = await asyncio.gather(
            consume_n(agent1, 2),
            consume_n(agent2, 5),
        )

        assert results[0] == 2
        assert results[1] == 5


@pytest.mark.asyncio
class TestStreamErrorOnDisconnect:
    """Tests for error handling during disconnect."""

    async def test_stream_error_after_partial_consumption(self) -> None:
        """Test that errors after partial consumption are handled."""

        class ErrorAgent(Agent):
            """Agent that raises error after some tokens."""

            def __init__(self, error_after: int) -> None:
                """Initialize with token count before error."""
                super().__init__()
                self.error_after = error_after

            async def astream_agent_tokens(
                self, *args: Any, **kwargs: Any
            ) -> AsyncIterator[tuple[Any, Any]]:
                """Yield tokens then raise."""
                for i in range(self.error_after):
                    yield (AIMessageChunk(content=f"Token{i}"), {})
                raise RuntimeError("Stream error")

        agent = ErrorAgent(error_after=3)
        events: list[StreamEvent] = []

        with pytest.raises(RuntimeError, match="Stream error"):
            async for event in agent.astream("test", provider="openai"):
                events.append(event)

        # Should have consumed some events before error
        assert len(events) >= 1

    async def test_disconnect_does_not_mask_underlying_errors(self) -> None:
        """Test that underlying errors are not masked by disconnect."""

        class FailingAgent(Agent):
            """Agent that fails after yielding some tokens."""

            async def astream_agent_tokens(
                self, *args: Any, **kwargs: Any
            ) -> AsyncIterator[tuple[Any, Any]]:
                """Yield one token then fail."""
                yield (AIMessageChunk(content="OK"), {})
                raise ValueError("Backend failure")

        agent = FailingAgent()

        with pytest.raises(ValueError, match="Backend failure"):
            async for _ in agent.astream("test", provider="openai"):
                pass


@pytest.mark.asyncio
class TestStreamVisibilityOnDisconnect:
    """Tests for visibility behavior on disconnect."""

    async def test_minimal_visibility_no_tool_events_before_disconnect(self) -> None:
        """Test that minimal visibility doesn't emit tool events even on disconnect."""
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
        final = AIMessageChunk(content="result")

        tokens = [(tool_call, {}), (final, {})]
        agent = SlowDummyAgent(tokens, delay=0.01)

        events: list[StreamEvent] = []
        async for event in agent.astream(
            "test",
            provider="openai",
            stream_config=StreamConfig(visibility="minimal"),
        ):
            events.append(event)
            if len(events) >= 3:
                break

        # Minimal visibility should only have tokens
        event_types = [e.type for e in events]
        assert "tool_start" not in event_types
        assert "tool_end" not in event_types

    async def test_disconnect_after_done_event(self) -> None:
        """Test that disconnect after done event is clean."""
        tokens = [
            (AIMessageChunk(content="Complete"), {}),
        ]
        agent = SlowDummyAgent(tokens, delay=0.01)

        events: list[StreamEvent] = []
        async for event in agent.astream("test", provider="openai"):
            events.append(event)
            if event.type == "done":
                break

        # Should have done event
        event_types = [e.type for e in events]
        assert "done" in event_types
