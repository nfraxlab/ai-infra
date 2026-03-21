"""Tests for Agent streaming responses.

Tests cover:
- StreamConfig configuration
- Stream visibility levels (minimal, standard, debug)
- Token events during streaming
- Tool events during streaming
- Stream completion events
- Error handling during streaming

Phase 1.1.3 of production readiness test plan.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from ai_infra import Agent
from ai_infra.llm.streaming import StreamConfig

# =============================================================================
# Test Helpers
# =============================================================================


class MockStreamingAgent(Agent):
    """Agent with mocked token stream for testing astream()."""

    def __init__(self, token_stream: list[tuple[Any, Any]], **kwargs):
        super().__init__(**kwargs)
        self._token_stream = token_stream

    async def astream_agent_tokens(self, *args, **kwargs):
        for token in self._token_stream:
            yield token


# =============================================================================
# StreamConfig Tests
# =============================================================================


class TestStreamConfig:
    """Tests for StreamConfig configuration."""

    def test_default_visibility_standard(self):
        """Default visibility is 'standard'."""
        config = StreamConfig()
        assert config.visibility == "standard"

    def test_minimal_visibility(self):
        """Minimal visibility hides tool info."""
        config = StreamConfig(visibility="minimal")
        assert config.visibility == "minimal"

    def test_debug_visibility(self):
        """Debug visibility shows all details."""
        config = StreamConfig(visibility="debug")
        assert config.visibility == "debug"

    def test_include_thinking(self):
        """include_thinking option can be set."""
        config = StreamConfig(include_thinking=True)
        assert config.include_thinking is True

    def test_deduplicate_tool_starts(self):
        """deduplicate_tool_starts option can be set."""
        config = StreamConfig(deduplicate_tool_starts=False)
        assert config.deduplicate_tool_starts is False

    def test_include_reasoning_default(self):
        """include_reasoning defaults to True."""
        config = StreamConfig()
        assert config.include_reasoning is True

    def test_reasoning_token_limit(self):
        """reasoning_token_limit can be customized."""
        config = StreamConfig(reasoning_token_limit=500)
        assert config.reasoning_token_limit == 500


# =============================================================================
# Token Streaming Tests
# =============================================================================


class TestTokenStreaming:
    """Tests for token streaming events."""

    @pytest.mark.asyncio
    async def test_token_events_emitted(self):
        """Token events are emitted during streaming."""
        chunks = [
            AIMessageChunk(content="Hello"),
            AIMessageChunk(content=" world"),
            AIMessageChunk(content="!"),
        ]
        token_stream = [(chunk, {}) for chunk in chunks]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard"),
            )
        ]

        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) == 3
        assert token_events[0].content == "Hello"
        assert token_events[1].content == " world"
        assert token_events[2].content == "!"

    @pytest.mark.asyncio
    async def test_done_event_emitted(self):
        """Done event is emitted at end of stream."""
        token_stream = [(AIMessageChunk(content="Done"), {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard"),
            )
        ]

        done_events = [e for e in events if e.type == "done"]
        assert len(done_events) == 1


# =============================================================================
# Tool Streaming Tests
# =============================================================================


class TestToolStreaming:
    """Tests for tool events during streaming."""

    @pytest.mark.asyncio
    async def test_tool_start_event(self):
        """Tool start events are emitted."""
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
        tool_result = ToolMessage(content="result", tool_call_id="call-1", name="search")

        token_stream = [(tool_call, {}), (tool_result, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="debug"),
            )
        ]

        tool_starts = [e for e in events if e.type == "tool_start"]
        assert len(tool_starts) == 1
        assert tool_starts[0].tool == "search"

    @pytest.mark.asyncio
    async def test_tool_end_event(self):
        """Tool end events are emitted."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-2",
                    "name": "calculate",
                    "args": '{"x": 5}',
                }
            ],
        )
        tool_result = ToolMessage(content="10", tool_call_id="call-2", name="calculate")

        token_stream = [(tool_call, {}), (tool_result, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="debug"),
            )
        ]

        tool_ends = [e for e in events if e.type == "tool_end"]
        assert len(tool_ends) == 1
        assert tool_ends[0].tool_id == "call-2"
        assert tool_ends[0].preview == "10"

    @pytest.mark.asyncio
    async def test_debug_visibility_includes_arguments(self):
        """Debug visibility includes tool arguments."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-3",
                    "name": "search",
                    "args": '{"query": "secret"}',
                }
            ],
        )
        tool_result = ToolMessage(content="found", tool_call_id="call-3", name="search")

        token_stream = [(tool_call, {}), (tool_result, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="debug"),
            )
        ]

        tool_starts = [e for e in events if e.type == "tool_start"]
        assert tool_starts[0].arguments == {"query": "secret"}

    @pytest.mark.asyncio
    async def test_standard_visibility_hides_arguments(self):
        """Standard visibility hides tool arguments."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-4",
                    "name": "search",
                    "args": '{"query": "secret"}',
                }
            ],
        )
        tool_result = ToolMessage(content="found", tool_call_id="call-4", name="search")

        token_stream = [(tool_call, {}), (tool_result, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard"),
            )
        ]

        tool_starts = [e for e in events if e.type == "tool_start"]
        assert len(tool_starts) == 1
        assert tool_starts[0].arguments is None


# =============================================================================
# Visibility Level Tests
# =============================================================================


class TestVisibilityLevels:
    """Tests for different visibility levels."""

    @pytest.mark.asyncio
    async def test_minimal_only_tokens(self):
        """Minimal visibility only shows tokens."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-5",
                    "name": "search",
                    "args": "{}",
                }
            ],
        )
        tool_result = ToolMessage(content="result", tool_call_id="call-5", name="search")
        final = AIMessageChunk(content="done")

        token_stream = [(tool_call, {}), (tool_result, {}), (final, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="minimal"),
            )
        ]

        event_types = [e.type for e in events]
        # Minimal should not include tool_start/tool_end
        assert "tool_start" not in event_types
        assert "tool_end" not in event_types

    @pytest.mark.asyncio
    async def test_done_includes_tool_count(self):
        """Done event includes count of tools called."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-6",
                    "name": "tool1",
                    "args": "{}",
                }
            ],
        )
        tool_result = ToolMessage(content="r1", tool_call_id="call-6", name="tool1")

        token_stream = [(tool_call, {}), (tool_result, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="debug"),
            )
        ]

        done_events = [e for e in events if e.type == "done"]
        assert len(done_events) == 1
        assert done_events[0].tools_called == 1


# =============================================================================
# Thinking Event Tests
# =============================================================================


class TestThinkingEvents:
    """Tests for thinking/reasoning events."""

    @pytest.mark.asyncio
    async def test_thinking_event_emitted(self):
        """Thinking events are emitted when tool is called."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "index": 0,
                    "id": "call-7",
                    "name": "analyze",
                    "args": "{}",
                }
            ],
        )
        tool_result = ToolMessage(content="analyzed", tool_call_id="call-7", name="analyze")

        token_stream = [(tool_call, {}), (tool_result, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="debug"),
            )
        ]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) >= 1


# =============================================================================
# Stream Error Handling Tests
# =============================================================================


class TestStreamErrorHandling:
    """Tests for error handling during streaming."""

    @pytest.mark.asyncio
    async def test_empty_stream_produces_done(self):
        """Empty stream still produces done event."""
        agent = MockStreamingAgent([])
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard"),
            )
        ]

        done_events = [e for e in events if e.type == "done"]
        assert len(done_events) == 1


# =============================================================================
# Reasoning Event Tests
# =============================================================================


class TestReasoningEvents:
    """Tests for reasoning/narration classification in astream()."""

    @pytest.mark.asyncio
    async def test_pre_tool_text_emitted_as_reasoning(self):
        """Text before a tool call is classified as reasoning."""
        token_stream = [
            (AIMessageChunk(content="Let me search for that."), {}),
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-r1", "name": "search", "args": '{"q":"test"}'}
                    ],
                ),
                {},
            ),
            (ToolMessage(content="found it", tool_call_id="call-r1", name="search"), {}),
            (AIMessageChunk(content="Here is the answer."), {}),
        ]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard", include_reasoning=True),
            )
        ]

        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(reasoning_events) == 1
        assert reasoning_events[0].content == "Let me search for that."

        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) == 1
        assert token_events[0].content == "Here is the answer."

    @pytest.mark.asyncio
    async def test_inter_tool_text_emitted_as_reasoning(self):
        """Text between tool calls is classified as reasoning."""
        token_stream = [
            (AIMessageChunk(content="Checking docs."), {}),
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-a", "name": "search", "args": '{"q":"a"}'}
                    ],
                ),
                {},
            ),
            (ToolMessage(content="result-a", tool_call_id="call-a", name="search"), {}),
            (AIMessageChunk(content="Now checking code."), {}),
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-b", "name": "read_file", "args": '{"f":"x.py"}'}
                    ],
                ),
                {},
            ),
            (ToolMessage(content="result-b", tool_call_id="call-b", name="read_file"), {}),
            (AIMessageChunk(content="Done."), {}),
        ]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard", include_reasoning=True),
            )
        ]

        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(reasoning_events) == 2
        assert reasoning_events[0].content == "Checking docs."
        assert reasoning_events[1].content == "Now checking code."

        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) == 1
        assert token_events[0].content == "Done."

    @pytest.mark.asyncio
    async def test_buffer_overflow_reclassifies_as_token(self):
        """When text exceeds reasoning_token_limit without a tool call, it becomes tokens."""
        long_text = "A" * 400  # Exceeds default 300 limit
        token_stream = [
            (AIMessageChunk(content=long_text), {}),
        ]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(
                    visibility="standard",
                    include_reasoning=True,
                    reasoning_token_limit=300,
                ),
            )
        ]

        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(reasoning_events) == 0

        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) >= 1

    @pytest.mark.asyncio
    async def test_no_tool_short_text_becomes_token(self):
        """Short text without any tool calls becomes token (flushed at end)."""
        token_stream = [
            (AIMessageChunk(content="Just a direct answer."), {}),
        ]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard", include_reasoning=True),
            )
        ]

        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(reasoning_events) == 0

        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) == 1
        assert token_events[0].content == "Just a direct answer."

    @pytest.mark.asyncio
    async def test_reasoning_disabled(self):
        """When include_reasoning=False, all text is emitted as token events."""
        token_stream = [
            (AIMessageChunk(content="Thinking..."), {}),
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-d", "name": "search", "args": '{"q":"x"}'}
                    ],
                ),
                {},
            ),
            (ToolMessage(content="ok", tool_call_id="call-d", name="search"), {}),
            (AIMessageChunk(content="Answer."), {}),
        ]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard", include_reasoning=False),
            )
        ]

        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(reasoning_events) == 0

        token_events = [e for e in events if e.type == "token"]
        assert len(token_events) == 2
