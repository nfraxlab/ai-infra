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
from ai_infra.llm.streaming import StreamConfig, StreamEvent, should_emit_event

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


# =============================================================================
# Stream Reasoning Immediately Tests
# =============================================================================


class TestStreamReasoningImmediately:
    """Tests for stream_reasoning_immediately mode.

    When enabled, pre-tool text is emitted as reasoning events immediately
    (no buffering).  There is NO char-limit auto-transition to tokens —
    the caller is responsible for reclassifying at stream end.
    """

    @pytest.mark.asyncio
    async def test_all_text_stays_reasoning_no_tools(self):
        """Without tools, all text is reasoning — no automatic transition to tokens."""
        long_text = "A" * 600  # Well past default 300 limit
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
                    visibility="detailed",
                    include_reasoning=True,
                    reasoning_token_limit=300,
                    stream_reasoning_immediately=True,
                ),
            )
        ]

        reasoning_events = [e for e in events if e.type == "reasoning"]
        token_events = [e for e in events if e.type == "token"]

        assert len(reasoning_events) == 1
        assert reasoning_events[0].content == long_text
        assert len(token_events) == 0  # No auto-transition to tokens

    @pytest.mark.asyncio
    async def test_post_tool_text_becomes_tokens(self):
        """After tool completes, subsequent text is emitted as tokens."""
        token_stream = [
            (AIMessageChunk(content="Let me check."), {}),
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-x", "name": "search", "args": '{"q":"test"}'}
                    ],
                ),
                {},
            ),
            (ToolMessage(content="found it", tool_call_id="call-x", name="search"), {}),
            (AIMessageChunk(content="Here is the answer."), {}),
        ]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(
                    visibility="detailed",
                    include_reasoning=True,
                    stream_reasoning_immediately=True,
                ),
            )
        ]

        reasoning_events = [e for e in events if e.type == "reasoning"]
        token_events = [e for e in events if e.type == "token"]

        assert len(reasoning_events) == 1
        assert reasoning_events[0].content == "Let me check."
        assert len(token_events) == 1
        assert token_events[0].content == "Here is the answer."

    @pytest.mark.asyncio
    async def test_no_duplication_long_response(self):
        """Long response without tools must NOT produce both reasoning and token events for same text."""
        chunks = [
            (AIMessageChunk(content="Part one. "), {}),
            (AIMessageChunk(content="Part two. "), {}),
            (AIMessageChunk(content="Part three."), {}),
        ]

        agent = MockStreamingAgent(chunks)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(
                    visibility="detailed",
                    include_reasoning=True,
                    reasoning_token_limit=10,  # Very low limit
                    stream_reasoning_immediately=True,
                ),
            )
        ]

        reasoning_events = [e for e in events if e.type == "reasoning"]
        token_events = [e for e in events if e.type == "token"]

        # ALL chunks should be reasoning, NONE as tokens — no mid-stream reclassify
        all_reasoning_text = "".join(e.content for e in reasoning_events)
        all_token_text = "".join(e.content for e in token_events)

        assert all_reasoning_text == "Part one. Part two. Part three."
        assert all_token_text == ""


# =============================================================================
# Rich Event Types (Phase 10.1) Tests
# =============================================================================


class TestRichEventTypes:
    """Tests for new StreamEvent types: usage, turn_start, turn_end, intent, todo."""

    def test_usage_event_construction(self):
        """Usage event carries token counts and model."""
        event = StreamEvent(
            type="usage",
            input_tokens=1200,
            output_tokens=340,
            cost=0.004,
            model="claude-sonnet-4-5",
        )
        assert event.type == "usage"
        assert event.input_tokens == 1200
        assert event.output_tokens == 340
        assert event.cost == 0.004
        assert event.model == "claude-sonnet-4-5"

    def test_turn_start_event_construction(self):
        """Turn start event carries turn_id."""
        event = StreamEvent(type="turn_start", turn_id=1)
        assert event.type == "turn_start"
        assert event.turn_id == 1

    def test_turn_end_event_construction(self):
        """Turn end event carries turn_id and tools_called."""
        event = StreamEvent(type="turn_end", turn_id=2, tools_called=3)
        assert event.type == "turn_end"
        assert event.turn_id == 2
        assert event.tools_called == 3

    def test_intent_event_construction(self):
        """Intent event carries human-readable description."""
        event = StreamEvent(type="intent", content="Searching codebase")
        assert event.type == "intent"
        assert event.content == "Searching codebase"

    def test_todo_event_construction(self):
        """Todo event carries a list of task items."""
        items = [
            {"id": 1, "title": "Fetch data", "status": "completed"},
            {"id": 2, "title": "Analyze results", "status": "in-progress"},
        ]
        event = StreamEvent(type="todo", todo_items=items)
        assert event.type == "todo"
        assert event.todo_items == items
        assert len(event.todo_items) == 2

    def test_new_fields_default_to_none(self):
        """New fields default to None on existing event types."""
        event = StreamEvent(type="token", content="hello")
        assert event.input_tokens is None
        assert event.output_tokens is None
        assert event.cost is None
        assert event.turn_id is None
        assert event.todo_items is None

    def test_usage_event_to_dict(self):
        """Usage event serializes all non-None fields."""
        event = StreamEvent(
            type="usage",
            input_tokens=500,
            output_tokens=100,
            cost=0.002,
            model="gpt-4.1",
        )
        d = event.to_dict()
        assert d["type"] == "usage"
        assert d["input_tokens"] == 500
        assert d["output_tokens"] == 100
        assert d["cost"] == 0.002
        assert d["model"] == "gpt-4.1"

    def test_todo_event_to_dict(self):
        """Todo event serializes todo_items."""
        items = [{"id": 1, "title": "Do thing", "status": "not-started"}]
        event = StreamEvent(type="todo", todo_items=items)
        d = event.to_dict()
        assert d["type"] == "todo"
        assert d["todo_items"] == items

    def test_turn_event_to_dict_excludes_none(self):
        """Turn events omit None fields in to_dict."""
        event = StreamEvent(type="turn_start", turn_id=3)
        d = event.to_dict()
        assert d == {"type": "turn_start", "turn_id": 3}
        assert "content" not in d
        assert "input_tokens" not in d


class TestRichEventVisibility:
    """Tests for visibility rules of new event types."""

    def test_usage_requires_detailed(self):
        """Usage events require detailed visibility."""
        assert should_emit_event("usage", "detailed") is True
        assert should_emit_event("usage", "debug") is True
        assert should_emit_event("usage", "standard") is False
        assert should_emit_event("usage", "minimal") is False

    def test_turn_start_requires_standard(self):
        """Turn start events require standard visibility."""
        assert should_emit_event("turn_start", "standard") is True
        assert should_emit_event("turn_start", "detailed") is True
        assert should_emit_event("turn_start", "minimal") is False

    def test_turn_end_requires_standard(self):
        """Turn end events require standard visibility."""
        assert should_emit_event("turn_end", "standard") is True
        assert should_emit_event("turn_end", "minimal") is False

    def test_intent_requires_standard(self):
        """Intent events require standard visibility."""
        assert should_emit_event("intent", "standard") is True
        assert should_emit_event("intent", "minimal") is False

    def test_todo_requires_standard(self):
        """Todo events require standard visibility."""
        assert should_emit_event("todo", "standard") is True
        assert should_emit_event("todo", "minimal") is False


# =============================================================================
# Usage Event Emission Tests (Phase 10.2)
# =============================================================================


class TestUsageEventEmission:
    """Tests for usage event emission from Agent.astream()."""

    @pytest.mark.asyncio
    async def test_usage_event_emitted_direct_answer(self):
        """Usage event is emitted after direct answer (no tools) at detailed visibility."""
        chunks = [
            AIMessageChunk(content="Hello world"),
            AIMessageChunk(
                content="!",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
        ]
        token_stream = [(chunk, {}) for chunk in chunks]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="detailed"),
            )
        ]

        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0].input_tokens == 10
        assert usage_events[0].output_tokens == 5
        assert usage_events[0].cost is None

    @pytest.mark.asyncio
    async def test_usage_event_before_tool_end(self):
        """Usage event is emitted before tool_end when tools are used."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "call-u1", "name": "search", "args": '{"q": "test"}'}
            ],
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )
        tool_result = ToolMessage(content="result", tool_call_id="call-u1", name="search")
        answer = AIMessageChunk(
            content="Here is your answer",
            usage_metadata={"input_tokens": 80, "output_tokens": 30, "total_tokens": 110},
        )

        token_stream = [(tool_call, {}), (tool_result, {}), (answer, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="detailed"),
            )
        ]

        usage_events = [e for e in events if e.type == "usage"]
        # Two usage events: one after tool call LLM response, one after final answer
        assert len(usage_events) == 2
        assert usage_events[0].input_tokens == 50
        assert usage_events[0].output_tokens == 20
        assert usage_events[1].input_tokens == 80
        assert usage_events[1].output_tokens == 30

    @pytest.mark.asyncio
    async def test_usage_event_ordering_relative_to_tool_end(self):
        """Usage event appears before tool_end in event sequence."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-o1", "name": "calc", "args": '{"x": 1}'}],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        tool_result = ToolMessage(content="2", tool_call_id="call-o1", name="calc")

        token_stream = [(tool_call, {}), (tool_result, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="detailed"),
            )
        ]

        event_types = [e.type for e in events]
        usage_idx = event_types.index("usage")
        tool_end_idx = event_types.index("tool_end")
        assert usage_idx < tool_end_idx

    @pytest.mark.asyncio
    async def test_usage_event_with_cost_calculation(self):
        """Usage event includes cost when StreamConfig has pricing."""
        chunks = [
            AIMessageChunk(
                content="answer",
                usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
            ),
        ]
        token_stream = [(chunk, {}) for chunk in chunks]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(
                    visibility="detailed",
                    cost_per_input_token=0.000003,
                    cost_per_output_token=0.000015,
                ),
            )
        ]

        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 1
        expected_cost = 1000 * 0.000003 + 200 * 0.000015
        assert usage_events[0].cost == pytest.approx(expected_cost)

    @pytest.mark.asyncio
    async def test_usage_event_not_emitted_at_standard(self):
        """Usage events are suppressed at standard visibility."""
        chunks = [
            AIMessageChunk(
                content="Hello",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
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

        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 0

    @pytest.mark.asyncio
    async def test_usage_accumulates_across_chunks(self):
        """Usage metadata is accumulated across multiple AIMessageChunks."""
        chunks = [
            AIMessageChunk(
                content="Hello",
                usage_metadata={"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
            ),
            AIMessageChunk(
                content=" world",
                usage_metadata={"input_tokens": 0, "output_tokens": 8, "total_tokens": 8},
            ),
        ]
        token_stream = [(chunk, {}) for chunk in chunks]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="detailed"),
            )
        ]

        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0].input_tokens == 10
        assert usage_events[0].output_tokens == 8

    @pytest.mark.asyncio
    async def test_no_usage_event_when_no_metadata(self):
        """No usage event when chunks lack usage_metadata."""
        chunks = [
            AIMessageChunk(content="Hello"),
            AIMessageChunk(content=" world"),
        ]
        token_stream = [(chunk, {}) for chunk in chunks]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="detailed"),
            )
        ]

        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 0

    @pytest.mark.asyncio
    async def test_usage_resets_between_tool_turns(self):
        """Usage accumulator resets after each tool turn, producing separate events."""
        # First LLM call: makes a tool call
        tool_call_1 = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "call-t1", "name": "search", "args": '{"q": "a"}'}
            ],
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        tool_result_1 = ToolMessage(content="result1", tool_call_id="call-t1", name="search")
        # Second LLM call: makes another tool call
        tool_call_2 = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-t2", "name": "calc", "args": '{"x": 1}'}],
            usage_metadata={"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        )
        tool_result_2 = ToolMessage(content="result2", tool_call_id="call-t2", name="calc")
        # Final LLM call: direct answer
        answer = AIMessageChunk(
            content="Final answer",
            usage_metadata={"input_tokens": 300, "output_tokens": 100, "total_tokens": 400},
        )

        token_stream = [
            (tool_call_1, {}),
            (tool_result_1, {}),
            (tool_call_2, {}),
            (tool_result_2, {}),
            (answer, {}),
        ]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="detailed"),
            )
        ]

        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 3
        # First turn
        assert usage_events[0].input_tokens == 100
        assert usage_events[0].output_tokens == 50
        # Second turn
        assert usage_events[1].input_tokens == 200
        assert usage_events[1].output_tokens == 80
        # Final answer
        assert usage_events[2].input_tokens == 300
        assert usage_events[2].output_tokens == 100

    def test_stream_config_cost_fields_default_none(self):
        """StreamConfig cost fields default to None."""
        cfg = StreamConfig()
        assert cfg.cost_per_input_token is None
        assert cfg.cost_per_output_token is None

    @pytest.mark.asyncio
    async def test_usage_event_includes_model(self):
        """Usage event includes the model name."""
        chunks = [
            AIMessageChunk(
                content="Hi",
                usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            ),
        ]
        token_stream = [(chunk, {}) for chunk in chunks]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                model_name="gpt-4o",
                stream_config=StreamConfig(visibility="detailed"),
            )
        ]

        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0].model == "gpt-4o"


# =============================================================================
# Turn Lifecycle Event Tests (Phase 10.3)
# =============================================================================


class TestTurnLifecycleEvents:
    """Tests for turn_start and turn_end event emission from Agent.astream()."""

    @pytest.mark.asyncio
    async def test_single_turn_direct_answer(self):
        """Direct answer (no tools) produces one turn_start and one turn_end."""
        chunks = [
            AIMessageChunk(content="Hello"),
            AIMessageChunk(content=" world"),
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

        turn_starts = [e for e in events if e.type == "turn_start"]
        turn_ends = [e for e in events if e.type == "turn_end"]
        assert len(turn_starts) == 1
        assert turn_starts[0].turn_id == 1
        assert len(turn_ends) == 1
        assert turn_ends[0].turn_id == 1
        assert turn_ends[0].tools_called == 0

    @pytest.mark.asyncio
    async def test_turn_with_tool_call(self):
        """Single turn with tool call emits turn_start, then turn_end with tools_called=1."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "call-tc1", "name": "search", "args": '{"q": "test"}'}
            ],
        )
        tool_result = ToolMessage(content="result", tool_call_id="call-tc1", name="search")
        answer = AIMessageChunk(content="Here is the answer")

        token_stream = [(tool_call, {}), (tool_result, {}), (answer, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard"),
            )
        ]

        turn_starts = [e for e in events if e.type == "turn_start"]
        turn_ends = [e for e in events if e.type == "turn_end"]
        # Turn 1: tool call, Turn 2: final answer
        assert len(turn_starts) == 2
        assert turn_starts[0].turn_id == 1
        assert turn_starts[1].turn_id == 2
        assert len(turn_ends) == 2
        assert turn_ends[0].turn_id == 1
        assert turn_ends[0].tools_called == 1
        assert turn_ends[1].turn_id == 2
        assert turn_ends[1].tools_called == 0

    @pytest.mark.asyncio
    async def test_multi_turn_tool_calls(self):
        """Multiple tool-calling turns produce sequential turn IDs."""
        # Turn 1: search tool
        tc1 = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "call-m1", "name": "search", "args": '{"q": "a"}'}
            ],
        )
        tr1 = ToolMessage(content="r1", tool_call_id="call-m1", name="search")
        # Turn 2: calc tool
        tc2 = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-m2", "name": "calc", "args": '{"x": 1}'}],
        )
        tr2 = ToolMessage(content="r2", tool_call_id="call-m2", name="calc")
        # Turn 3: final answer
        answer = AIMessageChunk(content="Done")

        token_stream = [
            (tc1, {}),
            (tr1, {}),
            (tc2, {}),
            (tr2, {}),
            (answer, {}),
        ]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard"),
            )
        ]

        turn_starts = [e for e in events if e.type == "turn_start"]
        turn_ends = [e for e in events if e.type == "turn_end"]
        assert len(turn_starts) == 3
        assert [e.turn_id for e in turn_starts] == [1, 2, 3]
        assert len(turn_ends) == 3
        assert [e.turn_id for e in turn_ends] == [1, 2, 3]
        assert turn_ends[0].tools_called == 1
        assert turn_ends[1].tools_called == 1
        assert turn_ends[2].tools_called == 0

    @pytest.mark.asyncio
    async def test_multiple_tools_in_single_turn(self):
        """Multiple tools called in one LLM turn count correctly."""
        tc = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "call-p1", "name": "search", "args": '{"q": "a"}'},
                {"index": 1, "id": "call-p2", "name": "calc", "args": '{"x": 1}'},
            ],
        )
        tr1 = ToolMessage(content="r1", tool_call_id="call-p1", name="search")
        tr2 = ToolMessage(content="r2", tool_call_id="call-p2", name="calc")
        answer = AIMessageChunk(content="Answer")

        token_stream = [(tc, {}), (tr1, {}), (tr2, {}), (answer, {})]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="standard"),
            )
        ]

        turn_starts = [e for e in events if e.type == "turn_start"]
        turn_ends = [e for e in events if e.type == "turn_end"]
        # Turn 1: 2 tools, Turn 2: final answer
        assert len(turn_starts) == 2
        assert len(turn_ends) == 2
        assert turn_ends[0].tools_called == 2
        assert turn_ends[1].tools_called == 0

    @pytest.mark.asyncio
    async def test_turn_events_not_emitted_at_minimal(self):
        """Turn events suppressed at minimal visibility."""
        chunks = [AIMessageChunk(content="Hello")]
        token_stream = [(chunk, {}) for chunk in chunks]

        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="minimal"),
            )
        ]

        turn_events = [e for e in events if e.type in ("turn_start", "turn_end")]
        assert len(turn_events) == 0

    @pytest.mark.asyncio
    async def test_turn_start_before_tokens(self):
        """turn_start is emitted before any token events in the turn."""
        chunks = [AIMessageChunk(content="Hello world")]
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

        event_types = [e.type for e in events]
        turn_start_idx = event_types.index("turn_start")
        first_token_idx = event_types.index("token")
        assert turn_start_idx < first_token_idx

    @pytest.mark.asyncio
    async def test_turn_end_before_done(self):
        """turn_end is emitted before the done event."""
        chunks = [AIMessageChunk(content="Hello")]
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

        event_types = [e.type for e in events]
        turn_end_idx = event_types.index("turn_end")
        done_idx = event_types.index("done")
        assert turn_end_idx < done_idx

    @pytest.mark.asyncio
    async def test_tool_turn_end_after_tool_end(self):
        """turn_end for a tool turn comes after the last tool_end."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "call-te1", "name": "search", "args": '{"q": "x"}'}
            ],
        )
        tool_result = ToolMessage(content="result", tool_call_id="call-te1", name="search")

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

        event_types = [e.type for e in events]
        tool_end_idx = event_types.index("tool_end")
        # First turn_end comes after tool_end
        turn_end_idx = event_types.index("turn_end")
        assert turn_end_idx > tool_end_idx


# =============================================================================
# 10.5 — Todo Events
# =============================================================================


class TestTodoEvents:
    """Tests for todo event emission from manage_todos tool results."""

    @pytest.mark.asyncio
    async def test_todo_event_emitted_for_manage_todos(self):
        """Todo event is emitted when manage_todos tool completes with items."""
        import json

        todo_items = [
            {"id": 1, "title": "Fetch data", "status": "completed"},
            {"id": 2, "title": "Parse results", "status": "in-progress"},
        ]
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-td1", "name": "manage_todos", "args": "{}"}],
        )
        tool_result = ToolMessage(
            content=json.dumps(todo_items),
            tool_call_id="call-td1",
            name="manage_todos",
        )

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

        todo_events = [e for e in events if e.type == "todo"]
        assert len(todo_events) == 1
        assert todo_events[0].todo_items == todo_items

    @pytest.mark.asyncio
    async def test_todo_event_not_emitted_for_other_tools(self):
        """Todo event is NOT emitted for non-todo tools."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "call-td2", "name": "search", "args": '{"q": "x"}'}
            ],
        )
        tool_result = ToolMessage(
            content='[{"id": 1, "title": "T", "status": "done"}]',
            tool_call_id="call-td2",
            name="search",
        )

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

        todo_events = [e for e in events if e.type == "todo"]
        assert len(todo_events) == 0

    @pytest.mark.asyncio
    async def test_todo_event_with_dict_format(self):
        """Todo event handles dict format with 'items' key."""
        import json

        todo_data = {"items": [{"id": 1, "title": "Do X", "status": "not-started"}]}
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-td3", "name": "manage_todos", "args": "{}"}],
        )
        tool_result = ToolMessage(
            content=json.dumps(todo_data),
            tool_call_id="call-td3",
            name="manage_todos",
        )

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

        todo_events = [e for e in events if e.type == "todo"]
        assert len(todo_events) == 1
        assert todo_events[0].todo_items == todo_data["items"]

    @pytest.mark.asyncio
    async def test_todo_event_not_emitted_for_empty_list(self):
        """No todo event when manage_todos returns empty list."""
        import json

        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-td4", "name": "manage_todos", "args": "{}"}],
        )
        tool_result = ToolMessage(
            content=json.dumps([]),
            tool_call_id="call-td4",
            name="manage_todos",
        )

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

        todo_events = [e for e in events if e.type == "todo"]
        assert len(todo_events) == 0

    @pytest.mark.asyncio
    async def test_todo_event_not_emitted_for_invalid_json(self):
        """No todo event when manage_todos returns unparseable content."""
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-td5", "name": "manage_todos", "args": "{}"}],
        )
        tool_result = ToolMessage(
            content="not json at all",
            tool_call_id="call-td5",
            name="manage_todos",
        )

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

        todo_events = [e for e in events if e.type == "todo"]
        assert len(todo_events) == 0

    @pytest.mark.asyncio
    async def test_todo_event_respects_visibility(self):
        """Todo event requires standard+ visibility (not emitted at minimal)."""
        import json

        todo_items = [{"id": 1, "title": "T", "status": "done"}]
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-td6", "name": "manage_todos", "args": "{}"}],
        )
        tool_result = ToolMessage(
            content=json.dumps(todo_items),
            tool_call_id="call-td6",
            name="manage_todos",
        )

        token_stream = [(tool_call, {}), (tool_result, {})]
        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(visibility="minimal"),
            )
        ]

        todo_events = [e for e in events if e.type == "todo"]
        assert len(todo_events) == 0

    @pytest.mark.asyncio
    async def test_custom_todo_tool_name(self):
        """StreamConfig.todo_tool_name overrides default tool name."""
        import json

        todo_items = [{"id": 1, "title": "T", "status": "done"}]
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[{"index": 0, "id": "call-td7", "name": "update_tasks", "args": "{}"}],
        )
        tool_result = ToolMessage(
            content=json.dumps(todo_items),
            tool_call_id="call-td7",
            name="update_tasks",
        )

        token_stream = [(tool_call, {}), (tool_result, {})]
        agent = MockStreamingAgent(token_stream)
        events = [
            event
            async for event in agent.astream(
                "hello",
                provider="openai",
                stream_config=StreamConfig(
                    visibility="standard",
                    todo_tool_name="update_tasks",
                ),
            )
        ]

        todo_events = [e for e in events if e.type == "todo"]
        assert len(todo_events) == 1
        assert todo_events[0].todo_items == todo_items

    def test_stream_config_todo_tool_name_default(self):
        """StreamConfig.todo_tool_name defaults to 'manage_todos'."""
        cfg = StreamConfig()
        assert cfg.todo_tool_name == "manage_todos"
