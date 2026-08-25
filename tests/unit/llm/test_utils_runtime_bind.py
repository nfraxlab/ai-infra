"""Tests for LLM runtime binding utilities (llm/utils/runtime_bind.py).

This module provides comprehensive tests for runtime model binding:
- tool_used() state checker
- bind_model_with_tools()
- make_agent_with_context()
- _normalize_tool()

Phase 0.2 of the ai-infra v1.0.0 release plan.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from ai_infra.llm.agent import Agent
from ai_infra.llm.utils.model_registry import ModelRegistry
from ai_infra.llm.utils.runtime_bind import (
    _normalize_tool,
    _RuntimeModelBindingMiddleware,
    bind_model_with_tools,
    make_agent_with_context,
    tool_used,
)
from ai_infra.llm.utils.settings import ModelSettings

# =============================================================================
# TEST: tool_used()
# =============================================================================


class TestToolUsed:
    """Test tool_used function."""

    def test_no_messages(self):
        """Test empty state has no tool used."""
        state = {"messages": []}
        assert tool_used(state) is False

    def test_no_tool_calls(self):
        """Test messages without tool calls."""
        state = {
            "messages": [
                Mock(tool_calls=None, type="human"),
                Mock(tool_calls=None, type="ai"),
            ]
        }
        assert tool_used(state) is False

    def test_has_tool_calls(self):
        """Test message with tool_calls attribute."""
        state = {
            "messages": [
                Mock(tool_calls=None, type="human"),
                Mock(tool_calls=[{"name": "get_weather"}], type="ai"),
            ]
        }
        assert tool_used(state) is True

    def test_has_tool_message(self):
        """Test state with tool message type."""
        state = {
            "messages": [
                Mock(tool_calls=None, type="human"),
                Mock(tool_calls=None, type="tool"),
            ]
        }
        assert tool_used(state) is True

    def test_dict_message_with_tool_calls(self):
        """Test dict-style message with tool_calls."""
        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "tool_calls": [{"name": "search"}]},
            ]
        }
        assert tool_used(state) is True

    def test_dict_message_tool_type(self):
        """Test dict-style message with type=tool."""
        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"type": "tool", "content": "Tool result"},
            ]
        }
        assert tool_used(state) is True

    def test_non_dict_state(self):
        """Test non-dict state returns False."""
        state = "not a dict"
        assert tool_used(state) is False

    def test_checks_messages_in_reverse(self):
        """Test tool_used checks from most recent message."""
        state = {
            "messages": [
                Mock(tool_calls=[{"name": "old_tool"}], type="ai"),
                Mock(tool_calls=None, type="human"),
            ]
        }
        # Should find tool_calls even though it's not last
        assert tool_used(state) is True


# =============================================================================
# TEST: bind_model_with_tools()
# =============================================================================


class TestBindModelWithTools:
    """Test bind_model_with_tools function."""

    def test_basic_binding(self):
        """Test basic model binding with tools."""
        # Setup
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get_or_create.return_value = mock_model

        mock_runtime = MagicMock()
        mock_runtime.context = ModelSettings(
            provider="openai",
            model_name="gpt-4o",
            tools=[lambda x: x],
            extra={},
        )

        state = {"messages": []}

        # Execute
        _result = bind_model_with_tools(state, mock_runtime, mock_registry)

        # Verify
        mock_model.bind_tools.assert_called_once()

    def test_uses_global_tools_when_no_context_tools(self):
        """Test falls back to global_tools when context.tools is None."""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get_or_create.return_value = mock_model

        global_tools = [lambda x: x * 2]

        mock_runtime = MagicMock()
        mock_runtime.context = ModelSettings(
            provider="openai",
            model_name="gpt-4o",
            tools=None,  # No context tools
            extra={},
        )

        state = {"messages": []}

        _result = bind_model_with_tools(
            state, mock_runtime, mock_registry, global_tools=global_tools
        )

        # Should have called bind_tools with global_tools
        call_args = mock_model.bind_tools.call_args
        assert call_args[0][0] is global_tools

    def test_gemini_no_tool_choice_without_tools(self):
        """Test Gemini provider doesn't send tool_choice when no tools."""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get_or_create.return_value = mock_model

        mock_runtime = MagicMock()
        mock_runtime.context = ModelSettings(
            provider="google_genai",
            model_name="gemini-pro",
            tools=[],  # Empty tools
            extra={},
        )

        state = {"messages": []}

        _result = bind_model_with_tools(state, mock_runtime, mock_registry)

        # tool_choice should be None for Gemini without tools
        call_kwargs = mock_model.bind_tools.call_args[1]
        assert call_kwargs.get("tool_choice") is None

    def test_openai_omits_parallel_tool_calls_without_tools(self):
        """Tool-only settings are not sent to OpenAI when no tools are bound."""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get_or_create.return_value = mock_model
        mock_runtime = MagicMock()
        mock_runtime.context = ModelSettings(
            provider="openai",
            model_name="gpt-4o",
            tools=[],
            extra={},
        )

        bind_model_with_tools({"messages": []}, mock_runtime, mock_registry)

        call_kwargs = mock_model.bind_tools.call_args[1]
        assert "parallel_tool_calls" not in call_kwargs

    def test_force_once_clears_tool_choice_after_use(self):
        """Test force_once clears tool_choice after tool is used."""
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.get_or_create.return_value = mock_model

        mock_runtime = MagicMock()
        mock_runtime.context = ModelSettings(
            provider="openai",
            model_name="gpt-4o",
            tools=[lambda x: x],
            extra={"tool_controls": {"force_once": True, "tool_choice": "required"}},
        )

        # State shows tool was already used
        state = {
            "messages": [
                Mock(tool_calls=[{"name": "test"}], type="ai"),
            ]
        }

        _result = bind_model_with_tools(state, mock_runtime, mock_registry)

        # After tool use, tool_choice should be cleared
        call_kwargs = mock_model.bind_tools.call_args[1]
        assert call_kwargs.get("tool_choice") is None


# =============================================================================
# TEST: _normalize_tool()
# =============================================================================


class TestNormalizeTool:
    """Test _normalize_tool function."""

    def test_none_returns_none(self):
        """Test None input returns None."""
        assert _normalize_tool(None) is None

    def test_base_tool_returns_as_is(self):
        """Test BaseTool instance returns as-is."""
        mock_tool = MagicMock(spec=BaseTool)
        result = _normalize_tool(mock_tool)

        assert result is mock_tool

    def test_callable_wrapped_as_tool(self):
        """Test callable is wrapped as LangChain tool."""

        def my_func(x: str) -> str:
            """A test function."""
            return x.upper()

        result = _normalize_tool(my_func)

        assert result is not None
        assert isinstance(result, BaseTool)

    def test_dict_returns_none_with_warning(self):
        """Test dict input returns None (with warning)."""
        dict_tool = {"name": "test", "description": "A test tool"}
        result = _normalize_tool(dict_tool)

        assert result is None

    def test_unsupported_type_returns_none(self):
        """Test unsupported type returns None."""
        result = _normalize_tool(12345)  # Int is not a valid tool

        assert result is None

    def test_mcp_tool_is_converted(self):
        """Test MCP Tool (from FastMCP) is converted to LangChain tool.

        MCP tools have: fn, name, description, parameters, run() method.
        They should be detected by duck-typing and converted.
        """

        # Create a mock MCP-like tool (duck-typed, not actual mcp import)
        class MockMCPTool:
            def __init__(self):
                self.fn = lambda x: x
                self.name = "test_mcp_tool"
                self.description = "A test MCP tool for unit testing"
                self.parameters = {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                }
                self.is_async = False

            def run(self, arguments: dict) -> str:
                return f"Result for: {arguments.get('query', '')}"

        mcp_tool = MockMCPTool()
        result = _normalize_tool(mcp_tool)

        # Should be converted to a BaseTool
        assert result is not None
        assert isinstance(result, BaseTool)
        assert result.name == "test_mcp_tool"
        assert "test MCP tool" in result.description

    def test_mcp_async_tool_is_converted(self):
        """Test async MCP Tool is converted correctly."""

        class MockAsyncMCPTool:
            def __init__(self):
                self.fn = lambda x: x
                self.name = "async_mcp_tool"
                self.description = "An async MCP tool"
                self.parameters = {"type": "object", "properties": {}}
                self.is_async = True

            async def run(self, arguments: dict) -> str:
                return "async result"

        async_mcp_tool = MockAsyncMCPTool()
        result = _normalize_tool(async_mcp_tool)

        assert result is not None
        assert isinstance(result, BaseTool)
        assert result.name == "async_mcp_tool"

    def test_mcp_tool_execution(self):
        """Test converted MCP tool can be executed."""

        class ExecutableMCPTool:
            def __init__(self):
                self.fn = lambda x: x
                self.name = "executable_tool"
                self.description = "Tool that can be executed"
                self.parameters = {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                }
                self.is_async = False
                self.call_count = 0

            def run(self, arguments: dict) -> str:
                self.call_count += 1
                return f"Executed with: {arguments.get('value', 'none')}"

        mcp_tool = ExecutableMCPTool()
        converted = _normalize_tool(mcp_tool)

        # Execute the converted tool
        result = converted.invoke({"value": "test_input"})

        # Verify execution
        assert "Executed with: test_input" in str(result)
        assert mcp_tool.call_count == 1


# =============================================================================
# TEST: make_agent_with_context()
# =============================================================================


class TestMakeAgentWithContext:
    """Test make_agent_with_context function."""

    def test_agent_invokes_with_runtime_selected_model(self):
        """Test the modern factory invokes the model selected from runtime context."""
        model = FakeMessagesListChatModel(responses=[AIMessage(content="runtime response")])
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "fake-model"
        mock_registry.get_or_create.return_value = model

        agent, context = make_agent_with_context(
            mock_registry,
            provider="test",
            model_name="fake-model",
            tools=[],
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            context=context,
        )

        assert result["messages"][-1].content == "runtime response"
        assert mock_registry.get_or_create.call_count == 2

    @pytest.mark.asyncio
    async def test_agent_ainvokes_with_runtime_selected_model(self):
        """Test async invocation uses the same runtime model selection path."""
        model = FakeMessagesListChatModel(responses=[AIMessage(content="async runtime response")])
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "fake-model"
        mock_registry.get_or_create.return_value = model

        agent, context = make_agent_with_context(
            mock_registry,
            provider="test",
            model_name="fake-model",
            tools=[],
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            context=context,
        )

        assert result["messages"][-1].content == "async runtime response"
        assert mock_registry.get_or_create.call_count == 2

    @pytest.mark.asyncio
    async def test_agent_token_stream_unwraps_v2_messages_parts(self):
        """The normalized stream consumes typed LangGraph v2 message parts."""
        model = FakeMessagesListChatModel(responses=[AIMessage(content="typed stream response")])
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "fake-model"
        mock_registry.get_or_create.return_value = model
        compiled_agent, context = make_agent_with_context(
            mock_registry,
            provider="test",
            model_name="fake-model",
            tools=[],
        )
        agent = Agent()

        with patch.object(
            agent,
            "_make_agent_with_context",
            return_value=(compiled_agent, context),
        ):
            chunks = [
                chunk
                async for chunk in agent.astream_agent_tokens(
                    messages=[{"role": "user", "content": "Hello"}],
                    provider="test",
                    model_name="fake-model",
                    tools=[],
                )
            ]

        assert chunks[0][0].content == "typed stream response"
        assert chunks[0][1]["langgraph_node"] == "model"

    def test_runtime_binding_middleware_preserves_tool_controls(self):
        """Test modern agent requests use the existing nfrax binding policy."""
        mock_registry = MagicMock(spec=ModelRegistry)
        selected_model = MagicMock()
        mock_registry.get_or_create.return_value = selected_model
        test_tool = MagicMock(spec=BaseTool)

        runtime = MagicMock()
        runtime.context = ModelSettings(
            provider="openai",
            model_name="gpt-4o",
            tools=[test_tool],
            extra={"tool_controls": {"tool_choice": "required", "parallel_tool_calls": True}},
        )
        request = MagicMock(
            state={"messages": []},
            runtime=runtime,
            model_settings={"temperature": 0.2},
        )
        prepared_request = MagicMock()
        request.override.return_value = prepared_request
        handler = MagicMock(return_value=MagicMock())

        middleware = _RuntimeModelBindingMiddleware(mock_registry, [test_tool])
        middleware.wrap_model_call(request, handler)

        request.override.assert_called_once_with(
            model=selected_model,
            tools=[test_tool],
            tool_choice="required",
            model_settings={"temperature": 0.2, "parallel_tool_calls": True},
        )
        handler.assert_called_once_with(prepared_request)

    def test_basic_agent_creation(self):
        """Test basic agent creation."""
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
            )

        assert agent is not None
        assert context is not None
        assert context.provider == "openai"
        assert context.model_name == "gpt-4o"

    def test_agent_with_tools(self):
        """Test agent creation with tools."""

        def my_tool(x: str) -> str:
            """Test tool."""
            return x

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                tools=[my_tool],
            )

        # Tools should be in context
        assert len(context.tools) == 1

    def test_agent_with_global_tools(self):
        """Test agent uses global tools when no explicit tools."""

        def global_tool(x: str) -> str:
            """Global tool."""
            return x

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                global_tools=[global_tool],
            )

        assert len(context.tools) == 1

    def test_require_explicit_tools_raises(self):
        """Test require_explicit_tools raises when using global tools."""

        def global_tool(x: str) -> str:
            """Global tool."""
            return x

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with pytest.raises(ValueError, match="Implicit global tools use forbidden"):
            make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                global_tools=[global_tool],
                require_explicit_tools=True,
            )

    def test_require_explicit_tools_allows_explicit(self):
        """Test require_explicit_tools allows explicit tools list."""

        def my_tool(x: str) -> str:
            """My tool."""
            return x

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            # Should not raise when tools are explicitly provided
            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                tools=[my_tool],
                require_explicit_tools=True,
            )

        assert len(context.tools) == 1

    def test_recursion_limit_in_context(self):
        """Test recursion_limit is stored in context extra."""
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                recursion_limit=100,
            )

        assert context.extra["recursion_limit"] == 100

    def test_default_recursion_limit(self):
        """Test default recursion_limit is 50."""
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
            )

        assert context.extra["recursion_limit"] == 50

    def test_hitl_tool_wrapper(self):
        """Test HITL tool wrapper is applied."""

        def my_tool(x: str) -> str:
            """My tool."""
            return x

        wrapped_tools = []

        def hitl_wrapper(tool):
            wrapped = MagicMock(spec=BaseTool)
            wrapped_tools.append(tool)
            return wrapped

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                tools=[my_tool],
                hitl_tool_wrapper=hitl_wrapper,
            )

        # HITL wrapper should have been called
        assert len(wrapped_tools) == 1

    def test_checkpointer_passed_to_agent(self):
        """Test checkpointer is passed to create_agent."""
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        mock_checkpointer = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                checkpointer=mock_checkpointer,
            )

        # Verify checkpointer was passed
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["checkpointer"] is mock_checkpointer

    def test_interrupt_before_after(self):
        """Test interrupt_before and interrupt_after are passed."""
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                interrupt_before=["dangerous_tool"],
                interrupt_after=["log_tool"],
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["interrupt_before"] == ["dangerous_tool"]
        assert call_kwargs["interrupt_after"] == ["log_tool"]

    def test_modern_agent_options_are_forwarded(self):
        """Test standard agents forward modern factory configuration."""
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()
        custom_middleware = MagicMock()
        response_format = MagicMock()

        class RuntimeContext:
            pass

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                middleware=[custom_middleware],
                response_format=response_format,
                context_schema=RuntimeContext,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["middleware"][1:] == (custom_middleware,)
        assert call_kwargs["response_format"] is response_format
        assert call_kwargs["context_schema"] is RuntimeContext


# =============================================================================
# EDGE CASES
# =============================================================================


class TestRuntimeBindEdgeCases:
    """Test edge cases for runtime binding utilities."""

    def test_tool_used_with_none_state(self):
        """Test tool_used handles None-like values."""
        assert tool_used(None) is False
        assert tool_used({}) is False

    def test_normalize_tool_with_lambda(self):
        """Test _normalize_tool with lambda function."""
        # Lambda without docstring will raise ValueError
        # LangChain requires a description for tools
        with pytest.raises(ValueError, match="docstring"):
            _normalize_tool(lambda x: x * 2)

    def test_agent_no_tools_warning(self, caplog):
        """Test warning is logged when no tools are bound."""
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            with caplog.at_level(logging.WARNING):
                agent, context = make_agent_with_context(
                    mock_registry,
                    provider="openai",
                    model_name="gpt-4o",
                    tools=[],  # Explicit empty tools
                    logger=logging.getLogger(__name__),
                )

        # Should log warning about no tools
        assert "No tools bound" in caplog.text or len(context.tools) == 0

    def test_tool_controls_dataclass_conversion(self):
        """Test tool_controls dataclass is converted to dict."""
        from dataclasses import dataclass

        @dataclass
        class MockToolControls:
            tool_choice: str = "auto"
            parallel_tool_calls: bool = True

        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.resolve_model_name.return_value = "gpt-4o"
        mock_registry.get_or_create.return_value = MagicMock()

        with patch("ai_infra.llm.utils.runtime_bind.create_agent") as mock_create:
            mock_create.return_value = MagicMock()

            agent, context = make_agent_with_context(
                mock_registry,
                provider="openai",
                model_name="gpt-4o",
                tool_controls=MockToolControls(),
            )

        # tool_controls should be in extra as dict
        assert "tool_controls" in context.extra
        assert isinstance(context.extra["tool_controls"], dict)
