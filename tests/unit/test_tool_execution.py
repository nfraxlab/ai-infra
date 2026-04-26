"""Tests for tool execution configuration (error handling, timeout, validation)."""

import asyncio
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool as lc_tool

from ai_infra.llm.tools.hitl import (
    ToolExecutionConfig,
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
    wrap_tool_with_execution_config,
)


# Test tools
def successful_tool(x: int) -> str:
    """A tool that succeeds."""
    return f"Result: {x * 2}"


def failing_tool(x: int) -> str:
    """A tool that always fails."""
    raise ValueError("Something went wrong")


def slow_tool(x: int) -> str:
    """A tool that takes time."""
    import time

    time.sleep(2)
    return f"Slow result: {x}"


async def async_slow_tool(x: int) -> str:
    """An async tool that takes time."""
    await asyncio.sleep(2)
    return f"Async slow result: {x}"


def wrong_type_tool(x: int) -> str:
    """A tool that returns wrong type."""
    return 123  # Should be str!


class TestToolExecutionConfig:
    """Test ToolExecutionConfig dataclass."""

    def test_default_values(self):
        config = ToolExecutionConfig()
        assert config.on_error == "return_error"
        assert config.max_retries == 1
        assert config.timeout is None
        assert config.validate_results is False
        assert config.on_timeout == "return_error"

    def test_custom_values(self):
        config = ToolExecutionConfig(
            on_error="retry",
            max_retries=3,
            timeout=30.0,
            validate_results=True,
            on_timeout="abort",
        )
        assert config.on_error == "retry"
        assert config.max_retries == 3
        assert config.timeout == 30.0
        assert config.validate_results is True
        assert config.on_timeout == "abort"

    def test_invalid_max_retries(self):
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            ToolExecutionConfig(max_retries=-1)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout must be > 0"):
            ToolExecutionConfig(timeout=0)
        with pytest.raises(ValueError, match="timeout must be > 0"):
            ToolExecutionConfig(timeout=-1)


class TestWrapToolWithExecutionConfig:
    """Test wrap_tool_with_execution_config function."""

    def test_no_config_returns_original(self):
        """When config is None, return the original tool."""
        result = wrap_tool_with_execution_config(successful_tool, None)
        assert result is successful_tool

    def test_wrap_function(self):
        """Wrapping a function works."""
        config = ToolExecutionConfig()
        wrapped = wrap_tool_with_execution_config(successful_tool, config)
        # Should be a wrapped tool, not the original function
        assert wrapped is not successful_tool
        # Should be invokable
        result = wrapped.invoke({"x": 5})
        assert result == "Result: 10"

    def test_wrap_langchain_tool(self):
        """Wrapping a LangChain tool works."""

        @lc_tool
        def my_lc_tool(x: int) -> str:
            """A LangChain tool."""
            return f"LC Result: {x}"

        config = ToolExecutionConfig()
        wrapped = wrap_tool_with_execution_config(my_lc_tool, config)
        result = wrapped.invoke({"x": 5})
        assert result == "LC Result: 5"

    def test_visualization_results_bypass_generic_truncation(self):
        """create_visualization payloads keep full prefixed HTML results."""
        from langchain_core.tools import BaseTool

        from ai_infra.llm.tools.hitl import _ExecutionConfigWrappedTool

        base_tool = MagicMock(spec=BaseTool)
        base_tool.name = "create_visualization"
        base_tool.description = "Create a visualization"

        config = ToolExecutionConfig(max_result_chars=100)
        wrapped = _ExecutionConfigWrappedTool(base_tool, config)

        html = "<!--PULSE_VIZ-->" + ("X" * 1000)

        preserved = wrapped._truncate_result(html)

        assert preserved == html
        assert "[TRUNCATED:" not in preserved


class TestErrorHandling:
    """Test error handling modes."""

    def test_return_error_mode(self):
        """on_error='return_error' returns error message to agent."""
        config = ToolExecutionConfig(on_error="return_error")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)
        result = wrapped.invoke({"x": 5})
        assert "[Tool Error: failing_tool]" in result
        assert "ValueError" in result
        assert "Something went wrong" in result

    def test_abort_mode(self):
        """on_error='abort' raises ToolExecutionError."""
        config = ToolExecutionConfig(on_error="abort")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)
        with pytest.raises(ToolExecutionError) as exc_info:
            wrapped.invoke({"x": 5})
        assert exc_info.value.tool_name == "failing_tool"
        assert isinstance(exc_info.value.original_error, ValueError)

    def test_retry_mode_eventual_failure(self):
        """on_error='retry' retries and then returns error."""
        call_count = 0

        def counting_failing_tool(x: int) -> str:
            """A tool that counts calls and fails."""
            nonlocal call_count
            call_count += 1
            raise ValueError("Still failing")

        config = ToolExecutionConfig(on_error="retry", max_retries=2)
        wrapped = wrap_tool_with_execution_config(counting_failing_tool, config)
        result = wrapped.invoke({"x": 5})

        # Should have tried 3 times (1 initial + 2 retries)
        assert call_count == 3
        assert "[Tool Error:" in result


class TestTimeoutHandling:
    """Test timeout handling."""

    def test_timeout_returns_error_by_default(self):
        """Timeout with on_timeout='return_error' returns message."""
        config = ToolExecutionConfig(timeout=0.1)  # Very short timeout
        wrapped = wrap_tool_with_execution_config(slow_tool, config)
        result = wrapped.invoke({"x": 5})
        assert "[Tool Timeout: slow_tool]" in result
        assert "timed out" in result

    def test_timeout_abort_raises(self):
        """Timeout with on_timeout='abort' raises ToolTimeoutError."""
        config = ToolExecutionConfig(timeout=0.1, on_timeout="abort")
        wrapped = wrap_tool_with_execution_config(slow_tool, config)
        with pytest.raises(ToolTimeoutError) as exc_info:
            wrapped.invoke({"x": 5})
        assert exc_info.value.tool_name == "slow_tool"


class TestAsyncErrorHandling:
    """Test async error handling."""

    @pytest.mark.asyncio
    async def test_async_return_error_mode(self):
        """Async on_error='return_error' returns error message."""
        config = ToolExecutionConfig(on_error="return_error")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)
        result = await wrapped.ainvoke({"x": 5})
        assert "[Tool Error: failing_tool]" in result

    @pytest.mark.asyncio
    async def test_async_abort_mode(self):
        """Async on_error='abort' raises ToolExecutionError."""
        config = ToolExecutionConfig(on_error="abort")
        wrapped = wrap_tool_with_execution_config(failing_tool, config)
        with pytest.raises(ToolExecutionError):
            await wrapped.ainvoke({"x": 5})

    @pytest.mark.asyncio
    async def test_async_timeout(self):
        """Async timeout returns error message."""
        config = ToolExecutionConfig(timeout=0.1)
        wrapped = wrap_tool_with_execution_config(async_slow_tool, config)
        result = await wrapped.ainvoke({"x": 5})
        assert "[Tool Timeout:" in result

    @pytest.mark.asyncio
    async def test_async_timeout_abort(self):
        """Async timeout with abort raises ToolTimeoutError."""
        config = ToolExecutionConfig(timeout=0.1, on_timeout="abort")
        wrapped = wrap_tool_with_execution_config(async_slow_tool, config)
        with pytest.raises(ToolTimeoutError):
            await wrapped.ainvoke({"x": 5})


class TestResultValidation:
    """Test tool result validation."""

    def test_validation_disabled_by_default(self):
        """Validation is disabled by default."""
        config = ToolExecutionConfig()
        wrapped = wrap_tool_with_execution_config(wrong_type_tool, config, expected_return_type=str)
        # Should not raise even though wrong type returned
        result = wrapped.invoke({"x": 5})
        assert result == 123

    def test_validation_enabled_raises(self):
        """With validate_results=True, wrong type raises ToolValidationError."""
        config = ToolExecutionConfig(validate_results=True)
        wrapped = wrap_tool_with_execution_config(wrong_type_tool, config, expected_return_type=str)
        with pytest.raises(ToolValidationError) as exc_info:
            wrapped.invoke({"x": 5})
        assert exc_info.value.tool_name == "wrong_type_tool"
        assert "int" in str(exc_info.value)  # Actual type
        assert "str" in str(exc_info.value)  # Expected type

    def test_validation_passes_correct_type(self):
        """Validation passes when types match."""
        config = ToolExecutionConfig(validate_results=True)
        wrapped = wrap_tool_with_execution_config(successful_tool, config, expected_return_type=str)
        result = wrapped.invoke({"x": 5})
        assert result == "Result: 10"

    def test_validation_auto_extracts_return_type(self):
        """Return type is extracted from function annotations."""
        config = ToolExecutionConfig(validate_results=True)
        wrapped = wrap_tool_with_execution_config(wrong_type_tool, config)
        # Should raise because function annotation says str but returns int
        with pytest.raises(ToolValidationError):
            wrapped.invoke({"x": 5})
