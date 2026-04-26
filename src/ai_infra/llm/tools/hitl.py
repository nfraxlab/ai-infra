"""Centralized HITL + tool policy utilities.

This module consolidates logic that previously lived ad‑hoc in LLM / runtime_bind:
  - HITLConfig: stores callbacks and provides a .set API (legacy)
  - ApprovalConfig: new approval-based HITL configuration
  - maybe_await: safe sync resolver for (possibly) async callbacks
  - apply_output_gate: applies model output moderation / modification gate
  - wrap_tool_for_hitl: wraps a tool with pre‑execution HITL policy
  - wrap_tool_for_approval: wraps a tool with approval workflow
  - ToolPolicy: configuration holder for tool selection policy
  - compute_effective_tools: merges per‑call tools with global tools under policy
  - ToolExecutionConfig: configuration for tool execution (errors, timeouts, validation)
  - wrap_tool_with_execution_config: wraps tools with error/timeout/validation handling

None of these functions mutate global state; they are pure / side‑effect free
(except logging) and can be composed by higher‑level orchestration code.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool

# Import tool errors from central location
from ai_infra.errors import ToolExecutionError, ToolTimeoutError, ToolValidationError

from .approval import (
    ApprovalHandler,
    ApprovalRequest,
    ApprovalResponse,
    AsyncApprovalHandler,
    AsyncOutputReviewer,
    OutputReviewer,
    OutputReviewRequest,
    OutputReviewResponse,
    console_approval_handler,
)

_UNTRUNCATED_TOOL_RESULT_PREFIXES: dict[str, tuple[str, ...]] = {
    # Pulse visualizations are rendered out-of-band by the client. Truncating the
    # prefixed HTML payload corrupts the inline bootstrap script and makes the
    # visualization render as title-only.
    "create_visualization": ("<!--PULSE_VIZ-->",),
}

if TYPE_CHECKING:
    from .events import ApprovalEvents

__all__ = [
    # Legacy HITL
    "HITLConfig",
    "maybe_await",
    "apply_output_gate",
    "apply_output_gate_async",
    "wrap_tool_for_hitl",
    # New approval-based HITL
    "ApprovalConfig",
    "wrap_tool_for_approval",
    # Tool policy
    "ToolPolicy",
    "compute_effective_tools",
    # Tool execution config
    "ToolExecutionConfig",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolValidationError",
    "wrap_tool_with_execution_config",
]

logger = logging.getLogger(__name__)


# ---------- HITL Configuration ----------
class HITLConfig:
    """Container for Human-In-The-Loop (HITL) callbacks.

    on_model_output(ai_msg) -> decision dict or None
        decision: {action: pass|modify|block, replacement: str}

    on_tool_call(name: str, args: dict) -> decision dict or None
        decision: {action: pass|modify|block, args: {...}, replacement: any}
    """

    def __init__(
        self,
        *,
        on_model_output: Callable[..., Any] | None = None,
        on_tool_call: Callable[..., Any] | None = None,
        on_model_output_async: Callable[..., Any] | None = None,
        on_tool_call_async: Callable[..., Any] | None = None,
    ):
        self.on_model_output = on_model_output
        self.on_tool_call = on_tool_call
        self.on_model_output_async = on_model_output_async
        self.on_tool_call_async = on_tool_call_async

    def set(
        self,
        *,
        on_model_output: Callable[..., Any] | None = None,
        on_tool_call: Callable[..., Any] | None = None,
        on_model_output_async: Callable[..., Any] | None = None,
        on_tool_call_async: Callable[..., Any] | None = None,
    ):
        if on_model_output is not None:
            self.on_model_output = on_model_output
        if on_tool_call is not None:
            self.on_tool_call = on_tool_call
        if on_model_output_async is not None:
            self.on_model_output_async = on_model_output_async
        if on_tool_call_async is not None:
            self.on_tool_call_async = on_tool_call_async
        return self

    async def call_model_output(self, ai_msg: Any):
        if self.on_model_output_async:
            return await self.on_model_output_async(ai_msg)
        if self.on_model_output:
            return await asyncio.to_thread(self.on_model_output, ai_msg)
        return None

    async def call_tool(self, name: str, args: dict[str, Any]):
        if self.on_tool_call_async:
            return await self.on_tool_call_async(name, args)
        if self.on_tool_call:
            return await asyncio.to_thread(self.on_tool_call, name, args)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "on_model_output": self.on_model_output,
            "on_tool_call": self.on_tool_call,
            "on_model_output_async": self.on_model_output_async,
            "on_tool_call_async": self.on_tool_call_async,
        }


class _HITLWrappedTool(BaseTool):
    """Async-first wrapper around a BaseTool enforcing async execution."""

    def __init__(self, base: BaseTool, hitl: HITLConfig):
        super().__init__(
            name=getattr(base, "name", "tool"),
            description=getattr(base, "description", "") or "",
        )
        self._base = base
        self._hitl = hitl
        # preserve args schema if present
        if hasattr(base, "args_schema") and base.args_schema is not None:
            self.args_schema = base.args_schema

    # Disallow sync path to avoid StructuredTool sync errors
    def _run(self, *args, **kwargs):
        raise NotImplementedError("HITL-wrapped tools are async-only. Use ainvoke/_arun.")

    async def _arun(self, *args, **kwargs):
        args_dict = dict(kwargs) if kwargs else {}
        try:
            decision = await self._hitl.call_tool(self.name, args_dict)
        except Exception:
            decision = {"action": "pass"}

        action = (decision or {}).get("action", "pass")
        if action == "block":
            return (decision or {}).get("replacement", "[blocked by reviewer]")
        if action == "modify":
            args_dict = (decision or {}).get("args", args_dict)

        # prefer async on the base tool
        if hasattr(self._base, "ainvoke"):
            return await self._base.ainvoke(args_dict)
        # fallback: run sync tool in a thread
        return await asyncio.to_thread(self._base.invoke, args_dict)


# ---------- New Approval-based HITL ----------
@dataclass
class ApprovalConfig:
    """Configuration for approval-based HITL.

    This is the recommended way to configure HITL for new code.
    It uses structured ApprovalRequest/ApprovalResponse models.

    Attributes:
        approval_handler: Sync handler for tool approval requests
        approval_handler_async: Async handler for tool approval requests
        output_reviewer: Sync handler for output review
        output_reviewer_async: Async handler for output review
        require_approval: Tools that require approval:
            - True: All tools need approval
            - List[str]: Only specified tools need approval
            - Callable[[str, dict], bool]: Function that decides based on tool name and args
        auto_approve: If True, auto-approve all requests (for testing)

    Example (Console):
        ```python
        config = ApprovalConfig(require_approval=True)
        # Uses console_approval_handler by default
        ```

    Example (Web App):
        ```python
        async def my_handler(req: ApprovalRequest) -> ApprovalResponse:
            # Send to frontend, wait for response
            ...

        config = ApprovalConfig(approval_handler_async=my_handler)
        ```

    Example (Selective Approval):
        ```python
        config = ApprovalConfig(
            require_approval=["dangerous_tool", "delete_file"],
        )
        ```

    Example (Conditional Approval):
        ```python
        def needs_approval(tool_name: str, args: dict) -> bool:
            # Only need approval for large amounts
            if tool_name == "transfer_money":
                return args.get("amount", 0) > 1000
            return False

        config = ApprovalConfig(require_approval=needs_approval)
        ```

    Example (With Event Hooks):
        ```python
        from ai_infra.llm.tools.events import ApprovalEvents

        def on_request(req: ApprovalRequest):
            logger.info(f"Approval requested for {req.tool_name}")

        config = ApprovalConfig(
            require_approval=True,
            events=ApprovalEvents(
                on_requested=on_request,
            ),
        )
        ```
    """

    approval_handler: ApprovalHandler | None = None
    approval_handler_async: AsyncApprovalHandler | None = None
    output_reviewer: OutputReviewer | None = None
    output_reviewer_async: AsyncOutputReviewer | None = None
    require_approval: bool | list[str] | Callable[[str, dict[str, Any]], bool] = False
    auto_approve: bool = False
    events: ApprovalEvents | None = None  # Event hooks for observability

    def __post_init__(self):
        # If require_approval is True but no handler, use console
        if self.require_approval and not self.approval_handler and not self.approval_handler_async:
            if not self.auto_approve:
                self.approval_handler = console_approval_handler

    def needs_approval(self, tool_name: str, args: dict[str, Any] | None = None) -> bool:
        """Check if a tool needs approval.

        Args:
            tool_name: Name of the tool being called
            args: Arguments being passed to the tool (used for conditional approval)

        Returns:
            True if approval is required, False otherwise
        """
        if self.auto_approve:
            return False
        if callable(self.require_approval):
            # Dynamic approval based on tool name and args
            return self.require_approval(tool_name, args or {})
        if isinstance(self.require_approval, bool):
            return self.require_approval
        if isinstance(self.require_approval, list):
            return tool_name in self.require_approval
        return False

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Request approval for a tool call."""
        if self.auto_approve:
            return ApprovalResponse.approve(reason="Auto-approved", approver="auto")

        if self.approval_handler_async:
            result = self.approval_handler_async(request)
            if inspect.isawaitable(result):
                return await cast("Awaitable[ApprovalResponse]", result)
            return cast("ApprovalResponse", result)
        elif self.approval_handler:
            return await asyncio.to_thread(self.approval_handler, request)
        else:
            # No handler, auto-approve
            return ApprovalResponse.approve(reason="No handler configured", approver="auto")

    async def review_output(self, request: OutputReviewRequest) -> OutputReviewResponse:
        """Review model output."""
        if self.output_reviewer_async:
            result = self.output_reviewer_async(request)
            if inspect.isawaitable(result):
                return await cast("Awaitable[OutputReviewResponse]", result)
            return cast("OutputReviewResponse", result)
        elif self.output_reviewer:
            return await asyncio.to_thread(self.output_reviewer, request)
        else:
            return OutputReviewResponse.allow()


class _ApprovalWrappedTool(BaseTool):
    """Wraps a tool with approval workflow using ApprovalRequest/Response models."""

    def __init__(self, base: BaseTool, config: ApprovalConfig):
        super().__init__(
            name=getattr(base, "name", "tool"),
            description=getattr(base, "description", "") or "",
        )
        self._base = base
        self._config = config
        # Preserve args schema if present
        if hasattr(base, "args_schema") and base.args_schema is not None:
            self.args_schema = base.args_schema

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Approval-wrapped tools are async-only. Use ainvoke/_arun.")

    async def _arun(self, *args, **kwargs):
        from .events import ApprovalEvent

        args_dict = dict(kwargs) if kwargs else {}
        events = self._config.events

        # Check if this tool needs approval (pass args for conditional approval)
        if not self._config.needs_approval(self.name, args_dict):
            # No approval needed, execute directly
            if hasattr(self._base, "ainvoke"):
                return await self._base.ainvoke(args_dict)
            return await asyncio.to_thread(self._base.invoke, args_dict)

        # Create approval request
        request = ApprovalRequest(
            tool_name=self.name,
            args=args_dict,
        )

        # Emit approval requested event
        if events:
            event = ApprovalEvent.requested(
                tool_name=self.name,
                args=args_dict,
            )
            await events.emit_async(event)

        # Request approval
        response = await self._config.request_approval(request)

        # Emit approval response event
        if events:
            if response.approved:
                if response.modified_args:
                    event = ApprovalEvent.modified(
                        tool_name=self.name,
                        args=args_dict,
                        approver=response.approver,
                        reason=response.reason,
                        modified_args=response.modified_args,
                    )
                else:
                    event = ApprovalEvent.granted(
                        tool_name=self.name,
                        args=args_dict,
                        approver=response.approver,
                        reason=response.reason,
                    )
            else:
                event = ApprovalEvent.denied(
                    tool_name=self.name,
                    args=args_dict,
                    approver=response.approver,
                    reason=response.reason,
                )
            await events.emit_async(event)

        if not response.approved:
            reason = response.reason or "Rejected by reviewer"
            return f"[Tool call rejected: {reason}]"

        # Use modified args if provided
        if response.modified_args is not None:
            args_dict = response.modified_args

        # Execute the tool
        if hasattr(self._base, "ainvoke"):
            return await self._base.ainvoke(args_dict)
        return await asyncio.to_thread(self._base.invoke, args_dict)


def wrap_tool_for_approval(
    tool_obj: Any,
    config: ApprovalConfig | None,
) -> Any:
    """Wrap a tool with approval workflow.

    Args:
        tool_obj: The tool to wrap (BaseTool, function, or callable)
        config: Approval configuration. If None, returns tool unchanged.

    Returns:
        Wrapped tool with approval workflow, or original if config is None
    """
    if not config:
        return tool_obj

    # Normalize to BaseTool
    if isinstance(tool_obj, BaseTool):
        base = tool_obj
    elif callable(tool_obj):
        base = lc_tool(tool_obj)
    else:
        return tool_obj

    return _ApprovalWrappedTool(base, config)


# ---------- Async helper ----------
def maybe_await(result: Any) -> Any:
    """Resolve an awaitable in a sync context safely.

    Behavior mirrors LLM._maybe_await:
      - If result is not awaitable, return as-is.
      - If an event loop is running, log warning and return None (cannot block).
      - If coroutine / awaitable and no loop, run it to completion.
    """
    if not inspect.isawaitable(result):
        return result
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        logger.warning(
            "maybe_await: async callback ignored (event loop active in sync pathway). Use async APIs for async callbacks."
        )
        return None
    if not asyncio.iscoroutine(result):

        async def _wrap(awaitable):
            return await awaitable

        result = _wrap(result)
    return asyncio.run(result)


# ---------- Output gating ----------
def apply_output_gate(ai_msg: Any, hitl: HITLConfig | dict[str, Any] | None) -> Any:
    """Apply HITL on_model_output gate to a model/agent final output.

    ai_msg can be:
      - a LangChain AIMessage (has .content)
      - a dict state with messages: [...]
      - any other object (left unchanged unless replaced)
    """
    if not hitl:
        return ai_msg
    on_out = hitl.on_model_output if isinstance(hitl, HITLConfig) else hitl.get("on_model_output")
    if not on_out:
        return ai_msg
    try:
        decision = maybe_await(on_out(ai_msg))
        if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
            replacement = decision.get("replacement", "")
            if (
                isinstance(ai_msg, dict)
                and isinstance(ai_msg.get("messages"), list)
                and ai_msg["messages"]
            ):
                last_msg = ai_msg["messages"][-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    last_msg["content"] = replacement
                elif hasattr(last_msg, "content"):
                    last_msg.content = replacement
                else:
                    ai_msg["messages"][-1] = {"role": "ai", "content": replacement}
            elif hasattr(ai_msg, "content"):
                ai_msg.content = replacement
            else:
                ai_msg = (
                    {"role": "ai", "content": replacement}
                    if not isinstance(ai_msg, dict)
                    else ai_msg
                )
    except Exception:  # pragma: no cover - defensive
        pass
    return ai_msg


async def apply_output_gate_async(ai_msg: Any, hitl: HITLConfig | None) -> Any:
    if not hitl:
        return ai_msg
    try:
        decision = await hitl.call_model_output(ai_msg)
        if isinstance(decision, dict) and decision.get("action") in ("modify", "block"):
            replacement = decision.get("replacement", "")
            # mirror your existing mutation logic:
            if (
                isinstance(ai_msg, dict)
                and isinstance(ai_msg.get("messages"), list)
                and ai_msg["messages"]
            ):
                last_msg = ai_msg["messages"][-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    last_msg["content"] = replacement
                elif hasattr(last_msg, "content"):
                    last_msg.content = replacement
                else:
                    ai_msg["messages"][-1] = {"role": "ai", "content": replacement}
            elif hasattr(ai_msg, "content"):
                ai_msg.content = replacement
            else:
                ai_msg = (
                    {"role": "ai", "content": replacement}
                    if not isinstance(ai_msg, dict)
                    else ai_msg
                )
    except Exception:
        pass
    return ai_msg


# ---------- Tool wrapping ----------
def wrap_tool_for_hitl(tool_obj: Any, hitl: HITLConfig | None):
    """Return an async-first HITL-wrapped tool when a tool_call gate is present."""
    if not hitl or not (hitl.on_tool_call or hitl.on_tool_call_async):
        return tool_obj

    # Normalize to BaseTool
    if isinstance(tool_obj, BaseTool):
        base = tool_obj
    elif callable(tool_obj):
        base = lc_tool(
            tool_obj
        )  # wraps function into a BaseTool (supports ainvoke when func is async)
    else:
        return tool_obj

    return _HITLWrappedTool(base, hitl)


# ---------- Tool policy ----------
class ToolPolicy:
    """Holds configuration flags controlling tool resolution.

    Attributes:
        require_explicit (bool): If True, implicit use of global tools is forbidden.
    """

    def __init__(self, *, require_explicit: bool = False):
        self.require_explicit = require_explicit

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ToolPolicy(require_explicit={self.require_explicit})"


def compute_effective_tools(
    call_tools: Sequence[Any] | None,
    global_tools: Sequence[Any] | None,
    policy: ToolPolicy,
    *,
    logger_: logging.Logger | None = None,
) -> list[Any]:
    """Compute effective tools for a call given per-call list & global list under policy.

    Logic mirrors the inline section previously in make_agent_with_context:
      - If call_tools is provided (even empty list), use it directly.
      - Else if global tools exist and policy requires explicit: raise.
      - Else use global tools (may be empty) and optionally log.
    """
    global_tools = list(global_tools or [])
    if call_tools is not None:
        return list(call_tools)  # explicit override (including empty list to suppress)

    if global_tools and policy.require_explicit:
        raise ValueError(
            "Implicit global tools use forbidden (require_explicit=True). "
            "Pass tools=[] to run without tools or tools=[...] to specify explicitly."
        )
    if global_tools and logger_:
        logger_.info(
            "[ToolPolicy] Using implicit global tools (%d). Pass tools=[] to suppress or enable require_explicit to forbid.",
            len(global_tools),
        )
    return global_tools


# ---------- Tool Execution Configuration ----------
@dataclass
class ToolExecutionConfig:
    """Configuration for tool execution behavior.

    Attributes:
        on_error: How to handle tool execution errors.
            - "return_error": Return error message to agent (default, allows recovery)
            - "retry": Retry the tool call up to max_retries times
            - "abort": Re-raise the exception and stop execution
        max_retries: Maximum retry attempts when on_error="retry" (default 1)
        timeout: Timeout in seconds per tool call (None = no timeout)
        validate_results: Validate tool results match expected return type annotations
        on_timeout: How to handle timeouts.
            - "return_error": Return timeout message to agent (default)
            - "abort": Re-raise TimeoutError
        max_result_chars: Maximum characters in tool result (default 60000, ~15k tokens).
            Results exceeding this limit are truncated with a note.
            This prevents massive tool outputs from blowing the context window.
            Set to 0 or None to disable truncation.

    Example:
        ```python
        # Allow agent to recover from errors
        config = ToolExecutionConfig(on_error="return_error")

        # Retry on failure with 30s timeout
        config = ToolExecutionConfig(
            on_error="retry",
            max_retries=2,
            timeout=30,
        )

        # Strict mode: fail fast
        config = ToolExecutionConfig(
            on_error="abort",
            validate_results=True,
        )

        # Limit tool result size to prevent context overflow
        config = ToolExecutionConfig(
            max_result_chars=30000,  # ~7.5k tokens
        )
        ```
    """

    on_error: Literal["return_error", "retry", "abort"] = "return_error"
    max_retries: int = 1
    timeout: float | None = None
    validate_results: bool = False
    on_timeout: Literal["return_error", "abort"] = "return_error"
    max_result_chars: int | None = 60000  # ~15k tokens, safe for most context windows

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.max_result_chars is not None and self.max_result_chars < 0:
            raise ValueError("max_result_chars must be >= 0 or None")


class _ExecutionConfigWrappedTool(BaseTool):
    """Wraps a BaseTool with error handling, timeout, and validation."""

    def __init__(
        self,
        base: BaseTool,
        config: ToolExecutionConfig,
        expected_return_type: type | None = None,
    ):
        super().__init__(
            name=getattr(base, "name", "tool"),
            description=getattr(base, "description", "") or "",
        )
        self._base = base
        self._config = config
        self._expected_return_type = expected_return_type
        # Preserve args schema if present
        if hasattr(base, "args_schema") and base.args_schema is not None:
            self.args_schema = base.args_schema

    def _validate_result(self, result: Any) -> None:
        """Validate the result matches expected type."""
        if not self._config.validate_results or self._expected_return_type is None:
            return
        # Skip validation for None returns when expected type allows it
        if result is None:
            return
        # Check type (basic isinstance check)
        if not isinstance(result, self._expected_return_type):
            raise ToolValidationError(
                f"Tool '{self.name}' returned {type(result).__name__}, expected {self._expected_return_type.__name__}",
                tool_name=self.name,
            )

    def _truncate_result(self, result: Any) -> Any:
        """Truncate result if it exceeds max_result_chars.

        This prevents massive tool outputs from blowing the context window
        and causing excessive token costs.
        """
        max_chars = self._config.max_result_chars
        if max_chars is None or max_chars == 0:
            return result  # Truncation disabled

        if isinstance(result, str):
            allowed_prefixes = _UNTRUNCATED_TOOL_RESULT_PREFIXES.get(self.name, ())
            if allowed_prefixes and result.startswith(allowed_prefixes):
                return result

        # Handle string results
        if isinstance(result, str):
            if len(result) > max_chars:
                return result[:max_chars] + f"\n\n[TRUNCATED: Result exceeded {max_chars} chars]"
            return result

        # Handle dict/list by converting to string for size check
        if isinstance(result, (dict, list)):
            try:
                import json

                result_str = json.dumps(result)
                if len(result_str) > max_chars:
                    # Return truncated JSON string with note
                    return (
                        result_str[:max_chars]
                        + f"\n\n[TRUNCATED: Result exceeded {max_chars} chars]"
                    )
            except (TypeError, ValueError):
                pass  # Can't serialize, return as-is
            return result

        # For other types, convert to string and check
        result_str = str(result)
        if len(result_str) > max_chars:
            return result_str[:max_chars] + f"\n\n[TRUNCATED: Result exceeded {max_chars} chars]"
        return result

    def _format_error(self, error: Exception) -> str:
        """Format error for returning to agent."""
        return f"[Tool Error: {self.name}] {type(error).__name__}: {error!s}"

    def _run(self, *args, **kwargs) -> Any:
        """Sync execution with error handling and retry."""
        config = self._config
        last_error: Exception | None = None
        attempts = config.max_retries + 1 if config.on_error == "retry" else 1

        for attempt in range(attempts):
            try:
                # Execute with optional timeout
                if config.timeout is not None:
                    import queue
                    import threading

                    result_queue: queue.Queue[Any] = queue.Queue()
                    error_queue: queue.Queue[Exception] = queue.Queue()

                    def run_with_timeout():
                        try:
                            res = self._base.invoke(kwargs or {})
                            result_queue.put(res)
                        except Exception as e:
                            error_queue.put(e)

                    thread = threading.Thread(target=run_with_timeout)
                    thread.start()
                    thread.join(timeout=config.timeout)

                    if thread.is_alive():
                        # Timeout occurred
                        if config.on_timeout == "abort":
                            raise ToolTimeoutError(
                                f"Tool '{self.name}' timed out after {config.timeout}s",
                                tool_name=self.name,
                                timeout=config.timeout,
                            )
                        return f"[Tool Timeout: {self.name}] Execution timed out after {config.timeout}s"

                    if not error_queue.empty():
                        raise error_queue.get()
                    result = result_queue.get()
                else:
                    result = self._base.invoke(kwargs or {})

                # Validate result
                self._validate_result(result)
                # Truncate result to prevent context overflow
                return self._truncate_result(result)

            except (ToolTimeoutError, ToolValidationError):
                raise  # Don't retry these
            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    continue  # Retry
                # Out of retries
                if config.on_error == "abort":
                    raise ToolExecutionError(
                        f"Tool '{self.name}' failed: {e}",
                        tool_name=self.name,
                        original_error=e,
                    ) from e
                return self._format_error(e)

        # Shouldn't reach here, but just in case
        if last_error:
            return self._format_error(last_error)
        return "[Tool Error] Unknown error occurred"

    async def _arun(self, *args, **kwargs) -> Any:
        """Async execution with error handling, timeout, and retry."""
        config = self._config
        last_error: Exception | None = None
        attempts = config.max_retries + 1 if config.on_error == "retry" else 1

        for attempt in range(attempts):
            try:
                # Execute with optional timeout
                if config.timeout is not None:
                    try:
                        if hasattr(self._base, "ainvoke"):
                            result = await asyncio.wait_for(
                                self._base.ainvoke(kwargs or {}),
                                timeout=config.timeout,
                            )
                        else:
                            result = await asyncio.wait_for(
                                asyncio.to_thread(self._base.invoke, kwargs or {}),
                                timeout=config.timeout,
                            )
                    except TimeoutError:
                        if config.on_timeout == "abort":
                            raise ToolTimeoutError(
                                f"Tool '{self.name}' timed out after {config.timeout}s",
                                tool_name=self.name,
                                timeout=config.timeout,
                            )
                        return f"[Tool Timeout: {self.name}] Execution timed out after {config.timeout}s"
                else:
                    if hasattr(self._base, "ainvoke"):
                        result = await self._base.ainvoke(kwargs or {})
                    else:
                        result = await asyncio.to_thread(self._base.invoke, kwargs or {})

                # Validate result
                self._validate_result(result)
                # Truncate result to prevent context overflow
                return self._truncate_result(result)

            except (ToolTimeoutError, ToolValidationError):
                raise  # Don't retry these
            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    continue  # Retry
                # Out of retries
                if config.on_error == "abort":
                    raise ToolExecutionError(
                        f"Tool '{self.name}' failed: {e}",
                        tool_name=self.name,
                        original_error=e,
                    ) from e
                return self._format_error(e)

        # Shouldn't reach here
        if last_error:
            return self._format_error(last_error)
        return "[Tool Error] Unknown error occurred"


def wrap_tool_with_execution_config(
    tool_obj: Any,
    config: ToolExecutionConfig | None,
    *,
    expected_return_type: type | None = None,
) -> Any:
    """Wrap a tool with execution configuration (error handling, timeout, validation).

    Args:
        tool_obj: The tool to wrap (BaseTool, function, or callable)
        config: Execution configuration. If None, returns tool unchanged.
        expected_return_type: Expected return type for validation (extracted from annotations if possible)

    Returns:
        Wrapped tool with execution config, or original if config is None

    Example:
        ```python
        from ai_infra.llm.tools import ToolExecutionConfig, wrap_tool_with_execution_config

        config = ToolExecutionConfig(
            on_error="return_error",
            timeout=30,
            validate_results=True,
        )

        wrapped_tool = wrap_tool_with_execution_config(my_tool, config)
        ```
    """
    if not config:
        return tool_obj

    # Normalize to BaseTool
    if isinstance(tool_obj, BaseTool):
        base = tool_obj
    elif callable(tool_obj):
        base = lc_tool(tool_obj)
        # Try to extract return type from function annotations
        if expected_return_type is None and hasattr(tool_obj, "__annotations__"):
            expected_return_type = tool_obj.__annotations__.get("return")
    else:
        return tool_obj

    return _ExecutionConfigWrappedTool(base, config, expected_return_type)
