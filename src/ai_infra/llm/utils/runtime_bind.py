from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime

from ai_infra.llm.tools.tool_controls import ToolCallControls, normalize_tool_controls

from .model_registry import ModelRegistry
from .settings import ModelSettings

_logger = logging.getLogger(__name__)


def tool_used(state: Any) -> bool:
    """Heuristic: did the agent already emit a tool call (or a tool message)?"""
    msgs = state.get("messages", []) if isinstance(state, dict) else []
    for m in reversed(msgs):
        if getattr(m, "tool_calls", None):
            return True
        if getattr(m, "type", None) == "tool":  # ToolMessage
            return True
        if isinstance(m, dict) and (m.get("tool_calls") or m.get("type") == "tool"):
            return True
    return False


def bind_model_with_tools(
    state: Any,
    runtime: Runtime[ModelSettings],
    registry: ModelRegistry,
    *,
    global_tools: list[Any] | None = None,
) -> Any:
    """Select (or lazily init) the model and bind tools according to controls.

    This mirrors the prior LLM._select_model method but is factored out for reuse.
    """
    ctx = runtime.context
    key_model_kwargs = ctx.extra.get("model_kwargs", {}) if ctx.extra else {}
    model = registry.get_or_create(ctx.provider, ctx.model_name, **key_model_kwargs)

    tools = ctx.tools if ctx.tools is not None else (global_tools or [])
    extra = ctx.extra or {}

    tool_choice, parallel_tool_calls, force_once = normalize_tool_controls(
        ctx.provider, extra.get("tool_controls")
    )

    # Gemini special-case: do not send explicit tool_choice if no tools are bound.
    if ctx.provider == "google_genai" and not tools:
        tool_choice = None

    if force_once and tool_used(state):
        tool_choice = None

    # Gemini doesn't support parallel_tool_calls parameter
    bind_kwargs: dict[str, Any] = {"tool_choice": tool_choice}
    if ctx.provider != "google_genai":
        bind_kwargs["parallel_tool_calls"] = parallel_tool_calls

    return model.bind_tools(tools, **bind_kwargs)


def make_agent_with_context(
    registry: ModelRegistry,
    *,
    provider: str,
    model_name: str | None,
    tools: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tool_controls: ToolCallControls | dict[str, Any] | None = None,
    require_explicit_tools: bool = False,
    global_tools: list[Any] | None = None,
    hitl_tool_wrapper=None,
    logger: logging.Logger | None = None,
    # Session/checkpoint config
    checkpointer: Any | None = None,
    store: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    # Safety limits
    recursion_limit: int = 50,
) -> tuple[Any, ModelSettings]:
    """Construct an agent (LangGraph ReAct) and its runtime context.

    Handles:
      - model warm-up via registry
      - optional tool control merging
      - implicit global tools policy
      - HITL tool wrapping
      - agent graph creation with deferred model binding
      - session persistence via checkpointer
      - pause/resume via interrupt_before/after

    Args:
        registry: Model registry for lazy model creation
        provider: LLM provider name
        model_name: Model name (or None for provider default)
        tools: Tools to bind (overrides global_tools if provided)
        extra: Additional context (tool_controls, model_kwargs, etc.)
        model_kwargs: Kwargs passed to model creation
        tool_controls: Tool calling controls (tool_choice, parallel_tool_calls)
        require_explicit_tools: If True, error when using implicit global tools
        global_tools: Default tools when none specified
        hitl_tool_wrapper: Function to wrap tools for HITL
        logger: Logger for debug messages
        checkpointer: LangGraph checkpointer for session persistence
        store: LangGraph store for cross-session memory
        interrupt_before: Tool names to pause before executing
        interrupt_after: Tool names to pause after executing
        recursion_limit: Maximum number of agent iterations (default: 50).
            Prevents infinite loops when agent keeps calling tools without
            making progress. A recursion limit error will be raised if exceeded.
            This is a critical safety measure to prevent runaway token costs.

    Returns:
        Tuple of (compiled agent, ModelSettings context)
    """
    model_kwargs = model_kwargs or {}
    effective_model = registry.resolve_model_name(provider, model_name)
    registry.get_or_create(provider, effective_model, **model_kwargs)
    if tool_controls is not None:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(tool_controls):
            tool_controls = asdict(tool_controls)
        extra = {**(extra or {}), "tool_controls": tool_controls}

    # Effective tools resolution
    effective_tools = global_tools or []
    if tools is not None:
        effective_tools = tools
    else:
        if (global_tools and len(global_tools) > 0) and require_explicit_tools:
            raise ValueError(
                "Implicit global tools use forbidden (require_tools_explicit=True). "
                "Pass tools=[] to run without tools or tools=[...] to specify explicitly."
            )
        if global_tools and len(global_tools) > 0 and logger:
            logger.info(
                "[LLM] Using global self.tools (%d). Pass tools=[] to suppress or set require_tools_explicit(True) to forbid implicit use.",
                len(global_tools),
            )

    effective_tools = [nt for nt in (_normalize_tool(t) for t in effective_tools) if nt is not None]

    if hitl_tool_wrapper is not None:
        wrapped_tools: list[Any] = []
        for t in effective_tools:
            try:
                w = hitl_tool_wrapper(t)
                wrapped_tools.append(w if w is not None else t)  # fallback to original tool
            except Exception:
                wrapped_tools.append(t)  # defensive fallback
        effective_tools = wrapped_tools

    if not effective_tools and logger:
        logger.warning("No tools bound; agent will not call tools.")

    # Store recursion_limit in extra for runtime config injection
    # IMPORTANT: recursion_limit is passed to invoke()/astream() config, NOT to create_react_agent()
    merged_extra = {
        "model_kwargs": model_kwargs or {},
        "recursion_limit": recursion_limit,
        **(extra or {}),
    }

    context = ModelSettings(
        provider=provider,
        model_name=effective_model,
        tools=effective_tools,
        extra=merged_extra,
    )

    def _selector(state, rt: Runtime[ModelSettings]):
        return bind_model_with_tools(state, rt, registry, global_tools=context.tools)

    # Build agent with optional session/interrupt config
    agent = create_react_agent(
        model=_selector,
        tools=effective_tools,
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
    return agent, context


def _normalize_tool(t):
    if t is None:
        return None
    if isinstance(t, BaseTool):
        return t
    if callable(t):
        return lc_tool(t)
    if isinstance(t, dict):  # leave dict (ignored by ToolNode) but log
        _logger.warning(
            "Dict-shaped tool provided and will be ignored by ToolNode: keys=%s",
            list(t.keys()),
        )
        return None

    # Handle MCP tools (mcp.server.fastmcp.tools.base.Tool)
    # Check by duck-typing to avoid hard dependency on mcp package
    if _is_mcp_tool(t):
        return _mcp_tool_to_langchain(t)

    _logger.warning("Unsupported tool type ignored: %r", type(t))
    return None


def _is_mcp_tool(t: Any) -> bool:
    """Check if object is an MCP Tool by duck-typing.

    MCP tools have: fn, name, description, parameters, run() method.
    We check by attributes rather than isinstance to avoid import dependency.
    """
    return (
        hasattr(t, "fn")
        and hasattr(t, "name")
        and hasattr(t, "description")
        and hasattr(t, "parameters")
        and hasattr(t, "run")
        and callable(getattr(t, "run", None))
    )


def _mcp_tool_to_langchain(mcp_tool: Any) -> BaseTool:
    """Convert an MCP Tool to a LangChain StructuredTool.

    MCP tools from FastMCP have:
    - fn: The underlying callable
    - name: Tool name
    - description: Tool description
    - parameters: JSON schema for input
    - is_async: Whether the tool is async
    - run(arguments, context=None): Execute the tool

    We create a StructuredTool that wraps the MCP tool's run() method.
    """
    name = getattr(mcp_tool, "name", "unknown_tool")
    description = getattr(mcp_tool, "description", "") or ""
    parameters = getattr(mcp_tool, "parameters", {}) or {}
    is_async = getattr(mcp_tool, "is_async", False)

    # Create wrapper functions that call mcp_tool.run()
    # MCP tool.run() takes a dict of arguments
    def sync_wrapper(**kwargs: Any) -> Any:
        """Sync wrapper for MCP tool."""
        result = mcp_tool.run(kwargs)
        # Handle async result if run() returns a coroutine
        if asyncio.iscoroutine(result):
            # Run in event loop - this handles the case where
            # an async MCP tool is called from sync context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(result, loop)
                return future.result(timeout=60)
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                return asyncio.run(result)
        return result

    async def async_wrapper(**kwargs: Any) -> Any:
        """Async wrapper for MCP tool."""
        result = mcp_tool.run(kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # Build the StructuredTool
    # Use args_schema as dict (JSON schema) - LangChain accepts this
    tool = StructuredTool(
        name=name,
        description=description,
        args_schema=parameters,  # JSON schema dict
        func=sync_wrapper if not is_async else None,
        coroutine=async_wrapper if is_async else async_wrapper,  # Always provide async
    )

    return tool
