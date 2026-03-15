"""copilot_tool decorator and _CopilotTool marker type."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ai_infra.llm.agents.copilot._guard import HAS_COPILOT, Tool, _missing_copilot


@dataclass
class _CopilotTool:
    """Internal marker wrapping a Python callable for Copilot use."""

    fn: Any
    name: str
    description: str
    skip_permission: bool = False
    overrides_built_in_tool: bool = False

    def to_sdk_tool(self) -> Any:
        """Convert to a native Copilot SDK ``Tool`` object."""
        if not HAS_COPILOT:
            _missing_copilot()

        fn = self.fn

        async def _handler(invocation: dict[str, Any]) -> Any:
            kwargs = invocation.get("arguments", {})
            if inspect.iscoroutinefunction(fn):
                return await fn(**kwargs)
            return fn(**kwargs)

        # Build JSON schema from type hints
        type_map: dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        hints = {k: v for k, v in (fn.__annotations__ or {}).items() if k != "return"}
        properties = {
            param: {"type": type_map.get(hint, "string")} for param, hint in hints.items()
        }

        sig = inspect.signature(fn)
        required = [
            k
            for k, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and k != "self"
        ]

        kwargs: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "handler": _handler,
        }
        if self.skip_permission:
            kwargs["skip_permission"] = True
        if self.overrides_built_in_tool:
            kwargs["overrides_built_in_tool"] = True

        return Tool(**kwargs)


def copilot_tool(
    func: Any = None,
    *,
    description: str | None = None,
    name: str | None = None,
    skip_permission: bool = False,
    overrides_built_in_tool: bool = False,
) -> Any:
    """Register a Python callable as a tool available to CopilotAgent.

    Wraps a plain Python function (sync or async) so the Copilot CLI runtime
    can invoke it during task execution. Type hints generate the JSON schema
    automatically — no manual schema definition required.

    Use this the same way you'd add a function tool to ``Agent(tools=[...])``.
    The function's docstring becomes the tool description unless overridden.

    Example — basic::

        @copilot_tool
        async def get_issue(id: str) -> str:
            "Fetch an issue from Linear."
            return await linear.get(id)

        agent = CopilotAgent(tools=[get_issue])

    Example — with explicit description::

        @copilot_tool(description="Query the Pulse backend for a user record")
        async def get_user(user_id: str) -> str:
            return await pulse_api.get_user(user_id)

    Example — read-only tool that skips permission prompts::

        @copilot_tool(description="Read project config", skip_permission=True)
        def read_config(path: str) -> str:
            return Path(path).read_text()

    Example — override a built-in Copilot tool::

        @copilot_tool(
            name="edit_file",
            description="Custom file editor with project-specific validation",
            overrides_built_in_tool=True,
        )
        async def edit_file(path: str, content: str) -> str:
            ...

    Args:
        func: The callable to wrap (used when decorator is applied without args).
        description: Tool description. Defaults to the function's docstring.
        name: Tool name. Defaults to ``func.__name__``.
        skip_permission: If ``True``, this tool runs without a permission prompt.
        overrides_built_in_tool: If ``True``, this tool replaces a built-in
            Copilot CLI tool with the same name (e.g. ``edit_file``,
            ``read_file``). The SDK requires this flag to be set explicitly
            to prevent accidental overrides.

    Returns:
        A ``_CopilotTool`` instance that ``CopilotAgent`` converts to a native
        SDK ``Tool`` at session creation time.
    """

    def _wrap(fn: Any) -> _CopilotTool:
        return _CopilotTool(
            fn=fn,
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or fn.__name__),
            skip_permission=skip_permission,
            overrides_built_in_tool=overrides_built_in_tool,
        )

    if func is not None:
        # @copilot_tool — no parentheses
        return _wrap(func)
    # @copilot_tool(...) — with arguments
    return _wrap


__all__ = [
    "_CopilotTool",
    "copilot_tool",
]
