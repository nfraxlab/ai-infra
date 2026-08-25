"""DeepAgents integration for autonomous multi-step task execution.

This module provides DeepAgents types, placeholders for when the package
is not installed, and helper functions for building deep agents.

DeepAgents mode enables autonomous task execution with built-in tools for:
- File operations (ls, read, write, edit, glob, grep, execute)
- Todo management
- Subagent orchestration
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig

    from ai_infra.llm.session import SessionConfig
    from ai_infra.llm.workspace import Workspace

logger = logging.getLogger(__name__)

# =============================================================================
# DeepAgents Types (re-exported for convenience)
# =============================================================================

try:
    from deepagents import (  # type: ignore[import-untyped]
        CompiledSubAgent,
        FilesystemMiddleware,
        SubAgent,
        SubAgentMiddleware,
    )
    from langchain.agents.middleware.types import AgentMiddleware

    HAS_DEEPAGENTS = True
except ImportError:
    HAS_DEEPAGENTS = False

    # Define placeholders when deepagents is not installed
    def _missing_deepagents(*args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "DeepAgents requires 'deepagents' package. Install with: pip install deepagents"
        )

    class SubAgent(dict):  # type: ignore[no-redef]
        """Placeholder for SubAgent when deepagents is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_deepagents()

    class CompiledSubAgent:  # type: ignore[no-redef]
        """Placeholder for CompiledSubAgent when deepagents is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_deepagents()

    class SubAgentMiddleware:  # type: ignore[no-redef]
        """Placeholder for SubAgentMiddleware when deepagents is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_deepagents()

    class FilesystemMiddleware:  # type: ignore[no-redef]
        """Placeholder for FilesystemMiddleware when deepagents is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_deepagents()

    class AgentMiddleware:  # type: ignore[no-redef]
        """Placeholder for AgentMiddleware when deepagents is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_deepagents()


__all__ = [
    "HAS_DEEPAGENTS",
    "AgentMiddleware",
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "build_deep_agent",
]


def build_deep_agent(
    model: Any,
    *,
    workspace: Workspace | None = None,
    session_config: SessionConfig | None = None,
    tools: list[Any] | None = None,
    system: str | None = None,
    middleware: Sequence[Any] | None = None,
    subagents: list[Any] | None = None,
    response_format: Any | None = None,
    context_schema: type[Any] | None = None,
) -> Any:
    """Build a DeepAgents agent for autonomous multi-step task execution.

    This function creates a deep agent using LangChain's deepagents package,
    which provides built-in file tools (ls, read, write, edit, glob, grep, execute),
    todo management, and subagent orchestration.

    Args:
        model: LangChain chat model instance
        workspace: Workspace configuration for file operations
        session_config: Session configuration for checkpointing and interrupts
        tools: Additional tools (added to built-in deep agent tools)
        system: System prompt / additional instructions
        middleware: Additional middleware to apply to the deep agent
        subagents: List of subagents for delegation
        response_format: Structured output format for agent responses
        context_schema: Schema for the deep agent context

    Returns:
        Compiled DeepAgent graph

    Raises:
        ImportError: If deepagents package is not installed
    """
    if not HAS_DEEPAGENTS:
        raise ImportError(
            "DeepAgents mode requires 'deepagents' package. Install with: pip install deepagents"
        )

    from deepagents import create_deep_agent

    # Extract session config
    checkpointer = None
    store = None
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
    if session_config:
        checkpointer = session_config.storage.get_checkpointer()
        store = session_config.storage.get_store()
        if session_config.pause_after:
            raise ValueError("pause_after is not supported for deep agents; use pause_before")
        if session_config.pause_before:
            interrupt_on = {}
            for tool_name in session_config.pause_before:
                interrupt_on[tool_name] = True

    # Get backend from workspace (if configured)
    backend = None
    if workspace:
        backend = workspace.get_deepagent_backend()

    return create_deep_agent(
        model=model,
        backend=backend,
        tools=tools if tools else None,
        system_prompt=system,
        middleware=tuple(middleware) if middleware else (),
        subagents=subagents,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        interrupt_on=interrupt_on,
    )
