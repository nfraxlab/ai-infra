"""Agent submodules for split functionality.

This package contains extracted modules from the main Agent class:
- deep: DeepAgents integration for autonomous multi-step tasks
- copilot: GitHub Copilot SDK integration for computer automation tasks
- callbacks: Callback wrapping utilities for tool instrumentation
"""

from ai_infra.llm.agents.callbacks import wrap_tool_with_callbacks
from ai_infra.llm.agents.copilot import (
    HAS_COPILOT,
    CopilotAgent,
    CopilotEvent,
    CopilotResult,
    PermissionMode,
    copilot_tool,
)
from ai_infra.llm.agents.deep import (
    HAS_DEEPAGENTS,
    AgentMiddleware,
    CompiledSubAgent,
    FilesystemMiddleware,
    SubAgent,
    SubAgentMiddleware,
    build_deep_agent,
)

__all__ = [
    # Copilot agents
    "HAS_COPILOT",
    "CopilotAgent",
    "CopilotEvent",
    "CopilotResult",
    "PermissionMode",
    "copilot_tool",
    # Deep agents
    "HAS_DEEPAGENTS",
    "SubAgent",
    "CompiledSubAgent",
    "SubAgentMiddleware",
    "FilesystemMiddleware",
    "AgentMiddleware",
    "build_deep_agent",
    # Callbacks
    "wrap_tool_with_callbacks",
]
