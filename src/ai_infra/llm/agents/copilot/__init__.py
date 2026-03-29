"""GitHub Copilot SDK integration for computer automation tasks.

This package provides CopilotAgent — a runtime-backed agent that delegates
execution to the GitHub Copilot CLI. Unlike Agent (which calls LLMs via
ai-infra's provider abstractions), CopilotAgent drives the full Copilot CLI
runtime, giving it autonomous access to:

  - Shell / bash execution
  - File system operations (read, write, edit, glob, grep)
  - Git operations (status, diff, commit, branch management)
  - Web fetch and search
  - Multi-step planning with automatic context compaction
  - Sub-agent orchestration

CopilotAgent is not an LLM provider — it is a computer automation runtime.
The LLM is managed internally by the Copilot CLI. You bring the task; the
agent determines the tools and steps required to complete it.

Usage::

    from ai_infra import CopilotAgent

    agent = CopilotAgent(cwd="/path/to/project")
    result = await agent.run("Add type hints to all public functions in src/")
    print(result.content)

Requires:
    - ``copilot`` CLI installed in PATH. See:
      https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli
    - ``pip install 'ai-infra[copilot]'``
    - A GitHub Copilot subscription, OR BYOK keys (see ``provider`` param).

Package layout:
    _guard.py       Optional dependency detection and placeholder stubs.
    _permissions.py PermissionMode enum, PermissionDeniedError, policy sets.
    _events.py      CopilotEvent and CopilotResult dataclasses.
    _tools.py       _CopilotTool marker and copilot_tool() decorator.
    _agent.py       CopilotAgent class (lifecycle, session config, streaming).
"""

from ai_infra.llm.agents.copilot._agent import CopilotAgent
from ai_infra.llm.agents.copilot._events import CopilotEvent, CopilotResult
from ai_infra.llm.agents.copilot._guard import (
    HAS_COPILOT,
    ModelCapabilities,
    ModelInfo,
    ModelLimits,
    ModelSupports,
)
from ai_infra.llm.agents.copilot._permissions import PermissionDeniedError, PermissionMode
from ai_infra.llm.agents.copilot._scratchpad import (
    SCRATCHPAD_TOOL_NAMES,
    create_scratchpad_tools,
)
from ai_infra.llm.agents.copilot._tools import copilot_tool

__all__ = [
    "HAS_COPILOT",
    "CopilotAgent",
    "CopilotEvent",
    "CopilotResult",
    "ModelCapabilities",
    "ModelInfo",
    "ModelLimits",
    "ModelSupports",
    "PermissionMode",
    "PermissionDeniedError",
    "SCRATCHPAD_TOOL_NAMES",
    "copilot_tool",
    "create_scratchpad_tools",
]
