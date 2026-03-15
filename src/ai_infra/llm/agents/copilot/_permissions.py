"""Permission control types for CopilotAgent.

Defines PermissionMode, PermissionDeniedError, the PermissionHandler type
alias, and the frozen sets of read-only / destructive built-in tools used
by the automatic policy hooks.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

_READ_ONLY_TOOLS = frozenset(
    {"view", "read_file", "grep", "glob", "ls", "find", "git_diff", "git_log", "git_status"}
)
_DESTRUCTIVE_TOOLS = frozenset(
    {
        "bash",
        "shell",
        "write_file",
        "edit_file",
        "delete_file",
        "git_commit",
        "git_push",
        "git_reset",
    }
)

# Callable signature: (tool_name: str, arguments: dict) -> bool
# Return True to allow, False to deny.
PermissionHandler = Callable[[str, dict[str, Any]], bool]


class PermissionMode(StrEnum):
    """Controls which tools CopilotAgent is allowed to invoke.

    Attributes:
        AUTO_APPROVE: All tools approved automatically. Use for trusted
            local automation (e.g. Pulse running on the user's own machine).
        READ_ONLY: Only read-safe tools allowed (view, read_file, grep, glob,
            git_status, git_diff). No writes, no shell execution.
        INTERACTIVE: Destructive tools (bash, write_file, delete_file, git ops)
            raise ``PermissionDeniedError`` unless a custom ``permission_handler``
            callable is also provided to approve them at runtime.
        DENY_ALL: No tools allowed. Useful for dry-run / replay mode.
    """

    AUTO_APPROVE = "auto-approve"
    READ_ONLY = "read-only"
    INTERACTIVE = "interactive"
    DENY_ALL = "deny-all"


class PermissionDeniedError(RuntimeError):
    """Raised when CopilotAgent blocks a tool call due to permission mode."""

    def __init__(self, tool_name: str, reason: str = "") -> None:
        self.tool_name = tool_name
        msg = f"Tool '{tool_name}' was blocked by CopilotAgent permission policy"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


__all__ = [
    "PermissionMode",
    "PermissionDeniedError",
    "PermissionHandler",
    "_READ_ONLY_TOOLS",
    "_DESTRUCTIVE_TOOLS",
]
