"""Unified workspace configuration for all agent file operations.

This module provides a single abstraction for configuring how agents
interact with filesystems, whether in-memory, sandboxed, or full access.
Works with both deep agents (via backends) and regular agents (via proj_mgmt).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from deepagents.backends import BackendProtocol

WorkspaceMode = Literal["virtual", "sandboxed", "full"]


class Workspace:
    """Unified workspace configuration for all agent file operations.

    The Workspace class provides a single way to configure how agents
    interact with the filesystem. It bridges:
    - deepagents backends (for deep=True agents)
    - proj_mgmt tools (for regular agents with file tools)

    Args:
        root: The workspace root directory. Defaults to current directory.
        mode: How the agent can access files:
            - "virtual": In-memory only. Files don't persist. Safe for untrusted code.
            - "sandboxed": Real filesystem, but confined to root. Recommended for most use.
            - "full": Full filesystem access. Use only for trusted automation.

    Example - Local development (sandboxed to project):
        ```python
        agent = Agent(
            deep=True,
            workspace=Workspace(".", mode="sandboxed"),
        )
        # Agent can read/write files in current directory only
        ```

    Example - Cloud/untrusted (virtual filesystem):
        ```python
        agent = Agent(
            deep=True,
            workspace=Workspace(mode="virtual"),
        )
        # Files exist only in memory, safe for untrusted prompts
        ```

    Example - Trusted automation (full access):
        ```python
        agent = Agent(
            deep=True,
            workspace=Workspace("/", mode="full"),
        )
        # Agent has full filesystem access - use with caution!
        ```
    """

    def __init__(
        self,
        root: str | Path = ".",
        mode: WorkspaceMode = "sandboxed",
    ):
        self.root = Path(root).resolve()
        self.mode = mode

    def get_deepagent_backend(self) -> BackendProtocol:
        """Get the deepagents backend for this workspace.

        Returns:
            Configured backend for use with create_deep_agent().

        Security Note:
            The sandboxing is enforced by deepagents' FilesystemBackend.
            For "sandboxed" mode, paths are validated to stay within root_dir.
            For "full" mode, the agent has full filesystem access from root.

            [!] If using "full" mode, the agent can access ANY file the process
            can access. Only use with trusted inputs in controlled environments.
        """
        from deepagents.backends import FilesystemBackend

        if self.mode == "virtual":
            # In-memory filesystem (no persistence, safe for untrusted code)
            # Note: virtual_mode=True WITHOUT root_dir creates in-memory only
            return FilesystemBackend(virtual_mode=True)
        elif self.mode == "sandboxed":
            # Real filesystem, sandboxed to root_dir with virtual path mode
            # virtual_mode=True ensures paths like "/src/file.py" are treated as
            # relative to root_dir, not as absolute system paths
            return FilesystemBackend(root_dir=str(self.root), virtual_mode=True)
        else:  # full
            # Full filesystem access from root
            # virtual_mode=False allows absolute paths outside root_dir
            # [!] SECURITY: No sandboxing - agent can access any file
            return FilesystemBackend(root_dir=str(self.root), virtual_mode=False)

    def configure_proj_mgmt(self) -> None:
        """Configure proj_mgmt tools to use this workspace.

        Sets the workspace root for file_read, file_write, files_list, etc.
        """
        from ai_infra.llm.tools.custom.proj_mgmt.utils import _set_workspace_root

        if self.mode == "virtual":
            # Virtual mode: proj_mgmt tools should error
            # (use deep agent's in-memory filesystem instead)
            _set_workspace_root(None)
        else:
            _set_workspace_root(self.root)

    def __repr__(self) -> str:
        return f"Workspace(root={self.root!r}, mode={self.mode!r})"


def workspace(
    root: str | Path = ".",
    mode: WorkspaceMode = "sandboxed",
) -> Workspace:
    """Create a workspace configuration.

    Convenience function for creating Workspace instances.

    Args:
        root: The workspace root directory
        mode: "virtual", "sandboxed", or "full"

    Returns:
        Configured Workspace instance

    Example:
        ```python
        from ai_infra.llm import Agent, workspace

        agent = Agent(
            deep=True,
            workspace=workspace(".", mode="sandboxed"),
        )
        ```
    """
    return Workspace(root=root, mode=mode)
