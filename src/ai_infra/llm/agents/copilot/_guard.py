"""Optional dependency guard for github-copilot-sdk.

Exports HAS_COPILOT flag and placeholder stubs used by the rest of the
copilot package when the SDK is not installed.
"""

from __future__ import annotations

from typing import Any


def _missing_copilot(*args: Any, **kwargs: Any) -> None:
    raise ImportError(
        "CopilotAgent requires 'github-copilot-sdk'. Install with: pip install 'ai-infra[copilot]'"
    )


# Placeholder classes always defined so static analysers see a consistent type.
# When the SDK is available the try-block below overwrites them with the real types.


class CopilotClient:
    """Placeholder when github-copilot-sdk is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _missing_copilot()


class SubprocessConfig:
    """Placeholder when github-copilot-sdk is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _missing_copilot()


class Tool:
    """Placeholder when github-copilot-sdk is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _missing_copilot()


try:
    from copilot import CopilotClient as CopilotClient  # type: ignore[import-untyped,no-redef]
    from copilot import (
        SubprocessConfig as SubprocessConfig,  # type: ignore[import-untyped,no-redef]
    )
    from copilot import Tool as Tool  # type: ignore[import-untyped,no-redef]

    HAS_COPILOT = True
except ImportError:
    HAS_COPILOT = False


__all__ = [
    "HAS_COPILOT",
    "CopilotClient",
    "SubprocessConfig",
    "Tool",
    "_missing_copilot",
]
