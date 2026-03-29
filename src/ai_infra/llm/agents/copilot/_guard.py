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


class ModelInfo:
    """Placeholder when github-copilot-sdk is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _missing_copilot()


class ModelCapabilities:
    """Placeholder when github-copilot-sdk is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _missing_copilot()


class ModelSupports:
    """Placeholder when github-copilot-sdk is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _missing_copilot()


class ModelLimits:
    """Placeholder when github-copilot-sdk is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _missing_copilot()


try:
    from copilot import (  # type: ignore[import-untyped,no-redef]
        CopilotClient as CopilotClient,
    )
    from copilot import (  # type: ignore[import-untyped,no-redef]
        ModelCapabilities as ModelCapabilities,
    )
    from copilot import (  # type: ignore[import-untyped,no-redef]
        ModelInfo as ModelInfo,
    )
    from copilot import (  # type: ignore[import-untyped,no-redef]
        ModelSupports as ModelSupports,
    )
    from copilot import (  # type: ignore[import-untyped,no-redef]
        SubprocessConfig as SubprocessConfig,
    )
    from copilot import (  # type: ignore[import-untyped,no-redef]
        Tool as Tool,
    )

    # ModelLimits may not be in the public API; fall back to the placeholder
    try:
        from copilot import ModelLimits as ModelLimits  # type: ignore[import-untyped,no-redef]
    except ImportError:
        pass

    HAS_COPILOT = True
except ImportError:
    HAS_COPILOT = False


__all__ = [
    "HAS_COPILOT",
    "CopilotClient",
    "ModelCapabilities",
    "ModelInfo",
    "ModelLimits",
    "ModelSupports",
    "SubprocessConfig",
    "Tool",
    "_missing_copilot",
]
