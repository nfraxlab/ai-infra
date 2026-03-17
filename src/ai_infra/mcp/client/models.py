# ai_infra/mcp/models.py
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class McpServerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transport: Literal["stdio", "streamable_http", "sse"]

    # http-like
    url: str | None = None
    headers: dict[str, str] | None = None

    # stdio
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None

    # opts
    stateless_http: bool | None = None
    json_response: bool | None = None
    oauth: dict[str, Any] | None = None

    # Called before every connection to obtain a fresh bearer token.
    # Use this for OAuth-managed credentials (e.g. via svc_infra.connect) so
    # tokens are refreshed automatically rather than going stale.
    # Accepts both sync and async callables: () -> str | Awaitable[str]
    token_fn: Callable[[], Awaitable[str] | str] | None = None

    @model_validator(mode="after")
    def _validate(self):
        if self.transport in ("streamable_http", "sse") and not self.url:
            raise ValueError(f"{self.transport} requires 'url'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("Remote stdio requires 'command'")
        return self
