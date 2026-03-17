"""Tests for MCP client connection management.

Tests cover:
- Connection establishment (stdio, http, sse transports)
- Connection closing/cleanup
- Auto-reconnect behavior
- Connection health checks
- Async context manager usage

Phase 1.1.4 of production readiness test plan.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.mcp.client import (
    MCPClient,
    McpServerConfig,
)

# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestMCPClientInit:
    """Tests for MCPClient initialization."""

    def test_init_with_stdio_config(self):
        """Client accepts stdio transport config."""
        client = MCPClient(
            [
                {"transport": "stdio", "command": "npx", "args": ["-y", "server"]},
            ]
        )
        assert len(client._configs) == 1
        assert client._configs[0].transport == "stdio"

    def test_init_with_http_config(self):
        """Client accepts streamable_http transport config."""
        client = MCPClient(
            [
                {"transport": "streamable_http", "url": "http://localhost:3000/mcp"},
            ]
        )
        assert len(client._configs) == 1
        assert client._configs[0].transport == "streamable_http"

    def test_init_with_sse_config(self):
        """Client accepts sse transport config."""
        client = MCPClient(
            [
                {"transport": "sse", "url": "http://localhost:3000/sse"},
            ]
        )
        assert len(client._configs) == 1
        assert client._configs[0].transport == "sse"

    def test_init_with_pydantic_config(self):
        """Client accepts Pydantic McpServerConfig objects."""
        config = McpServerConfig(transport="stdio", command="npx", args=[])
        client = MCPClient([config])
        assert client._configs[0] is config

    def test_init_with_multiple_configs(self):
        """Client accepts multiple server configs."""
        client = MCPClient(
            [
                {"transport": "stdio", "command": "server1"},
                {"transport": "sse", "url": "http://localhost:3001"},
            ]
        )
        assert len(client._configs) == 2

    def test_init_invalid_config_type_raises(self):
        """Non-list config raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            MCPClient({"command": "npx"})  # type: ignore


# =============================================================================
# Connection Options Tests
# =============================================================================


class TestConnectionOptions:
    """Tests for connection management options."""

    def test_auto_reconnect_stored(self):
        """auto_reconnect option is stored."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            auto_reconnect=True,
        )
        assert client._auto_reconnect is True

    def test_reconnect_delay_stored(self):
        """reconnect_delay option is stored."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            reconnect_delay=5.0,
        )
        assert client._reconnect_delay == 5.0

    def test_max_reconnect_attempts_stored(self):
        """max_reconnect_attempts option is stored."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            max_reconnect_attempts=10,
        )
        assert client._max_reconnect_attempts == 10

    def test_pool_size_stored(self):
        """pool_size option is stored."""
        client = MCPClient(
            [{"transport": "stdio", "command": "npx"}],
            pool_size=20,
        )
        assert client._pool_size == 20


# =============================================================================
# Config Validation Tests
# =============================================================================


class TestConfigValidation:
    """Tests for config validation."""

    def test_stdio_requires_command(self):
        """stdio transport requires command."""
        with pytest.raises(ValueError, match="requires 'command'"):
            MCPClient([{"transport": "stdio"}])

    def test_http_requires_url(self):
        """streamable_http transport requires url."""
        with pytest.raises(ValueError, match="requires 'url'"):
            MCPClient([{"transport": "streamable_http"}])

    def test_sse_requires_url(self):
        """sse transport requires url."""
        with pytest.raises(ValueError, match="requires 'url'"):
            MCPClient([{"transport": "sse"}])


# =============================================================================
# Async Context Manager Tests
# =============================================================================


class TestAsyncContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_discover(self):
        """Entering context manager calls discover."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        with patch.object(client, "discover", new_callable=AsyncMock) as mock:
            mock.return_value = {}
            async with client as c:
                assert c is client
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_calls_close(self):
        """Exiting context manager calls close."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        with patch.object(client, "discover", new_callable=AsyncMock):
            with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
                async with client:
                    pass
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        """close() resets client state."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._discovered = True
        client._by_name = {"test": MagicMock()}
        client._health_status = {"test": "healthy"}

        await client.close()

        assert client._discovered is False
        assert client._by_name == {}
        assert client._health_status == {}


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self):
        """health_check returns status dict."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        mock_session = MagicMock()
        mock_session.mcp_server_info = {"name": "test-server"}

        @asynccontextmanager
        async def mock_ctx():
            yield mock_session

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.return_value = mock_ctx()
            result = await client.health_check()

        assert isinstance(result, dict)
        assert "test-server" in result
        assert result["test-server"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_marks_unhealthy(self):
        """health_check marks failed server as unhealthy."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])

        with patch.object(client, "_open_session_from_config") as mock_open:
            mock_open.side_effect = Exception("Connection failed")
            result = await client.health_check()

        assert isinstance(result, dict)
        assert any(status == "unhealthy" for status in result.values())


# =============================================================================
# Server Names Tests
# =============================================================================


class TestServerNames:
    """Tests for server_names method."""

    def test_server_names_empty_before_discover(self):
        """server_names returns empty before discovery."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        assert client.server_names() == []

    def test_server_names_after_discover(self):
        """server_names returns names after discovery."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        client._by_name = {"server1": MagicMock(), "server2": MagicMock()}
        assert client.server_names() == ["server1", "server2"]


# =============================================================================
# Last Errors Tests
# =============================================================================


class TestLastErrors:
    """Tests for last_errors tracking."""

    def test_last_errors_empty_initially(self):
        """last_errors returns empty list initially."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        assert client.last_errors() == []


# =============================================================================
# Config Identity Tests
# =============================================================================


class TestConfigIdentity:
    """Tests for config identity generation."""

    def test_cfg_identity_stdio(self):
        """Config identity for stdio includes command and args."""
        client = MCPClient([{"transport": "stdio", "command": "npx", "args": ["-y", "server"]}])
        identity = client._cfg_identity(client._configs[0])

        assert "stdio" in identity
        assert "npx" in identity
        assert "-y server" in identity

    def test_cfg_identity_http(self):
        """Config identity for HTTP includes URL."""
        client = MCPClient([{"transport": "streamable_http", "url": "http://localhost:3000"}])
        identity = client._cfg_identity(client._configs[0])

        assert "streamable_http" in identity
        assert "localhost:3000" in identity


# =============================================================================
# Unique Name Generation Tests
# =============================================================================


class TestResolveHeaders:
    """Tests for _resolve_headers dynamic token injection."""

    @pytest.mark.asyncio
    async def test_no_token_fn_returns_static_headers(self):
        """When token_fn is not set, static headers are returned unchanged."""
        cfg = McpServerConfig(
            transport="streamable_http",
            url="http://localhost:3000",
            headers={"X-Custom": "value"},
        )
        result = await MCPClient._resolve_headers(cfg)
        assert result == {"X-Custom": "value"}

    @pytest.mark.asyncio
    async def test_no_token_fn_no_headers_returns_none(self):
        """When token_fn and headers are both absent, None is returned."""
        cfg = McpServerConfig(transport="streamable_http", url="http://localhost:3000")
        result = await MCPClient._resolve_headers(cfg)
        assert result is None

    @pytest.mark.asyncio
    async def test_sync_token_fn_injects_bearer(self):
        """Sync token_fn result is injected as Authorization header."""
        cfg = McpServerConfig(
            transport="streamable_http",
            url="http://localhost:3000",
            token_fn=lambda: "sync-token-abc",
        )
        result = await MCPClient._resolve_headers(cfg)
        assert result == {"Authorization": "Bearer sync-token-abc"}

    @pytest.mark.asyncio
    async def test_async_token_fn_injects_bearer(self):
        """Async token_fn result is awaited and injected as Authorization header."""

        async def get_token() -> str:
            return "async-token-xyz"

        cfg = McpServerConfig(
            transport="streamable_http",
            url="http://localhost:3000",
            token_fn=get_token,
        )
        result = await MCPClient._resolve_headers(cfg)
        assert result == {"Authorization": "Bearer async-token-xyz"}

    @pytest.mark.asyncio
    async def test_token_fn_merges_with_existing_headers(self):
        """token_fn result is merged with pre-existing headers."""

        async def get_token() -> str:
            return "fresh-token"

        cfg = McpServerConfig(
            transport="streamable_http",
            url="http://localhost:3000",
            headers={"X-Tenant": "acme"},
            token_fn=get_token,
        )
        result = await MCPClient._resolve_headers(cfg)
        assert result == {"X-Tenant": "acme", "Authorization": "Bearer fresh-token"}

    @pytest.mark.asyncio
    async def test_token_fn_overrides_static_authorization(self):
        """token_fn takes precedence over a static Authorization header."""

        async def get_token() -> str:
            return "new-token"

        cfg = McpServerConfig(
            transport="streamable_http",
            url="http://localhost:3000",
            headers={"Authorization": "Bearer old-token"},
            token_fn=get_token,
        )
        result = await MCPClient._resolve_headers(cfg)
        assert result == {"Authorization": "Bearer new-token"}


class TestUniqueName:
    """Tests for unique name generation."""

    def test_uniq_name_no_collision(self):
        """No collision returns original name."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        used: set[str] = set()
        name = client._uniq_name("server", used)
        assert name == "server"

    def test_uniq_name_with_collision(self):
        """Collision appends counter."""
        client = MCPClient([{"transport": "stdio", "command": "npx"}])
        used = {"server", "server#2"}
        name = client._uniq_name("server", used)
        assert name == "server#3"
