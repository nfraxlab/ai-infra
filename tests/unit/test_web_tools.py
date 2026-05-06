"""Tests for hosted web fetch and search tools."""

from __future__ import annotations

import sys
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestFetchURLTool:
    def test_tool_has_correct_name(self) -> None:
        from ai_infra.llm.tools.custom.web import fetch_url

        assert fetch_url.name == "fetch_url"

    def test_tool_exported_from_main_module(self) -> None:
        from ai_infra.llm.tools import fetch_url

        assert fetch_url is not None

    @pytest.mark.asyncio
    async def test_fetch_url_uses_public_fetch_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_infra.llm.tools.custom.web import fetch_url

        fetch_public_url = AsyncMock(
            return_value=SimpleNamespace(
                content="Page body",
                content_type="text/plain",
                metadata={"final_url": "https://example.com/final"},
            )
        )
        svc_loaders = import_module("svc_infra.loaders")
        monkeypatch.setattr(svc_loaders, "fetch_public_url", fetch_public_url, raising=False)

        result = await fetch_url.ainvoke({"url": "https://example.com/page", "max_chars": 1000})

        fetch_public_url.assert_awaited_once()
        assert "Requested URL: https://example.com/page" in result
        assert "Final URL: https://example.com/final" in result
        assert "Page body" in result

    @pytest.mark.asyncio
    async def test_fetch_url_falls_back_to_url_loader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_infra.llm.tools.custom.web import fetch_url

        monkeypatch.delattr("svc_infra.loaders.fetch_public_url", raising=False)

        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(
            return_value=[
                SimpleNamespace(
                    content="Page body",
                    content_type="text/plain",
                    metadata={"final_url": "https://example.com/final"},
                )
            ]
        )

        loader_factory = MagicMock(return_value=mock_loader)
        monkeypatch.setattr("svc_infra.loaders.URLLoader", loader_factory)

        result = await fetch_url.ainvoke({"url": "https://example.com/page", "max_chars": 1000})

        loader_factory.assert_called_once()
        _, kwargs = loader_factory.call_args
        assert kwargs["extract_text"] is True
        assert kwargs["on_error"] == "raise"
        assert "Requested URL: https://example.com/page" in result
        assert "Final URL: https://example.com/final" in result
        assert "Page body" in result


class TestSearchWebTool:
    def test_tool_has_correct_name(self) -> None:
        from ai_infra.llm.tools.custom.web import search_web

        assert search_web.name == "search_web"

    def test_tool_exported_from_main_module(self) -> None:
        from ai_infra.llm.tools import search_web

        assert search_web is not None

    @pytest.mark.asyncio
    async def test_search_web_uses_google_news_rss_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_infra.llm.tools.custom.web import search_web

        monkeypatch.delenv("AI_INFRA_WEB_SEARCH_PROVIDER", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        fetch_public_url = AsyncMock(
            return_value=SimpleNamespace(
                content=(
                    "<rss><channel>"
                    "<item><title>Result One</title><link>https://example.com/1</link>"
                    "<description><![CDATA[First <b>snippet</b>]]></description>"
                    "<source url='https://publisher.example'>Example News</source>"
                    "<pubDate>Mon, 05 May 2026 12:00:00 GMT</pubDate></item>"
                    "<item><title>Result Two</title><link>https://example.com/2</link>"
                    "<description>Second snippet</description></item>"
                    "</channel></rss>"
                )
            )
        )
        svc_loaders = import_module("svc_infra.loaders")
        monkeypatch.setattr(svc_loaders, "fetch_public_url", fetch_public_url, raising=False)

        result = await search_web.ainvoke({"query": "latest AI news", "limit": 3})

        assert "1. Result One" in result
        assert "https://example.com/1" in result
        assert "First snippet" in result
        assert "Source: Example News" in result
        assert "2. Result Two" in result

    @pytest.mark.asyncio
    async def test_search_web_uses_tavily_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ai_infra.llm.tools.custom.web import search_web

        class FakeTavilyClient:
            def __init__(self, *, api_key: str) -> None:
                self.api_key = api_key

            def search(self, **_: object) -> dict[str, object]:
                return {
                    "results": [
                        {
                            "title": "Result One",
                            "url": "https://example.com/1",
                            "content": "First snippet",
                        },
                        {
                            "title": "Result Two",
                            "url": "https://example.com/2",
                            "content": "Second snippet",
                        },
                    ]
                }

        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.delenv("AI_INFRA_WEB_SEARCH_PROVIDER", raising=False)
        monkeypatch.setitem(sys.modules, "tavily", SimpleNamespace(TavilyClient=FakeTavilyClient))

        result = await search_web.ainvoke({"query": "latest AI news", "limit": 2})

        assert "1. Result One" in result
        assert "https://example.com/1" in result
        assert "First snippet" in result
        assert "2. Result Two" in result

    @pytest.mark.asyncio
    async def test_search_web_handles_missing_tavily_package(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_infra.llm.tools.custom.web import search_web

        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setenv("AI_INFRA_WEB_SEARCH_PROVIDER", "tavily")
        monkeypatch.delitem(sys.modules, "tavily", raising=False)

        real_import_module = __import__("importlib").import_module

        def fake_import_module(name: str, package: str | None = None):
            if name == "tavily":
                raise ImportError("missing tavily")
            return real_import_module(name, package)

        monkeypatch.setattr("importlib.import_module", fake_import_module)

        result = await search_web.ainvoke({"query": "latest AI news", "limit": 2})

        assert "Install ai-infra[web-search]" in result

    @pytest.mark.asyncio
    async def test_search_web_handles_invalid_google_news_rss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_infra.llm.tools.custom.web import search_web

        monkeypatch.setenv("AI_INFRA_WEB_SEARCH_PROVIDER", "google-news-rss")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        fetch_public_url = AsyncMock(return_value=SimpleNamespace(content="not-xml"))
        svc_loaders = import_module("svc_infra.loaders")
        monkeypatch.setattr(svc_loaders, "fetch_public_url", fetch_public_url, raising=False)

        result = await search_web.ainvoke({"query": "latest AI news", "limit": 2})

        assert "invalid RSS response" in result
