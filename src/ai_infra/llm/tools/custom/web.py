"""Hosted web fetch and search tools for ai-infra agents.

These tools are intended for server-side agent execution where external HTTP
access is allowed by policy. They provide a stable agent-facing interface while
delegating low-level safety and fetch behavior to svc-infra.
"""

from __future__ import annotations

import asyncio
import html
import importlib
import os
import re
from typing import Any
from urllib.parse import quote_plus

from defusedxml import ElementTree
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class FetchURLInput(BaseModel):
    """Input schema for hosted URL fetches."""

    url: str = Field(description="Public http or https URL to fetch")
    max_chars: int = Field(
        default=12_000,
        ge=500,
        le=50_000,
        description="Maximum characters of extracted page text to return",
    )


class SearchWebInput(BaseModel):
    """Input schema for hosted web search."""

    query: str = Field(description="Search query for current public web information")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of results to return")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _truncate(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "\n...[truncated]"


async def _load_public_url_content(url: str, *, extract_text: bool) -> Any | None:
    loaders_module = importlib.import_module("svc_infra.loaders")
    timeout = _float_env("AI_INFRA_WEB_FETCH_TIMEOUT_SECONDS", 10.0)
    max_bytes = _int_env("AI_INFRA_WEB_FETCH_MAX_BYTES", 1_048_576)

    fetch_public_url = getattr(loaders_module, "fetch_public_url", None)
    if callable(fetch_public_url):
        return await fetch_public_url(
            url,
            extract_text=extract_text,
            timeout=timeout,
            max_bytes=max_bytes,
        )

    loader_factory = loaders_module.URLLoader
    loader = loader_factory(
        url,
        extract_text=extract_text,
        timeout=timeout,
        on_error="raise",
    )
    contents = await loader.load()
    if not contents:
        return None
    return contents[0]


async def _fetch_url_impl(url: str, max_chars: int = 12_000) -> str:
    try:
        content = await _load_public_url_content(url, extract_text=True)
    except Exception as exc:
        return f"Unable to fetch URL: {exc}"

    if content is None:
        return "Unable to fetch URL: no content was returned."

    excerpt = _truncate(content.content, max_chars=max_chars)
    final_url = str(content.metadata.get("final_url", url))
    content_type = content.content_type or "unknown"
    return (
        f"Requested URL: {url}\nFinal URL: {final_url}\nContent-Type: {content_type}\n\n{excerpt}"
    )


def _detect_search_provider() -> str | None:
    configured = os.getenv("AI_INFRA_WEB_SEARCH_PROVIDER", "").strip().lower()
    if configured:
        if configured in {"google-news-rss", "google_news_rss", "google_news", "news"}:
            return "google_news_rss"
        return configured
    if os.getenv("TAVILY_API_KEY", "").strip():
        return "tavily"
    return "google_news_rss"


def _format_search_results(results: list[dict[str, Any]], limit: int) -> str:
    if not results:
        return "No web search results found."

    lines: list[str] = []
    for index, item in enumerate(results[:limit], start=1):
        title = str(item.get("title") or item.get("url") or f"Result {index}").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "..."
        lines.append(f"{index}. {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(lines)


def _clean_search_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text or "")
    unescaped = html.unescape(without_tags)
    return " ".join(unescaped.split()).strip()


def _parse_google_news_rss_items(xml_text: str) -> list[dict[str, str]]:
    root = ElementTree.fromstring(xml_text)
    items: list[dict[str, str]] = []

    for item in root.findall("./channel/item"):
        title = _clean_search_text(item.findtext("title", default=""))
        url = (item.findtext("link", default="") or "").strip()
        description = _clean_search_text(item.findtext("description", default=""))
        source_name = _clean_search_text(item.findtext("source", default=""))
        pub_date = _clean_search_text(item.findtext("pubDate", default=""))

        snippet_parts = [
            part
            for part in (description, f"Source: {source_name}" if source_name else "", pub_date)
            if part
        ]
        items.append(
            {
                "title": title or url or "News result",
                "url": url,
                "snippet": " | ".join(snippet_parts),
            }
        )

    return items


async def _search_with_google_news_rss(query: str, limit: int) -> str:
    feed_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"

    try:
        content = await _load_public_url_content(feed_url, extract_text=False)
    except Exception as exc:
        return f"Web search failed: {exc}"

    if content is None:
        return "Web search failed: no RSS content was returned."

    try:
        results = _parse_google_news_rss_items(content.content)
    except ElementTree.ParseError as exc:
        return f"Web search failed: invalid RSS response ({exc})."

    return _format_search_results(results, limit=limit)


async def _search_with_tavily(query: str, limit: int) -> str:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "Web search is not configured in this environment."

    try:
        tavily_module = importlib.import_module("tavily")
    except ImportError:
        return (
            "Web search provider 'tavily' requires the optional dependency "
            "tavily-python. Install ai-infra[web-search]."
        )

    tavily_client_factory = getattr(tavily_module, "TavilyClient", None)
    if tavily_client_factory is None:
        return "Web search failed: tavily provider is missing TavilyClient."

    client = tavily_client_factory(api_key=api_key)
    try:
        response = await asyncio.to_thread(
            client.search,
            query=query,
            max_results=limit,
            include_answer=False,
            include_images=False,
            include_raw_content=False,
            search_depth="basic",
        )
    except Exception as exc:
        return f"Web search failed: {exc}"

    results = response.get("results", []) if isinstance(response, dict) else []
    return _format_search_results(results, limit=limit)


async def _search_web_impl(query: str, limit: int = 5) -> str:
    provider = _detect_search_provider()
    if provider == "tavily":
        return await _search_with_tavily(query, limit)
    if provider == "google_news_rss":
        return await _search_with_google_news_rss(query, limit)
    return f"Web search provider '{provider}' is not supported in this build."


fetch_url = StructuredTool.from_function(
    coroutine=_fetch_url_impl,
    name="fetch_url",
    description=(
        "Fetch and extract text from a public web page. Use this when the user provides "
        "a URL and asks what the page says, asks for a summary, comparison, or answer "
        "grounded in that page."
    ),
    args_schema=FetchURLInput,
)

search_web = StructuredTool.from_function(
    coroutine=_search_web_impl,
    name="search_web",
    description=(
        "Search the public web for current external information. Use this when the user "
        "needs recent or external facts that are not available in the conversation or "
        "workspace context."
    ),
    args_schema=SearchWebInput,
)


__all__ = ["fetch_url", "search_web"]
