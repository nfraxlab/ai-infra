"""
Dynamic model and provider discovery for ai-infra.

This module provides functions to discover available providers and models
at runtime by querying the provider APIs directly.

NOTE: This module now uses the centralized provider registry at
`ai_infra.providers`. The constants and functions here are maintained
for backwards compatibility but delegate to the registry.

Usage:
    from ai_infra.llm.providers.discovery import (
        list_providers,
        list_models,
        list_all_models,
        is_provider_configured,
    )

    # List all supported providers
    providers = list_providers()

    # List models for a specific provider
    models = list_models("openai")

    # Check if a provider has API key configured
    is_configured = is_provider_configured("openai")
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, cast

from ai_infra.providers import ProviderCapability, ProviderRegistry

log = logging.getLogger(__name__)

# =============================================================================
# DEPRECATED: These constants are maintained for backwards compatibility.
# Use ai_infra.providers.ProviderRegistry instead.
# =============================================================================


# Provider configuration - now sourced from registry
# Kept for backwards compatibility with code that imports these directly
def _get_provider_env_vars() -> dict[str, str]:
    """Get provider env vars from registry (backwards compat)."""
    result = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.CHAT):
        config = ProviderRegistry.get(name)
        if config:
            result[name] = config.env_var
    return result


def _get_provider_alt_env_vars() -> dict[str, list[str]]:
    """Get alternative env vars from registry (backwards compat)."""
    result = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.CHAT):
        config = ProviderRegistry.get(name)
        if config and config.alt_env_vars:
            result[name] = config.alt_env_vars
    return result


# Backwards compatibility - these are now computed from registry
PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
}

PROVIDER_ALT_ENV_VARS: dict[str, list[str]] = {
    "google_genai": ["GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"],
}

SUPPORTED_PROVIDERS: list[str] = ["openai", "anthropic", "google_genai", "xai"]

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "ai-infra"
CACHE_FILE = CACHE_DIR / "models.json"
CACHE_TTL_SECONDS = 3600  # 1 hour


def list_providers() -> list[str]:
    """
    List all supported provider names for CHAT capability.

    Returns:
        List of provider names: ["openai", "anthropic", "google_genai", "xai"]
    """
    return ProviderRegistry.list_for_capability(ProviderCapability.CHAT)


def list_configured_providers() -> list[str]:
    """
    List providers that have API keys configured.

    Returns:
        List of provider names that have their API key env var set.
    """
    return ProviderRegistry.list_configured_for_capability(ProviderCapability.CHAT)


def is_provider_configured(provider: str) -> bool:
    """
    Check if a provider has its API key configured.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")

    Returns:
        True if the provider's API key environment variable is set.

    Raises:
        ValueError: If provider is not supported.
    """
    config = ProviderRegistry.get(provider)
    if not config:
        supported = ProviderRegistry.list_for_capability(ProviderCapability.CHAT)
        raise ValueError(f"Unknown provider: {provider}. Supported: {', '.join(supported)}")
    return ProviderRegistry.is_configured(provider)


def get_api_key(provider: str) -> str | None:
    """
    Get the API key for a provider.

    Args:
        provider: Provider name

    Returns:
        API key string or None if not configured.
    """
    return ProviderRegistry.get_api_key(provider)


def get_default_provider() -> str | None:
    """
    Auto-detect the default provider based on configured API keys.

    Checks providers in priority order:
    1. OpenAI (most common)
    2. Anthropic
    3. Google GenAI
    4. xAI

    Returns:
        Provider name if one is configured, None otherwise.

    Example:
        >>> # With OPENAI_API_KEY set
        >>> get_default_provider()
        'openai'
    """
    # Priority order for auto-detection
    priority_order = ["openai", "anthropic", "google_genai", "xai"]
    return ProviderRegistry.get_default_for_capability(
        ProviderCapability.CHAT, priority=priority_order
    )


# -----------------------------------------------------------------------------
# Per-provider model fetchers
# -----------------------------------------------------------------------------


# =============================================================================
# Model Capability Detection
# =============================================================================
# Instead of excluding models, we categorize them by capability.
# This allows the same model list to serve different use cases:
# - chat: Text generation, conversation
# - embedding: Vector embeddings for RAG
# - audio: TTS, STT, realtime voice
# - image: Image generation (DALL-E, Imagen)
# - moderation: Content moderation
# - vision: Image understanding (many chat models also support this)
# =============================================================================


class ModelCapability(str, Enum):
    """Capabilities a model can have."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    AUDIO = "audio"
    IMAGE = "image"
    MODERATION = "moderation"
    VISION = "vision"
    REALTIME = "realtime"
    VIDEO = "video"
    CODE = "code"  # Legacy codex models
    UNKNOWN = "unknown"


# =============================================================================
# Provider-specific capability patterns
# Format: (pattern, capabilities) - if pattern in model_id, add these capabilities
# Patterns are checked in order, and capabilities accumulate (a model can have multiple)
#
# IMPORTANT: More specific patterns should come first, and exclusion patterns
# (like "codex", "instruct") are handled specially to prevent false positives.
# =============================================================================

# Patterns that indicate a model is NOT a chat model (overrides other patterns)
OPENAI_NON_CHAT_PATTERNS = [
    "codex",  # Code completion models (not chat)
    "instruct",  # Instruct models (legacy, use completions API)
    "davinci",  # Legacy completion models
    "curie",  # Legacy completion models
    "babbage",  # Legacy completion models
    "ada",  # Legacy models (except embeddings handled separately)
]

# OpenAI model capability patterns
OPENAI_CAPABILITY_PATTERNS = [
    # Legacy/Code models (check these FIRST to avoid false chat detection)
    ("codex", {ModelCapability.CODE}),
    ("davinci", {ModelCapability.CODE}),
    ("curie", {ModelCapability.CODE}),
    ("babbage", {ModelCapability.CODE}),
    ("instruct", {ModelCapability.CODE}),
    # Embedding models
    ("embedding", {ModelCapability.EMBEDDING}),
    ("text-embedding", {ModelCapability.EMBEDDING}),
    # Audio/Speech models
    ("whisper", {ModelCapability.AUDIO}),
    ("tts", {ModelCapability.AUDIO}),
    ("-audio", {ModelCapability.AUDIO}),
    ("realtime", {ModelCapability.REALTIME, ModelCapability.AUDIO}),
    # Image generation
    ("dall-e", {ModelCapability.IMAGE}),
    # Moderation
    ("moderation", {ModelCapability.MODERATION}),
    ("omni-moderation", {ModelCapability.MODERATION}),
    # GPT-4o series (multimodal: chat + vision + audio)
    (
        "gpt-4o-audio",
        {ModelCapability.CHAT, ModelCapability.VISION, ModelCapability.AUDIO},
    ),
    (
        "gpt-4o-realtime",
        {ModelCapability.CHAT, ModelCapability.REALTIME, ModelCapability.AUDIO},
    ),
    ("gpt-4o", {ModelCapability.CHAT, ModelCapability.VISION}),
    # GPT-4 variants
    ("gpt-4-turbo", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gpt-4-vision", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gpt-4", {ModelCapability.CHAT}),
    # GPT-5+ (future-proofing - assume chat + vision)
    ("gpt-5", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gpt-6", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gpt-7", {ModelCapability.CHAT, ModelCapability.VISION}),
    # GPT-3.5
    ("gpt-3.5", {ModelCapability.CHAT}),
    # ChatGPT branded models
    ("chatgpt", {ModelCapability.CHAT}),
    # o-series reasoning models (o1, o3, o4, etc.)
    ("o1", {ModelCapability.CHAT}),
    ("o3", {ModelCapability.CHAT}),
    ("o4", {ModelCapability.CHAT}),
    ("o5", {ModelCapability.CHAT}),
    ("o6", {ModelCapability.CHAT}),
]

# Anthropic model capability patterns
ANTHROPIC_CAPABILITY_PATTERNS = [
    # Claude 3+ models have vision
    ("claude-3", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("claude-4", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("claude-5", {ModelCapability.CHAT, ModelCapability.VISION}),
    # Older Claude models (chat only)
    ("claude-2", {ModelCapability.CHAT}),
    ("claude-instant", {ModelCapability.CHAT}),
    # Catch-all for any claude model
    ("claude", {ModelCapability.CHAT}),
]

# Google model capability patterns
GOOGLE_CAPABILITY_PATTERNS = [
    # Gemini models (all have vision)
    ("gemini-2", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gemini-1.5", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gemini-1", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gemini-pro-vision", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gemini-ultra", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("gemini", {ModelCapability.CHAT, ModelCapability.VISION}),
    # Embedding models
    ("embedding", {ModelCapability.EMBEDDING}),
    ("text-embedding", {ModelCapability.EMBEDDING}),
    # Image generation
    ("imagen", {ModelCapability.IMAGE}),
    # Video generation
    ("veo", {ModelCapability.VIDEO}),
    # Other
    ("aqa", {ModelCapability.CHAT}),
    # PaLM models (legacy)
    ("palm", {ModelCapability.CHAT}),
    ("bison", {ModelCapability.CHAT}),
]

# xAI model capability patterns
XAI_CAPABILITY_PATTERNS = [
    # Grok models with vision
    ("grok-2-vision", {ModelCapability.CHAT, ModelCapability.VISION}),
    ("grok-vision", {ModelCapability.CHAT, ModelCapability.VISION}),
    # Grok chat models
    ("grok-3", {ModelCapability.CHAT}),
    ("grok-2", {ModelCapability.CHAT}),
    ("grok-1", {ModelCapability.CHAT}),
    ("grok", {ModelCapability.CHAT}),
]

# Mapping of provider to capability patterns
PROVIDER_CAPABILITY_PATTERNS = {
    "openai": OPENAI_CAPABILITY_PATTERNS,
    "anthropic": ANTHROPIC_CAPABILITY_PATTERNS,
    "google_genai": GOOGLE_CAPABILITY_PATTERNS,
    "xai": XAI_CAPABILITY_PATTERNS,
}

# Mapping of provider to non-chat patterns (models that should NOT be marked as chat)
PROVIDER_NON_CHAT_PATTERNS = {
    "openai": OPENAI_NON_CHAT_PATTERNS,
    "anthropic": [],  # All Anthropic models are chat
    "google_genai": [],  # Handled by patterns
    "xai": [],  # All xAI models are chat
}


def detect_model_capabilities(model_id: str, provider: str) -> set[ModelCapability]:
    """
    Detect capabilities of a model based on its ID and provider.

    Args:
        model_id: The model identifier (e.g., "gpt-4o", "claude-3-opus")
        provider: The provider name (e.g., "openai", "anthropic")

    Returns:
        Set of ModelCapability enums the model supports.
    """
    model_lower = model_id.lower()
    capabilities: set[ModelCapability] = set()

    # Check if this model matches any non-chat patterns (prevents false chat detection)
    non_chat_patterns = PROVIDER_NON_CHAT_PATTERNS.get(provider, [])
    is_non_chat_model = any(pattern in model_lower for pattern in non_chat_patterns)

    # Get patterns for this provider
    patterns = PROVIDER_CAPABILITY_PATTERNS.get(provider, [])

    # Check each pattern
    for pattern, caps in patterns:
        if pattern in model_lower:
            # If this is a non-chat model, don't add CHAT or VISION capability
            # (legacy/code models don't support chat completions or vision)
            if is_non_chat_model:
                caps_to_add = {
                    c for c in caps if c not in (ModelCapability.CHAT, ModelCapability.VISION)
                }
                capabilities.update(caps_to_add)
            else:
                capabilities.update(caps)

    # If no capabilities detected, mark as unknown (still include it!)
    if not capabilities:
        capabilities.add(ModelCapability.UNKNOWN)

    return capabilities


def filter_models_by_capability(
    models: list[str],
    provider: str,
    capability: ModelCapability,
) -> list[str]:
    """
    Filter models to only those with a specific capability.

    Args:
        models: List of model IDs
        provider: Provider name
        capability: The capability to filter for

    Returns:
        List of model IDs that have the specified capability.
    """
    return [m for m in models if capability in detect_model_capabilities(m, provider)]


def categorize_models(
    models: list[str],
    provider: str,
) -> dict[ModelCapability, list[str]]:
    """
    Categorize models by their capabilities.

    A model can appear in multiple categories if it has multiple capabilities.

    Args:
        models: List of model IDs
        provider: Provider name

    Returns:
        Dict mapping capability to list of models with that capability.
    """
    result: dict[ModelCapability, list[str]] = {cap: [] for cap in ModelCapability}

    for model in models:
        caps = detect_model_capabilities(model, provider)
        for cap in caps:
            result[cap].append(model)

    # Sort each category
    for cap in result:
        result[cap] = sorted(result[cap])

    return result


def _list_openai_models() -> list[str]:
    """Fetch all models from OpenAI API."""
    try:
        import openai

        client = openai.OpenAI()
        models = client.models.list()
        return sorted(set(m.id for m in models.data))
    except Exception as e:
        log.warning(f"Failed to fetch OpenAI models: {e}")
        return []


def _list_openai_chat_models() -> list[str]:
    """Fetch chat-capable models from OpenAI API."""
    models = _list_openai_models()
    return filter_models_by_capability(models, "openai", ModelCapability.CHAT)


def _list_anthropic_models() -> list[str]:
    """Fetch all models from Anthropic API."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        models = client.models.list()
        # Anthropic only has chat models
        return sorted([m.id for m in models.data])
    except Exception as e:
        log.warning(f"Failed to fetch Anthropic models: {e}")
        return []


def _list_google_models() -> list[str]:
    """Fetch all models from Google GenAI API."""
    try:
        from google import genai

        api_key = get_api_key("google_genai")
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        names: list[str] = []
        for m in models:
            name = getattr(m, "name", None)
            if isinstance(name, str):
                names.append(name.replace("models/", ""))
        return sorted(set(names))
    except Exception as e:
        log.warning(f"Failed to fetch Google GenAI models: {e}")
        return []


def _list_google_chat_models() -> list[str]:
    """Fetch chat-capable models from Google GenAI API."""
    models = _list_google_models()
    return filter_models_by_capability(models, "google_genai", ModelCapability.CHAT)


def _list_xai_models(*, timeout: float | None = None) -> list[str]:
    """Fetch all models from xAI's models endpoint."""
    try:
        import httpx

        request_kwargs: dict[str, Any] = {
            "headers": {"Authorization": f"Bearer {get_api_key('xai')}"},
        }
        if timeout is not None:
            request_kwargs["timeout"] = timeout

        response = httpx.get("https://api.x.ai/v1/models", **request_kwargs)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        # xAI only has chat models (Grok)
        return sorted(
            model_id
            for model in data
            if isinstance(model, dict) and isinstance((model_id := model.get("id")), str)
        )
    except Exception as e:
        log.warning(f"Failed to fetch xAI models: {e}")
        return []


# Fetcher dispatch - returns ALL models (use filter_models_by_capability for specific types)
_FETCHERS = {
    "openai": _list_openai_models,
    "anthropic": _list_anthropic_models,
    "google_genai": _list_google_models,
    "xai": _list_xai_models,
}

# Convenience fetchers for chat models specifically
_CHAT_FETCHERS = {
    "openai": _list_openai_chat_models,
    "anthropic": _list_anthropic_models,  # All Anthropic models are chat
    "google_genai": _list_google_chat_models,
    "xai": _list_xai_models,  # All xAI models are chat
}


# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------


def _load_cache() -> dict[str, Any]:
    """Load cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE) as f:
            return cast("dict[str, Any]", json.load(f))
    except Exception:
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    """Save cache to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save cache: {e}")


def _is_cache_valid(cache: dict[str, Any], provider: str) -> bool:
    """Check if cache entry is still valid."""
    if provider not in cache:
        return False
    entry = cache[provider]
    if "timestamp" not in entry:
        return False
    timestamp = entry.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return False
    age = time.time() - float(timestamp)
    return age < CACHE_TTL_SECONDS


def clear_cache() -> None:
    """Clear the model cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        log.info("Model cache cleared")


# -----------------------------------------------------------------------------
# Main discovery functions
# -----------------------------------------------------------------------------


def list_models(
    provider: str,
    *,
    capability: ModelCapability | None = None,
    refresh: bool = False,
    use_cache: bool = True,
    timeout: float | None = None,
) -> list[str]:
    """
    List available models for a specific provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        capability: Optional capability to filter by (e.g., ModelCapability.CHAT).
                   If None, returns all models.
        refresh: Force refresh from API, bypassing cache
        use_cache: Whether to use cached results (default: True)
        timeout: Optional request timeout in seconds for HTTP-based fetchers.

    Returns:
        List of model IDs available from the provider.

    Raises:
        ValueError: If provider is not supported.
        RuntimeError: If provider is not configured (no API key).

    Example:
        # Get all models
        all_models = list_models("openai")

        # Get only chat models
        chat_models = list_models("openai", capability=ModelCapability.CHAT)

        # Get only embedding models
        embedding_models = list_models("openai", capability=ModelCapability.EMBEDDING)
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    if not is_provider_configured(provider):
        raise RuntimeError(
            f"Provider '{provider}' is not configured. "
            f"Set {PROVIDER_ENV_VARS[provider]} environment variable."
        )

    # Check cache
    if use_cache and not refresh:
        cache = _load_cache()
        if _is_cache_valid(cache, provider):
            log.debug(f"Using cached models for {provider}")
            cached = cache.get(provider, {}).get("models")
            models: list[str]
            if isinstance(cached, list) and all(isinstance(m, str) for m in cached):
                models = cached
            else:
                models = []
            if capability:
                return filter_models_by_capability(models, provider, capability)
            return models

    # Fetch from API
    log.info(f"Fetching models from {provider}...")
    fetcher = _FETCHERS.get(provider)
    if not fetcher:
        return []

    if provider == "xai":
        models = fetcher(timeout=timeout)
    else:
        models = fetcher()

    # Update cache
    if use_cache and models:
        cache = _load_cache()
        cache[provider] = {
            "models": models,
            "timestamp": time.time(),
        }
        _save_cache(cache)

    # Filter by capability if specified
    if capability:
        return filter_models_by_capability(models, provider, capability)

    return models


def list_all_models(
    *,
    refresh: bool = False,
    use_cache: bool = True,
    skip_unconfigured: bool = True,
    timeout: float | None = None,
) -> dict[str, list[str]]:
    """
    List models for all configured providers.

    Args:
        refresh: Force refresh from API, bypassing cache
        use_cache: Whether to use cached results
        skip_unconfigured: Skip providers without API keys (default: True)
        timeout: Optional request timeout in seconds for HTTP-based fetchers.

    Returns:
        Dict mapping provider name to list of model IDs.
        Example: {"openai": ["gpt-4o", "gpt-4o-mini", ...], ...}
    """
    result: dict[str, list[str]] = {}

    for provider in SUPPORTED_PROVIDERS:
        if not is_provider_configured(provider):
            if skip_unconfigured:
                log.debug(f"Skipping {provider} (not configured)")
                continue
            else:
                result[provider] = []
                continue

        try:
            models = list_models(
                provider,
                refresh=refresh,
                use_cache=use_cache,
                timeout=timeout,
            )
            result[provider] = models
        except Exception as e:
            log.warning(f"Failed to list models for {provider}: {e}")
            result[provider] = []

    return result


# -----------------------------------------------------------------------------
# Convenience exports
# -----------------------------------------------------------------------------

__all__ = [
    # Core functions
    "list_providers",
    "list_configured_providers",
    "list_models",
    "list_all_models",
    "is_provider_configured",
    "get_api_key",
    "clear_cache",
    # Model capability detection
    "ModelCapability",
    "detect_model_capabilities",
    "filter_models_by_capability",
    "categorize_models",
    # Constants
    "SUPPORTED_PROVIDERS",
    "PROVIDER_ENV_VARS",
]
