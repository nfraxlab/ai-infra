"""Provider and model discovery for image generation.

This module provides functions to discover available providers and models,
including live fetching from provider APIs.

Usage:
    from ai_infra.imagegen.discovery import (
        list_providers,
        list_configured_providers,
        list_known_models,
        list_models,
        list_all_models,
        is_provider_configured,
    )

    # List all supported providers
    providers = list_providers()

    # List configured providers (have API keys)
    configured = list_configured_providers()

    # List built-in fallback models for a provider
    known_models = list_known_models("google")

    # Fetch live models from the provider API
    models = list_models("openai")

    # Fetch live models for all configured providers
    all_models = list_all_models()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from ai_infra.providers import ProviderCapability, ProviderRegistry

log = logging.getLogger(__name__)

# Provider aliases for backwards compatibility
_PROVIDER_ALIASES = {"google": "google_genai"}
_REVERSE_ALIASES = {"google_genai": "google"}


# Get supported providers from registry
def _get_supported_providers() -> list[str]:
    """Get imagegen providers from registry."""
    providers = ProviderRegistry.list_for_capability(ProviderCapability.IMAGEGEN)
    # Map back to user-facing names
    return [_REVERSE_ALIASES.get(p, p) for p in providers]


# Legacy constants (built from registry)
SUPPORTED_PROVIDERS = _get_supported_providers()


# Environment variables for each provider (from registry)
def _build_provider_env_vars() -> dict[str, str]:
    """Build PROVIDER_ENV_VARS dict from registry."""
    result = {}
    for name in ProviderRegistry.list_for_capability(ProviderCapability.IMAGEGEN):
        config = ProviderRegistry.get(name)
        if config:
            # Use user-facing name
            user_name = _REVERSE_ALIASES.get(name, name)
            result[user_name] = config.env_var
    return result


PROVIDER_ENV_VARS = _build_provider_env_vars()

# Cache settings
CACHE_TTL = 3600  # 1 hour
CACHE_DIR = Path.home() / ".cache" / "ai_infra" / "imagegen"


def list_providers() -> list[str]:
    """List all supported image generation providers.

    Returns:
        List of provider names.
    """
    return _get_supported_providers()


def is_provider_configured(provider: str) -> bool:
    """Check if a provider has an API key configured.

    Args:
        provider: Provider name.

    Returns:
        True if the provider has an API key set.
    """
    # Map to registry name
    registry_name = _PROVIDER_ALIASES.get(provider, provider)
    return ProviderRegistry.is_configured(registry_name)


def get_api_key(provider: str) -> str | None:
    """Get the API key for a provider.

    Args:
        provider: Provider name.

    Returns:
        API key if configured, None otherwise.
    """
    # Map to registry name
    registry_name = _PROVIDER_ALIASES.get(provider, provider)
    return ProviderRegistry.get_api_key(registry_name)


def list_configured_providers() -> list[str]:
    """List providers that have API keys configured.

    Returns:
        List of configured provider names.
    """
    return [p for p in list_providers() if is_provider_configured(p)]


def list_known_models(provider: str) -> list[str]:
    """List the built-in fallback model catalog for a provider.

    Args:
        provider: Provider name.

    Returns:
        List of statically known model names.

    Raises:
        ValueError: If provider is not supported.
    """
    # Map to registry name
    registry_name = _PROVIDER_ALIASES.get(provider, provider)
    config = ProviderRegistry.get(registry_name)
    if not config:
        raise ValueError(f"Unknown provider: {provider}. Supported: {', '.join(list_providers())}")

    cap = config.get_capability(ProviderCapability.IMAGEGEN)
    if not cap:
        raise ValueError(f"Provider {provider} does not support image generation")

    return cap.models or []


def list_models(
    provider: str,
    *,
    refresh: bool = False,
    use_cache: bool = True,
    timeout: float | None = None,
) -> list[str]:
    """List live models for a provider by querying the provider API.

    This mirrors the LLM discovery API, so callers can rely on ``list_models``
    returning live provider data instead of a hardcoded catalog.

    Args:
        provider: Provider name (e.g., "openai", "google").
        refresh: Force refresh from API, bypassing cache.
        use_cache: Whether to use cached results (default: True).
        timeout: Optional request timeout in seconds for HTTP-based fetchers.

    Returns:
        List of live model IDs available from the provider.
    """
    return list_available_models(
        provider,
        refresh=refresh,
        use_cache=use_cache,
        timeout=timeout,
    )


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------


def _get_cache_path() -> Path:
    """Get the cache file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "models_cache.json"


def _load_cache() -> dict[str, Any]:
    """Load the cache from disk."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        try:
            import json

            with open(cache_path) as f:
                return cast("dict[str, Any]", json.load(f))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    """Save the cache to disk."""
    import json

    cache_path = _get_cache_path()
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save cache: {e}")


def _is_cache_valid(cache: dict[str, Any], provider: str) -> bool:
    """Check if cache entry is valid (not expired)."""
    if provider not in cache:
        return False
    entry = cache[provider]
    if "timestamp" not in entry or "models" not in entry:
        return False
    timestamp = entry.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return False
    return time.time() - float(timestamp) < CACHE_TTL


def clear_cache() -> None:
    """Clear the model cache."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        cache_path.unlink()
        log.info("Cache cleared")


# -----------------------------------------------------------------------------
# Live model fetchers
# -----------------------------------------------------------------------------


def _fetch_openai_models() -> list[str]:
    """Fetch available image models from OpenAI API."""
    import openai

    client = openai.OpenAI()
    models = client.models.list()

    # Filter to image generation models
    image_models = [
        m.id for m in models.data if m.id.startswith("dall-e") or "image" in m.id.lower()
    ]

    return sorted(image_models)


def _fetch_xai_models(*, timeout: float | None = None) -> list[str]:
    """Fetch available image models from xAI's model listing endpoint."""
    import httpx

    api_key = get_api_key("xai")
    request_kwargs: dict[str, Any] = {"headers": {"Authorization": f"Bearer {api_key}"}}
    if timeout is not None:
        request_kwargs["timeout"] = timeout

    response = httpx.get("https://api.x.ai/v1/models", **request_kwargs)
    response.raise_for_status()
    payload = response.json()

    data = payload.get("data", []) if isinstance(payload, dict) else []
    models = [model for model in data if isinstance(model, dict)]

    # xAI's /models payload currently exposes only bare IDs for imagine models,
    # so we filter by the ID itself and explicitly exclude video variants.
    image_models = [
        model_id
        for model in models
        if isinstance((model_id := model.get("id")), str)
        and ("imagine" in model_id.lower() or "image" in model_id.lower())
        and "video" not in model_id.lower()
    ]
    return sorted(set(image_models))


def _fetch_google_models() -> list[str]:
    """Fetch available image models from Google API."""
    from google import genai

    api_key = get_api_key("google")
    client = genai.Client(api_key=api_key)

    # List models and filter for image generation
    image_models = []
    for model in client.models.list():
        name = getattr(model, "name", None) or ""
        # Models that can generate images
        if "imagen" in name.lower() or "image" in name.lower():
            # Remove 'models/' prefix if present
            model_id = name.replace("models/", "")
            image_models.append(model_id)

    return sorted(image_models)


def _fetch_stability_models(*, timeout: float | None = None) -> list[str]:
    """Fetch available models from Stability AI API."""
    import httpx

    api_key = get_api_key("stability")

    try:
        request_kwargs: dict[str, Any] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        if timeout is not None:
            request_kwargs["timeout"] = timeout

        response = httpx.get("https://api.stability.ai/v1/engines/list", **request_kwargs)
        response.raise_for_status()
        engines = response.json()
        return sorted([e["id"] for e in engines])
    except Exception as e:
        log.warning(f"Failed to fetch Stability models: {e}")
        return list_known_models("stability")  # Fall back to known catalog


def _fetch_replicate_models() -> list[str]:
    """Fetch popular image models from Replicate.

    Note: Replicate has thousands of models, so we return curated image models.
    """
    # Replicate doesn't have a simple "list image models" API,
    # so return the built-in curated catalog.
    return list_known_models("replicate")


class _Fetcher(Protocol):
    def __call__(self, *, timeout: float | None = None) -> list[str]: ...


def _wrap_fetcher(fetcher: Callable[[], list[str]]) -> _Fetcher:
    def wrapped(*, timeout: float | None = None) -> list[str]:
        _ = timeout
        return fetcher()

    return wrapped


_FETCHERS: dict[str, _Fetcher] = {
    "openai": _wrap_fetcher(_fetch_openai_models),
    "google": _wrap_fetcher(_fetch_google_models),
    "xai": _fetch_xai_models,
    "stability": _fetch_stability_models,
    "replicate": _wrap_fetcher(_fetch_replicate_models),
}

_TIMEOUT_FETCHER_PROVIDERS = frozenset({"xai", "stability"})


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def list_available_models(
    provider: str,
    *,
    refresh: bool = False,
    use_cache: bool = True,
    timeout: float | None = None,
) -> list[str]:
    """List available models for a provider by querying the API.

    This fetches live data from the provider's API to get the current
    list of available models.

    Args:
        provider: Provider name (e.g., "openai", "google").
        refresh: Force refresh from API, bypassing cache.
        use_cache: Whether to use cached results (default: True).
        timeout: Optional request timeout in seconds for HTTP-based fetchers.

    Returns:
        List of model IDs available from the provider.

    Raises:
        ValueError: If provider is not supported.
        RuntimeError: If provider is not configured (no API key).

    Example:
        >>> list_available_models("openai")
        ['dall-e-2', 'dall-e-3']

        >>> list_available_models("google", refresh=True)
        ['gemini-2.0-flash-exp-image-generation', 'gemini-2.5-flash-image', ...]
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
            cached_models = cache.get(provider, {}).get("models")
            if isinstance(cached_models, list) and all(isinstance(m, str) for m in cached_models):
                return cached_models

    # Fetch from API
    log.info(f"Fetching image models from {provider}...")
    fetcher = _FETCHERS.get(provider)
    if not fetcher:
        return list_known_models(provider)  # Fall back to known catalog

    fetched_models: list[str]
    try:
        if provider in _TIMEOUT_FETCHER_PROVIDERS:
            fetched_models = fetcher(timeout=timeout)
        else:
            fetched_models = fetcher()
    except Exception as e:
        log.warning(f"Failed to fetch models from {provider}: {e}")
        return list_known_models(provider)  # Fall back to known catalog

    # Update cache
    if use_cache and fetched_models:
        cache = _load_cache()
        cache[provider] = {
            "models": fetched_models,
            "timestamp": time.time(),
        }
        _save_cache(cache)

    return fetched_models


def list_all_available_models(
    *,
    refresh: bool = False,
    use_cache: bool = True,
    skip_unconfigured: bool = True,
    timeout: float | None = None,
) -> dict[str, list[str]]:
    """List models for all configured providers.

    Args:
        refresh: Force refresh from API, bypassing cache.
        use_cache: Whether to use cached results.
        skip_unconfigured: Skip providers without API keys (default: True).
        timeout: Optional request timeout in seconds for HTTP-based fetchers.

    Returns:
        Dict mapping provider name to list of model IDs.
        Example: {"openai": ["dall-e-2", "dall-e-3"], ...}
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
            models = list_available_models(
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


def list_all_models(
    *,
    refresh: bool = False,
    use_cache: bool = True,
    skip_unconfigured: bool = True,
    timeout: float | None = None,
) -> dict[str, list[str]]:
    """List live image models for all configured providers.

    This mirrors the LLM discovery API naming while preserving the existing
    ``list_all_available_models`` entry point for backwards compatibility.

    Args:
        refresh: Force refresh from API, bypassing cache.
        use_cache: Whether to use cached results.
        skip_unconfigured: Skip providers without API keys (default: True).
        timeout: Optional request timeout in seconds for HTTP-based fetchers.

    Returns:
        Dict mapping provider name to live model IDs.
    """
    return list_all_available_models(
        refresh=refresh,
        use_cache=use_cache,
        skip_unconfigured=skip_unconfigured,
        timeout=timeout,
    )


__all__ = [
    "PROVIDER_ENV_VARS",
    "SUPPORTED_PROVIDERS",
    "clear_cache",
    "get_api_key",
    "is_provider_configured",
    "list_known_models",
    "list_all_models",
    "list_all_available_models",
    "list_available_models",
    "list_configured_providers",
    "list_models",
    "list_providers",
    "_FETCHERS",
]
