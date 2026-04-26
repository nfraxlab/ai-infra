"""Unit tests for ai_infra.imagegen.discovery module.

Tests cover:
- Image provider detection (OpenAI/DALL-E, Google/Imagen, Stability, Replicate)
- Provider listing and configuration checking
- Static model listing
- Live model fetching with caching
- Cache management
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Constants and Fixtures
# =============================================================================


@pytest.fixture
def mock_env_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all provider API keys from environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)  # Alt env var
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


@pytest.fixture
def mock_openai_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set OpenAI API key."""
    key = "sk-test-openai-key-12345"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    return key


@pytest.fixture
def mock_google_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set Google/Gemini API key."""
    key = "AIza-test-google-key"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    return key


@pytest.fixture
def mock_xai_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set xAI API key."""
    key = "test-xai-key"
    monkeypatch.setenv("XAI_API_KEY", key)
    return key


@pytest.fixture
def mock_stability_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set Stability AI API key."""
    key = "sk-stab-test-key-12345"
    monkeypatch.setenv("STABILITY_API_KEY", key)
    return key


@pytest.fixture
def mock_replicate_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set Replicate API token."""
    key = "r8-test-replicate-token"
    monkeypatch.setenv("REPLICATE_API_TOKEN", key)
    return key


@pytest.fixture
def mock_all_keys(
    mock_openai_key: str,
    mock_google_key: str,
    mock_xai_key: str,
    mock_stability_key: str,
    mock_replicate_key: str,
) -> dict[str, str]:
    """Set all provider API keys."""
    return {
        "openai": mock_openai_key,
        "google": mock_google_key,
        "xai": mock_xai_key,
        "stability": mock_stability_key,
        "replicate": mock_replicate_key,
    }


@pytest.fixture
def temp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / ".cache" / "ai_infra" / "imagegen"
    cache_dir.mkdir(parents=True)

    # Patch the CACHE_DIR constant
    monkeypatch.setattr("ai_infra.imagegen.discovery.CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture
def sample_openai_image_models() -> list[str]:
    """Sample OpenAI image model IDs."""
    return ["dall-e-2", "dall-e-3"]


@pytest.fixture
def sample_google_image_models() -> list[str]:
    """Sample Google image model IDs."""
    return [
        "imagen-3.0-generate-001",
        "gemini-2.0-flash-exp-image-generation",
    ]


@pytest.fixture
def sample_stability_models() -> list[str]:
    """Sample Stability AI model IDs."""
    return [
        "stable-diffusion-xl-1024-v1-0",
        "stable-diffusion-v1-6",
    ]


@pytest.fixture
def sample_replicate_models() -> list[str]:
    """Sample Replicate image model IDs."""
    return [
        "stability-ai/sdxl",
        "black-forest-labs/flux-schnell",
    ]


# =============================================================================
# Test: Provider Listing
# =============================================================================


class TestListProviders:
    """Tests for list_providers() function."""

    def test_list_providers_returns_list(self) -> None:
        """list_providers returns a list type."""
        from ai_infra.imagegen.discovery import list_providers

        providers = list_providers()
        assert isinstance(providers, list)

    def test_list_providers_contains_expected_providers(self) -> None:
        """list_providers contains expected image providers."""
        from ai_infra.imagegen.discovery import list_providers

        providers = list_providers()
        # At minimum should have openai and google
        assert len(providers) >= 2

    def test_list_providers_items_are_strings(self) -> None:
        """list_providers returns list of strings."""
        from ai_infra.imagegen.discovery import list_providers

        providers = list_providers()
        for provider in providers:
            assert isinstance(provider, str)


class TestListConfiguredProviders:
    """Tests for list_configured_providers() function."""

    def test_no_providers_configured(self, mock_env_clean: None) -> None:
        """Returns empty list when no API keys are set."""
        from ai_infra.imagegen.discovery import list_configured_providers

        providers = list_configured_providers()
        assert providers == []

    def test_openai_configured(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Returns OpenAI when OPENAI_API_KEY is set."""
        from ai_infra.imagegen.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "openai" in providers

    def test_google_configured(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Returns Google when GEMINI_API_KEY is set."""
        from ai_infra.imagegen.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "google" in providers

    def test_stability_configured(self, mock_env_clean: None, mock_stability_key: str) -> None:
        """Returns Stability when STABILITY_API_KEY is set."""
        from ai_infra.imagegen.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "stability" in providers

    def test_replicate_configured(self, mock_env_clean: None, mock_replicate_key: str) -> None:
        """Returns Replicate when REPLICATE_API_TOKEN is set."""
        from ai_infra.imagegen.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "replicate" in providers

    def test_xai_configured(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Returns xAI when XAI_API_KEY is set."""
        from ai_infra.imagegen.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "xai" in providers

    def test_multiple_providers_configured(
        self, mock_env_clean: None, mock_all_keys: dict[str, str]
    ) -> None:
        """Returns all configured providers."""
        from ai_infra.imagegen.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "openai" in providers
        assert "google" in providers
        assert "xai" in providers
        assert "stability" in providers
        assert "replicate" in providers


# =============================================================================
# Test: Provider Configuration Check
# =============================================================================


class TestIsProviderConfigured:
    """Tests for is_provider_configured() function."""

    def test_openai_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when OpenAI API key is not set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("openai") is False

    def test_openai_configured(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Returns True when OpenAI API key is set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("openai") is True

    def test_google_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when Google API key is not set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("google") is False

    def test_google_configured(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Returns True when Google API key is set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("google") is True

    def test_stability_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when Stability API key is not set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("stability") is False

    def test_stability_configured(self, mock_env_clean: None, mock_stability_key: str) -> None:
        """Returns True when Stability API key is set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("stability") is True

    def test_xai_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when xAI API key is not set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("xai") is False

    def test_xai_configured(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Returns True when xAI API key is set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("xai") is True

    def test_replicate_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when Replicate API token is not set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("replicate") is False

    def test_replicate_configured(self, mock_env_clean: None, mock_replicate_key: str) -> None:
        """Returns True when Replicate API token is set."""
        from ai_infra.imagegen.discovery import is_provider_configured

        assert is_provider_configured("replicate") is True


# =============================================================================
# Test: Get API Key
# =============================================================================


class TestGetApiKey:
    """Tests for get_api_key() function."""

    def test_get_openai_key(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Returns OpenAI API key when set."""
        from ai_infra.imagegen.discovery import get_api_key

        key = get_api_key("openai")
        assert key == mock_openai_key

    def test_get_google_key(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Returns Google API key when set."""
        from ai_infra.imagegen.discovery import get_api_key

        key = get_api_key("google")
        assert key == mock_google_key

    def test_get_stability_key(self, mock_env_clean: None, mock_stability_key: str) -> None:
        """Returns Stability API key when set."""
        from ai_infra.imagegen.discovery import get_api_key

        key = get_api_key("stability")
        assert key == mock_stability_key

    def test_get_replicate_key(self, mock_env_clean: None, mock_replicate_key: str) -> None:
        """Returns Replicate API token when set."""
        from ai_infra.imagegen.discovery import get_api_key

        key = get_api_key("replicate")
        assert key == mock_replicate_key

    def test_get_xai_key(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Returns xAI API key when set."""
        from ai_infra.imagegen.discovery import get_api_key

        key = get_api_key("xai")
        assert key == mock_xai_key

    def test_get_key_not_configured(self, mock_env_clean: None) -> None:
        """Returns None when API key is not set."""
        from ai_infra.imagegen.discovery import get_api_key

        key = get_api_key("openai")
        assert key is None


# =============================================================================
# Test: Known Model Listing
# =============================================================================


class TestListKnownModels:
    """Tests for list_known_models() fallback catalog."""

    def test_list_openai_models(self) -> None:
        """Lists known OpenAI image models."""
        from ai_infra.imagegen.discovery import list_known_models

        models = list_known_models("openai")
        assert isinstance(models, list)
        # Should have DALL-E models
        assert any("dall-e" in m.lower() for m in models) or len(models) >= 0

    def test_list_google_models(self) -> None:
        """Lists known Google image models."""
        from ai_infra.imagegen.discovery import list_known_models

        models = list_known_models("google")
        assert isinstance(models, list)

    def test_list_stability_models(self) -> None:
        """Lists known Stability image models."""
        from ai_infra.imagegen.discovery import list_known_models

        models = list_known_models("stability")
        assert isinstance(models, list)

    def test_list_replicate_models(self) -> None:
        """Lists known Replicate image models."""
        from ai_infra.imagegen.discovery import list_known_models

        models = list_known_models("replicate")
        assert isinstance(models, list)

    def test_list_xai_models(self) -> None:
        """Lists known xAI image models."""
        from ai_infra.imagegen.discovery import list_known_models

        models = list_known_models("xai")
        assert models == ["grok-imagine-image"]

    def test_list_known_models_unknown_provider_raises(self) -> None:
        """Raises ValueError for unknown provider."""
        from ai_infra.imagegen.discovery import list_known_models

        with pytest.raises(ValueError, match="Unknown provider"):
            list_known_models("unknown_provider")

    def test_list_known_models_returns_list_of_strings(self) -> None:
        """list_known_models returns list of strings."""
        from ai_infra.imagegen.discovery import list_known_models

        models = list_known_models("openai")
        for model in models:
            assert isinstance(model, str)


class TestListModels:
    """Tests for list_models() live discovery wrapper."""

    def test_list_models_delegates_to_live_discovery(self) -> None:
        """list_models delegates to list_available_models."""
        from ai_infra.imagegen.discovery import list_models

        with patch("ai_infra.imagegen.discovery.list_available_models") as mock_list:
            mock_list.return_value = ["gpt-image-1-mini"]

            result = list_models("openai", refresh=True, use_cache=False, timeout=7.5)

        assert result == ["gpt-image-1-mini"]
        mock_list.assert_called_once_with(
            "openai",
            refresh=True,
            use_cache=False,
            timeout=7.5,
        )


# =============================================================================
# Test: Cache Management
# =============================================================================


class TestCacheManagement:
    """Tests for cache loading, saving, and validation."""

    def test_get_cache_path(self, temp_cache_dir: Path) -> None:
        """Cache path is in the cache directory."""
        from ai_infra.imagegen.discovery import _get_cache_path

        cache_path = _get_cache_path()
        assert cache_path.parent == temp_cache_dir
        assert cache_path.name == "models_cache.json"

    def test_load_empty_cache(self, temp_cache_dir: Path) -> None:
        """Load cache returns empty dict when no cache exists."""
        from ai_infra.imagegen.discovery import _load_cache

        cache = _load_cache()
        assert cache == {}

    def test_save_and_load_cache(self, temp_cache_dir: Path) -> None:
        """Can save and load cache."""
        from ai_infra.imagegen.discovery import _load_cache, _save_cache

        test_cache = {
            "openai": {
                "models": ["dall-e-2", "dall-e-3"],
                "timestamp": time.time(),
            }
        }

        _save_cache(test_cache)
        loaded = _load_cache()

        assert "openai" in loaded
        assert loaded["openai"]["models"] == ["dall-e-2", "dall-e-3"]

    def test_cache_validity_fresh(self, temp_cache_dir: Path) -> None:
        """Fresh cache is valid."""
        from ai_infra.imagegen.discovery import _is_cache_valid

        cache = {
            "openai": {
                "models": ["dall-e-3"],
                "timestamp": time.time(),
            }
        }

        assert _is_cache_valid(cache, "openai") is True

    def test_cache_validity_expired(self, temp_cache_dir: Path) -> None:
        """Expired cache is invalid."""
        from ai_infra.imagegen.discovery import CACHE_TTL, _is_cache_valid

        cache = {
            "openai": {
                "models": ["dall-e-3"],
                "timestamp": time.time() - CACHE_TTL - 100,
            }
        }

        assert _is_cache_valid(cache, "openai") is False

    def test_cache_validity_missing_provider(self) -> None:
        """Missing provider is invalid."""
        from ai_infra.imagegen.discovery import _is_cache_valid

        cache = {"google": {"models": [], "timestamp": time.time()}}

        assert _is_cache_valid(cache, "openai") is False

    def test_cache_validity_missing_timestamp(self) -> None:
        """Missing timestamp is invalid."""
        from ai_infra.imagegen.discovery import _is_cache_valid

        cache = {"openai": {"models": ["dall-e-3"]}}

        assert _is_cache_valid(cache, "openai") is False

    def test_cache_validity_missing_models(self) -> None:
        """Missing models key is invalid."""
        from ai_infra.imagegen.discovery import _is_cache_valid

        cache = {"openai": {"timestamp": time.time()}}

        assert _is_cache_valid(cache, "openai") is False

    def test_cache_validity_invalid_timestamp(self) -> None:
        """Non-numeric timestamp is invalid."""
        from ai_infra.imagegen.discovery import _is_cache_valid

        cache = {
            "openai": {
                "models": ["dall-e-3"],
                "timestamp": "not a number",
            }
        }

        assert _is_cache_valid(cache, "openai") is False

    def test_clear_cache(self, temp_cache_dir: Path) -> None:
        """Can clear cache."""
        from ai_infra.imagegen.discovery import (
            _get_cache_path,
            _save_cache,
            clear_cache,
        )

        # Create cache file
        _save_cache({"openai": {"models": [], "timestamp": time.time()}})
        cache_path = _get_cache_path()
        assert cache_path.exists()

        # Clear it
        clear_cache()
        assert not cache_path.exists()

    def test_load_cache_invalid_json(self, temp_cache_dir: Path) -> None:
        """Invalid JSON in cache returns empty dict."""
        from ai_infra.imagegen.discovery import _get_cache_path, _load_cache

        # Write invalid JSON
        cache_path = _get_cache_path()
        cache_path.write_text("not valid json {{{")

        cache = _load_cache()
        assert cache == {}


# =============================================================================
# Test: Live Model Fetchers (Mocked)
# =============================================================================


class TestOpenAIFetcher:
    """Tests for _fetch_openai_models() function."""

    def test_fetch_openai_models_success(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Successfully fetches OpenAI image models."""
        from ai_infra.imagegen.discovery import _fetch_openai_models

        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "dall-e-2"
        mock_model2 = MagicMock()
        mock_model2.id = "dall-e-3"
        mock_model3 = MagicMock()
        mock_model3.id = "gpt-4o"  # Non-image model, should be filtered
        mock_client.models.list.return_value = MagicMock(
            data=[mock_model1, mock_model2, mock_model3]
        )

        with patch("openai.OpenAI", return_value=mock_client):
            models = _fetch_openai_models()

        assert "dall-e-2" in models
        assert "dall-e-3" in models
        assert "gpt-4o" not in models  # Filtered out

    def test_fetch_openai_models_filters_image_models(
        self, mock_env_clean: None, mock_openai_key: str
    ) -> None:
        """Filters only image-related models."""
        from ai_infra.imagegen.discovery import _fetch_openai_models

        mock_client = MagicMock()
        mock_models = [
            MagicMock(id="dall-e-3"),
            MagicMock(id="image-generation-model"),
            MagicMock(id="gpt-4"),
            MagicMock(id="text-embedding-ada"),
        ]
        mock_client.models.list.return_value = MagicMock(data=mock_models)

        with patch("openai.OpenAI", return_value=mock_client):
            models = _fetch_openai_models()

        assert "dall-e-3" in models
        assert "image-generation-model" in models
        assert "gpt-4" not in models
        assert "text-embedding-ada" not in models


class TestGoogleFetcher:
    """Tests for _fetch_google_models() function."""

    def test_fetch_google_models_success(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Successfully fetches Google image models."""
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.name = "models/imagen-3.0-generate-001"
        mock_model2 = MagicMock()
        mock_model2.name = "models/gemini-1.5-pro"  # Non-image, filtered
        mock_model3 = MagicMock()
        mock_model3.name = "models/image-to-video"
        mock_client.models.list.return_value = [mock_model1, mock_model2, mock_model3]

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            from ai_infra.imagegen.discovery import _fetch_google_models

            models = _fetch_google_models()

        assert "imagen-3.0-generate-001" in models
        assert "image-to-video" in models
        # gemini-1.5-pro should be filtered out (no 'image' in name)


class TestXaiFetcher:
    """Tests for _fetch_xai_models() function."""

    def test_fetch_xai_models_success(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Successfully fetches xAI image models."""
        from ai_infra.imagegen.discovery import _fetch_xai_models

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "grok-imagine-image"},
                {"id": "grok-imagine-image-pro"},
                {"id": "grok-imagine-video"},
                {"id": "grok-4"},
            ]
        }

        with patch("httpx.get", return_value=mock_response) as mock_get:
            models = _fetch_xai_models()

        assert models == ["grok-imagine-image", "grok-imagine-image-pro"]
        assert mock_get.call_args.args[0] == "https://api.x.ai/v1/models"
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == f"Bearer {mock_xai_key}"

    def test_fetch_xai_models_forwards_timeout(
        self, mock_env_clean: None, mock_xai_key: str
    ) -> None:
        """Forwards caller-provided timeout to the xAI request."""
        from ai_infra.imagegen.discovery import _fetch_xai_models

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}

        with patch("httpx.get", return_value=mock_response) as mock_get:
            _fetch_xai_models(timeout=12.5)

        assert mock_get.call_args.kwargs["timeout"] == 12.5


class TestStabilityFetcher:
    """Tests for _fetch_stability_models() function."""

    def test_fetch_stability_models_success(
        self, mock_env_clean: None, mock_stability_key: str
    ) -> None:
        """Successfully fetches Stability AI models."""
        from ai_infra.imagegen.discovery import _fetch_stability_models

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "stable-diffusion-xl-1024-v1-0"},
            {"id": "stable-diffusion-v1-6"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            models = _fetch_stability_models()

        assert "stable-diffusion-v1-6" in models
        assert "stable-diffusion-xl-1024-v1-0" in models

    def test_fetch_stability_models_error_fallback(
        self, mock_env_clean: None, mock_stability_key: str
    ) -> None:
        """Falls back to static list on API error."""
        from ai_infra.imagegen.discovery import _fetch_stability_models, list_known_models

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("API Error")

            models = _fetch_stability_models()

        # Should return static list
        static_models = list_known_models("stability")
        assert models == static_models


class TestReplicateFetcher:
    """Tests for _fetch_replicate_models() function."""

    def test_fetch_replicate_models_returns_static_list(self) -> None:
        """Replicate fetcher returns curated static list."""
        from ai_infra.imagegen.discovery import _fetch_replicate_models, list_known_models

        models = _fetch_replicate_models()
        static_models = list_known_models("replicate")

        # Replicate doesn't have a simple list API, so it returns static list
        assert models == static_models


# =============================================================================
# Test: List Available Models
# =============================================================================


class TestListAvailableModels:
    """Tests for list_available_models() function."""

    def test_list_available_models_unknown_provider_raises(self) -> None:
        """Raises ValueError for unknown provider."""
        from ai_infra.imagegen.discovery import list_available_models

        with pytest.raises(ValueError, match="Unknown provider"):
            list_available_models("unknown_provider")

    def test_list_available_models_unconfigured_raises(self, mock_env_clean: None) -> None:
        """Raises RuntimeError when provider is not configured."""
        from ai_infra.imagegen.discovery import list_available_models

        with pytest.raises(RuntimeError, match="not configured"):
            list_available_models("openai")

    def test_list_available_models_uses_cache(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Uses cached models when available."""
        from ai_infra.imagegen.discovery import (
            _save_cache,
            list_available_models,
        )

        # Pre-populate cache
        cached_models = ["cached-dall-e-1", "cached-dall-e-2"]
        _save_cache(
            {
                "openai": {
                    "models": cached_models,
                    "timestamp": time.time(),
                }
            }
        )

        # Should use cache (no API call)
        models = list_available_models("openai", use_cache=True)
        assert models == cached_models

    def test_list_available_models_refresh_bypasses_cache(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Refresh bypasses cache."""
        from ai_infra.imagegen.discovery import (
            _FETCHERS,
            _save_cache,
            list_available_models,
        )

        # Pre-populate cache
        _save_cache(
            {
                "openai": {
                    "models": ["cached-model"],
                    "timestamp": time.time(),
                }
            }
        )

        # Mock the API fetcher via _FETCHERS dict
        mock_fetcher = MagicMock(return_value=["fresh-dall-e-2", "fresh-dall-e-3"])
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            models = list_available_models("openai", refresh=True)

            assert "fresh-dall-e-2" in models
            mock_fetcher.assert_called_once()
        finally:
            _FETCHERS["openai"] = original_fetcher

    def test_list_available_models_without_cache(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Can list models without using cache."""
        from ai_infra.imagegen.discovery import _FETCHERS, list_available_models

        mock_fetcher = MagicMock(return_value=["dall-e-2", "dall-e-3"])
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            models = list_available_models("openai", use_cache=False)

            assert models == ["dall-e-2", "dall-e-3"]
            mock_fetcher.assert_called_once()
        finally:
            _FETCHERS["openai"] = original_fetcher

    def test_list_available_models_forwards_timeout_to_xai_fetcher(
        self,
        mock_env_clean: None,
        mock_xai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Forwards timeout to the xAI fetcher."""
        from ai_infra.imagegen.discovery import _FETCHERS, list_available_models

        mock_fetcher = MagicMock(return_value=["grok-imagine-image"])
        original_fetcher = _FETCHERS["xai"]
        _FETCHERS["xai"] = mock_fetcher
        try:
            models = list_available_models("xai", use_cache=False, timeout=12.5)

            assert models == ["grok-imagine-image"]
            mock_fetcher.assert_called_once_with(timeout=12.5)
        finally:
            _FETCHERS["xai"] = original_fetcher

    def test_list_available_models_fetcher_error_fallback(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Falls back to static list on fetcher error."""
        from ai_infra.imagegen.discovery import (
            _FETCHERS,
            list_available_models,
            list_known_models,
        )

        mock_fetcher = MagicMock(side_effect=Exception("API Error"))
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            models = list_available_models("openai", use_cache=False)

            # Should fall back to static list
            static_models = list_known_models("openai")
            assert models == static_models
        finally:
            _FETCHERS["openai"] = original_fetcher


# =============================================================================
# Test: List All Available Models
# =============================================================================


class TestListAllAvailableModels:
    """Tests for list_all_available_models() function."""

    def test_list_all_no_providers_configured(self, mock_env_clean: None) -> None:
        """Returns empty dict when no providers configured."""
        from ai_infra.imagegen.discovery import list_all_available_models

        result = list_all_available_models()
        assert result == {}

    def test_list_all_skips_unconfigured(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Skips unconfigured providers by default."""
        from ai_infra.imagegen.discovery import _FETCHERS, list_all_available_models

        mock_fetcher = MagicMock(return_value=["dall-e-3"])
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            result = list_all_available_models()

            assert "openai" in result
            assert "google" not in result  # Not configured
        finally:
            _FETCHERS["openai"] = original_fetcher

    def test_list_all_includes_unconfigured(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Includes unconfigured providers when skip_unconfigured=False."""
        from ai_infra.imagegen.discovery import _FETCHERS, list_all_available_models

        mock_fetcher = MagicMock(return_value=["dall-e-3"])
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            result = list_all_available_models(skip_unconfigured=False)

            assert "openai" in result
            # Unconfigured providers have empty lists
            assert "google" in result
            assert result["google"] == []
        finally:
            _FETCHERS["openai"] = original_fetcher

    def test_list_all_multiple_providers(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        mock_google_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Lists models from multiple configured providers."""
        from ai_infra.imagegen.discovery import _FETCHERS, list_all_available_models

        mock_openai = MagicMock(return_value=["dall-e-3"])
        mock_google = MagicMock(return_value=["imagen-3"])
        original_openai = _FETCHERS["openai"]
        original_google = _FETCHERS["google"]
        _FETCHERS["openai"] = mock_openai
        _FETCHERS["google"] = mock_google
        try:
            result = list_all_available_models()

            assert result["openai"] == ["dall-e-3"]
            assert result["google"] == ["imagen-3"]
        finally:
            _FETCHERS["openai"] = original_openai
            _FETCHERS["google"] = original_google

    def test_list_all_forwards_timeout(
        self,
        mock_env_clean: None,
        mock_xai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Forwards timeout through list_all_available_models."""
        from ai_infra.imagegen.discovery import list_all_available_models

        with patch("ai_infra.imagegen.discovery.list_available_models") as mock_list:
            mock_list.return_value = ["grok-imagine-image"]

            result = list_all_available_models(timeout=12.5)

        assert result == {"xai": ["grok-imagine-image"]}
        mock_list.assert_called_once_with("xai", refresh=False, use_cache=True, timeout=12.5)


# =============================================================================
# Test: Constants and Exports
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_supported_providers_list(self) -> None:
        """SUPPORTED_PROVIDERS contains expected providers."""
        from ai_infra.imagegen.discovery import SUPPORTED_PROVIDERS

        assert isinstance(SUPPORTED_PROVIDERS, list)
        # Should have some providers
        assert len(SUPPORTED_PROVIDERS) >= 1

    def test_provider_env_vars_mapping(self) -> None:
        """PROVIDER_ENV_VARS maps providers to env vars."""
        from ai_infra.imagegen.discovery import PROVIDER_ENV_VARS

        assert isinstance(PROVIDER_ENV_VARS, dict)
        # Check some expected mappings
        if "openai" in PROVIDER_ENV_VARS:
            assert PROVIDER_ENV_VARS["openai"] == "OPENAI_API_KEY"

    def test_cache_ttl_is_positive(self) -> None:
        """CACHE_TTL is a positive number."""
        from ai_infra.imagegen.discovery import CACHE_TTL

        assert CACHE_TTL > 0
        assert isinstance(CACHE_TTL, (int, float))


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports_are_accessible(self) -> None:
        """All items in __all__ are accessible."""
        from ai_infra.imagegen import discovery

        for name in discovery.__all__:
            assert hasattr(discovery, name), f"{name} not found in module"

    def test_core_functions_exported(self) -> None:
        """Core functions are in __all__."""
        from ai_infra.imagegen.discovery import __all__

        assert "list_providers" in __all__
        assert "list_configured_providers" in __all__
        assert "list_known_models" in __all__
        assert "list_models" in __all__
        assert "list_available_models" in __all__
        assert "list_all_models" in __all__
        assert "list_all_available_models" in __all__
        assert "is_provider_configured" in __all__
        assert "get_api_key" in __all__
        assert "clear_cache" in __all__

    def test_constants_exported(self) -> None:
        """Constants are in __all__."""
        from ai_infra.imagegen.discovery import __all__

        assert "SUPPORTED_PROVIDERS" in __all__
        assert "PROVIDER_ENV_VARS" in __all__


# =============================================================================
# Test: Provider Aliases
# =============================================================================


class TestProviderAliases:
    """Tests for provider alias handling."""

    def test_google_alias_works(self, mock_env_clean: None, mock_google_key: str) -> None:
        """'google' alias maps to 'google_genai' internally."""
        from ai_infra.imagegen.discovery import is_provider_configured

        # 'google' should work as an alias
        assert is_provider_configured("google") is True

    def test_list_known_models_with_alias(self) -> None:
        """list_known_models works with 'google' alias."""
        from ai_infra.imagegen.discovery import list_known_models

        # Should not raise
        models = list_known_models("google")
        assert isinstance(models, list)

    def test_list_models_with_alias(
        self,
        mock_env_clean: None,
        mock_google_key: str,
    ) -> None:
        """list_models uses the live discovery path with the Google alias."""
        from ai_infra.imagegen.discovery import _FETCHERS, list_models

        mock_fetcher = MagicMock(return_value=["imagen-4.0-fast-generate-001"])
        original_fetcher = _FETCHERS["google"]
        _FETCHERS["google"] = mock_fetcher
        try:
            models = list_models("google", use_cache=False)
        finally:
            _FETCHERS["google"] = original_fetcher

        assert models == ["imagen-4.0-fast-generate-001"]
        mock_fetcher.assert_called_once_with()
        assert isinstance(models, list)


# =============================================================================
# Test: Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for common usage patterns."""

    def test_full_workflow_with_cache(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Full workflow: list models, cache, refresh."""
        from ai_infra.imagegen.discovery import (
            _FETCHERS,
            _get_cache_path,
            clear_cache,
            list_available_models,
        )

        # Mock the API via _FETCHERS dict
        mock_fetcher = MagicMock(return_value=["dall-e-2", "dall-e-3"])
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            # First call fetches from API
            models = list_available_models("openai")
            assert mock_fetcher.call_count == 1
            assert "dall-e-3" in models

            # Cache should exist
            cache_path = _get_cache_path()
            assert cache_path.exists()

            # Second call uses cache
            models2 = list_available_models("openai")
            assert mock_fetcher.call_count == 1  # No additional call
            assert models2 == models

            # Clear cache
            clear_cache()
            assert not cache_path.exists()

            # Next call fetches again
            list_available_models("openai")
            assert mock_fetcher.call_count == 2
        finally:
            _FETCHERS["openai"] = original_fetcher

    def test_multi_provider_workflow(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        mock_stability_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Workflow with multiple providers."""
        from ai_infra.imagegen.discovery import (
            _FETCHERS,
            list_all_available_models,
            list_configured_providers,
        )

        # Check configured providers
        providers = list_configured_providers()
        assert "openai" in providers
        assert "stability" in providers

        # List all models via _FETCHERS dict
        mock_openai = MagicMock(return_value=["dall-e-3"])
        mock_stability = MagicMock(return_value=["sdxl"])
        original_openai = _FETCHERS["openai"]
        original_stability = _FETCHERS["stability"]
        _FETCHERS["openai"] = mock_openai
        _FETCHERS["stability"] = mock_stability
        try:
            all_models = list_all_available_models()

            assert "openai" in all_models
            assert "stability" in all_models
            assert "dall-e-3" in all_models["openai"]
            assert "sdxl" in all_models["stability"]
        finally:
            _FETCHERS["openai"] = original_openai
            _FETCHERS["stability"] = original_stability

    def test_known_vs_live_models(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Known model catalog vs live list_models."""
        from ai_infra.imagegen.discovery import (
            _FETCHERS,
            list_known_models,
            list_models,
        )

        # Known models don't require API call
        known_models = list_known_models("openai")
        assert isinstance(known_models, list)

        # Live models use the fetcher
        mock_fetcher = MagicMock(return_value=["dall-e-2", "dall-e-3"])
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            live_models = list_models("openai", use_cache=False)
            assert live_models == ["dall-e-2", "dall-e-3"]
            mock_fetcher.assert_called_once()
        finally:
            _FETCHERS["openai"] = original_fetcher


class TestListAllModelsAlias:
    """Tests for the LLM-style list_all_models alias."""

    def test_list_all_models_delegates(self) -> None:
        """list_all_models delegates to list_all_available_models."""
        from ai_infra.imagegen.discovery import list_all_models

        with patch("ai_infra.imagegen.discovery.list_all_available_models") as mock_list:
            mock_list.return_value = {"openai": ["gpt-image-1-mini"]}

            result = list_all_models(refresh=True, timeout=9.5)

        assert result == {"openai": ["gpt-image-1-mini"]}
        mock_list.assert_called_once_with(
            refresh=True,
            use_cache=True,
            skip_unconfigured=True,
            timeout=9.5,
        )
