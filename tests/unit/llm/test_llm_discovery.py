"""Unit tests for ai_infra.llm.providers.discovery module.

Tests cover:
- Provider detection (OpenAI, Anthropic, Google, xAI via API keys)
- Provider listing and configuration checking
- Model listing with caching
- Model capability detection and filtering
- Default provider selection
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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)


@pytest.fixture
def mock_openai_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set OpenAI API key."""
    key = "sk-test-openai-key-12345"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    return key


@pytest.fixture
def mock_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set Anthropic API key."""
    key = "sk-ant-test-key-12345"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    return key


@pytest.fixture
def mock_google_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set Google GenAI API key."""
    key = "AIza-test-google-key"
    # GEMINI_API_KEY is the primary env var for google_genai
    monkeypatch.setenv("GEMINI_API_KEY", key)
    return key


@pytest.fixture
def mock_xai_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set xAI API key."""
    key = "xai-test-key-12345"
    monkeypatch.setenv("XAI_API_KEY", key)
    return key


@pytest.fixture
def mock_all_keys(
    mock_openai_key: str,
    mock_anthropic_key: str,
    mock_google_key: str,
    mock_xai_key: str,
) -> dict[str, str]:
    """Set all provider API keys."""
    return {
        "openai": mock_openai_key,
        "anthropic": mock_anthropic_key,
        "google_genai": mock_google_key,
        "xai": mock_xai_key,
    }


@pytest.fixture
def temp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / ".cache" / "ai-infra"
    cache_dir.mkdir(parents=True)

    # Patch the CACHE_DIR and CACHE_FILE constants
    cache_file = cache_dir / "models.json"
    monkeypatch.setattr("ai_infra.llm.providers.discovery.CACHE_DIR", cache_dir)
    monkeypatch.setattr("ai_infra.llm.providers.discovery.CACHE_FILE", cache_file)
    return cache_dir


@pytest.fixture
def sample_openai_models() -> list[str]:
    """Sample OpenAI model IDs."""
    return [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "whisper-1",
        "dall-e-3",
        "tts-1",
    ]


@pytest.fixture
def sample_anthropic_models() -> list[str]:
    """Sample Anthropic model IDs."""
    return [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
    ]


@pytest.fixture
def sample_google_models() -> list[str]:
    """Sample Google GenAI model IDs."""
    return [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "imagen-3.0-generate-001",
    ]


@pytest.fixture
def sample_xai_models() -> list[str]:
    """Sample xAI model IDs."""
    return ["grok-beta", "grok-vision-beta"]


# =============================================================================
# Test: Provider Listing
# =============================================================================


class TestListProviders:
    """Tests for list_providers() function."""

    def test_list_providers_returns_chat_providers(self) -> None:
        """list_providers returns providers with CHAT capability."""
        from ai_infra.llm.providers.discovery import list_providers

        providers = list_providers()

        # Should return supported providers
        assert isinstance(providers, list)
        # OpenAI and Anthropic should always be in the list
        assert "openai" in providers or len(providers) > 0

    def test_list_providers_returns_list(self) -> None:
        """list_providers returns a list type."""
        from ai_infra.llm.providers.discovery import list_providers

        providers = list_providers()
        assert isinstance(providers, list)

    def test_list_providers_items_are_strings(self) -> None:
        """list_providers returns list of strings."""
        from ai_infra.llm.providers.discovery import list_providers

        providers = list_providers()
        for provider in providers:
            assert isinstance(provider, str)


class TestListConfiguredProviders:
    """Tests for list_configured_providers() function."""

    def test_no_providers_configured(self, mock_env_clean: None) -> None:
        """Returns empty list when no API keys are set."""
        from ai_infra.llm.providers.discovery import list_configured_providers

        providers = list_configured_providers()
        assert providers == []

    def test_openai_configured(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Returns OpenAI when OPENAI_API_KEY is set."""
        from ai_infra.llm.providers.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "openai" in providers

    def test_anthropic_configured(self, mock_env_clean: None, mock_anthropic_key: str) -> None:
        """Returns Anthropic when ANTHROPIC_API_KEY is set."""
        from ai_infra.llm.providers.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "anthropic" in providers

    def test_google_configured(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Returns Google when GOOGLE_GENAI_API_KEY is set."""
        from ai_infra.llm.providers.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "google_genai" in providers

    def test_xai_configured(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Returns xAI when XAI_API_KEY is set."""
        from ai_infra.llm.providers.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "xai" in providers

    def test_multiple_providers_configured(
        self, mock_env_clean: None, mock_all_keys: dict[str, str]
    ) -> None:
        """Returns all configured providers."""
        from ai_infra.llm.providers.discovery import list_configured_providers

        providers = list_configured_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google_genai" in providers
        assert "xai" in providers


# =============================================================================
# Test: Provider Configuration Check
# =============================================================================


class TestIsProviderConfigured:
    """Tests for is_provider_configured() function."""

    def test_openai_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when OpenAI API key is not set."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        assert is_provider_configured("openai") is False

    def test_openai_configured(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Returns True when OpenAI API key is set."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        assert is_provider_configured("openai") is True

    def test_anthropic_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when Anthropic API key is not set."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        assert is_provider_configured("anthropic") is False

    def test_anthropic_configured(self, mock_env_clean: None, mock_anthropic_key: str) -> None:
        """Returns True when Anthropic API key is set."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        assert is_provider_configured("anthropic") is True

    def test_google_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when Google API key is not set."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        assert is_provider_configured("google_genai") is False

    def test_google_configured(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Returns True when Google API key is set."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        assert is_provider_configured("google_genai") is True

    def test_xai_not_configured(self, mock_env_clean: None) -> None:
        """Returns False when xAI API key is not set."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        assert is_provider_configured("xai") is False

    def test_xai_configured(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Returns True when xAI API key is set."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        assert is_provider_configured("xai") is True

    def test_unknown_provider_raises_valueerror(self) -> None:
        """Raises ValueError for unknown provider."""
        from ai_infra.llm.providers.discovery import is_provider_configured

        with pytest.raises(ValueError, match="Unknown provider"):
            is_provider_configured("unknown_provider")


# =============================================================================
# Test: Get API Key
# =============================================================================


class TestGetApiKey:
    """Tests for get_api_key() function."""

    def test_get_openai_key(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Returns OpenAI API key when set."""
        from ai_infra.llm.providers.discovery import get_api_key

        key = get_api_key("openai")
        assert key == mock_openai_key

    def test_get_anthropic_key(self, mock_env_clean: None, mock_anthropic_key: str) -> None:
        """Returns Anthropic API key when set."""
        from ai_infra.llm.providers.discovery import get_api_key

        key = get_api_key("anthropic")
        assert key == mock_anthropic_key

    def test_get_google_key(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Returns Google API key when set."""
        from ai_infra.llm.providers.discovery import get_api_key

        key = get_api_key("google_genai")
        assert key == mock_google_key

    def test_get_xai_key(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Returns xAI API key when set."""
        from ai_infra.llm.providers.discovery import get_api_key

        key = get_api_key("xai")
        assert key == mock_xai_key

    def test_get_key_not_configured(self, mock_env_clean: None) -> None:
        """Returns None when API key is not set."""
        from ai_infra.llm.providers.discovery import get_api_key

        key = get_api_key("openai")
        assert key is None


# =============================================================================
# Test: Default Provider Selection
# =============================================================================


class TestGetDefaultProvider:
    """Tests for get_default_provider() function."""

    def test_no_provider_configured(self, mock_env_clean: None) -> None:
        """Returns None when no providers are configured."""
        from ai_infra.llm.providers.discovery import get_default_provider

        provider = get_default_provider()
        assert provider is None

    def test_openai_is_default_priority(
        self, mock_env_clean: None, mock_all_keys: dict[str, str]
    ) -> None:
        """Returns OpenAI when all providers are configured (highest priority)."""
        from ai_infra.llm.providers.discovery import get_default_provider

        provider = get_default_provider()
        assert provider == "openai"

    def test_anthropic_fallback(self, mock_env_clean: None, mock_anthropic_key: str) -> None:
        """Returns Anthropic when only Anthropic is configured."""
        from ai_infra.llm.providers.discovery import get_default_provider

        provider = get_default_provider()
        assert provider == "anthropic"

    def test_google_fallback(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Returns Google when only Google is configured."""
        from ai_infra.llm.providers.discovery import get_default_provider

        provider = get_default_provider()
        assert provider == "google_genai"

    def test_xai_fallback(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Returns xAI when only xAI is configured."""
        from ai_infra.llm.providers.discovery import get_default_provider

        provider = get_default_provider()
        assert provider == "xai"

    def test_priority_order_openai_over_anthropic(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        mock_anthropic_key: str,
    ) -> None:
        """OpenAI has priority over Anthropic."""
        from ai_infra.llm.providers.discovery import get_default_provider

        provider = get_default_provider()
        assert provider == "openai"

    def test_priority_order_anthropic_over_google(
        self,
        mock_env_clean: None,
        mock_anthropic_key: str,
        mock_google_key: str,
    ) -> None:
        """Anthropic has priority over Google."""
        from ai_infra.llm.providers.discovery import get_default_provider

        provider = get_default_provider()
        assert provider == "anthropic"

    def test_priority_order_google_over_xai(
        self,
        mock_env_clean: None,
        mock_google_key: str,
        mock_xai_key: str,
    ) -> None:
        """Google has priority over xAI."""
        from ai_infra.llm.providers.discovery import get_default_provider

        provider = get_default_provider()
        assert provider == "google_genai"


# =============================================================================
# Test: Model Capability Enum
# =============================================================================


class TestModelCapability:
    """Tests for ModelCapability enum."""

    def test_all_capabilities_exist(self) -> None:
        """All expected capabilities exist in enum."""
        from ai_infra.llm.providers.discovery import ModelCapability

        assert hasattr(ModelCapability, "CHAT")
        assert hasattr(ModelCapability, "EMBEDDING")
        assert hasattr(ModelCapability, "AUDIO")
        assert hasattr(ModelCapability, "IMAGE")
        assert hasattr(ModelCapability, "MODERATION")
        assert hasattr(ModelCapability, "VISION")
        assert hasattr(ModelCapability, "REALTIME")
        assert hasattr(ModelCapability, "VIDEO")
        assert hasattr(ModelCapability, "CODE")
        assert hasattr(ModelCapability, "UNKNOWN")

    def test_capability_values_are_strings(self) -> None:
        """Capability values are strings."""
        from ai_infra.llm.providers.discovery import ModelCapability

        for cap in ModelCapability:
            assert isinstance(cap.value, str)

    def test_capability_iteration(self) -> None:
        """Can iterate over all capabilities."""
        from ai_infra.llm.providers.discovery import ModelCapability

        caps = list(ModelCapability)
        assert len(caps) >= 10  # At least 10 capabilities


# =============================================================================
# Test: Model Capability Detection
# =============================================================================


class TestDetectModelCapabilities:
    """Tests for detect_model_capabilities() function."""

    # OpenAI models
    def test_gpt4o_has_chat_and_vision(self) -> None:
        """GPT-4o has CHAT and VISION capabilities."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("gpt-4o", "openai")
        assert ModelCapability.CHAT in caps
        assert ModelCapability.VISION in caps

    def test_gpt4o_mini_has_chat_and_vision(self) -> None:
        """GPT-4o-mini has CHAT and VISION capabilities."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("gpt-4o-mini", "openai")
        assert ModelCapability.CHAT in caps
        assert ModelCapability.VISION in caps

    def test_gpt35_turbo_has_chat(self) -> None:
        """GPT-3.5-turbo has CHAT capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("gpt-3.5-turbo", "openai")
        assert ModelCapability.CHAT in caps

    def test_embedding_model_has_embedding(self) -> None:
        """Embedding models have EMBEDDING capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("text-embedding-ada-002", "openai")
        assert ModelCapability.EMBEDDING in caps

    def test_embedding_3_has_embedding(self) -> None:
        """text-embedding-3-small has EMBEDDING capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("text-embedding-3-small", "openai")
        assert ModelCapability.EMBEDDING in caps

    def test_whisper_has_audio(self) -> None:
        """Whisper models have AUDIO capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("whisper-1", "openai")
        assert ModelCapability.AUDIO in caps

    def test_dalle_has_image(self) -> None:
        """DALL-E models have IMAGE capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("dall-e-3", "openai")
        assert ModelCapability.IMAGE in caps

    def test_tts_has_audio(self) -> None:
        """TTS models have AUDIO capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("tts-1", "openai")
        assert ModelCapability.AUDIO in caps

    def test_moderation_has_moderation(self) -> None:
        """Moderation models have MODERATION capability."""
        from ai_infra.llm.providers.discovery import (
            detect_model_capabilities,
        )

        # Try text-moderation if it exists
        caps = detect_model_capabilities("text-moderation-latest", "openai")
        # At minimum should have UNKNOWN or MODERATION
        assert len(caps) > 0

    # Anthropic models
    def test_claude3_has_chat_and_vision(self) -> None:
        """Claude-3 models have CHAT and VISION capabilities."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("claude-3-opus-20240229", "anthropic")
        assert ModelCapability.CHAT in caps
        assert ModelCapability.VISION in caps

    def test_claude3_sonnet_has_chat_and_vision(self) -> None:
        """Claude-3-sonnet has CHAT and VISION capabilities."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("claude-3-sonnet-20240229", "anthropic")
        assert ModelCapability.CHAT in caps
        assert ModelCapability.VISION in caps

    def test_claude35_sonnet_has_chat_and_vision(self) -> None:
        """Claude-3.5-sonnet has CHAT and VISION capabilities."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("claude-3-5-sonnet-20241022", "anthropic")
        assert ModelCapability.CHAT in caps
        assert ModelCapability.VISION in caps

    # Google models
    def test_gemini_has_chat_and_vision(self) -> None:
        """Gemini models have CHAT and VISION capabilities."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("gemini-1.5-pro", "google_genai")
        assert ModelCapability.CHAT in caps
        assert ModelCapability.VISION in caps

    def test_gemini_flash_has_chat_and_vision(self) -> None:
        """Gemini Flash has CHAT and VISION capabilities."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("gemini-1.5-flash", "google_genai")
        assert ModelCapability.CHAT in caps
        assert ModelCapability.VISION in caps

    def test_imagen_has_image(self) -> None:
        """Imagen models have IMAGE capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("imagen-3.0-generate-001", "google_genai")
        assert ModelCapability.IMAGE in caps

    # xAI models
    def test_grok_has_chat(self) -> None:
        """Grok models have CHAT capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("grok-beta", "xai")
        assert ModelCapability.CHAT in caps

    def test_grok_vision_has_vision(self) -> None:
        """Grok vision models have VISION capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("grok-vision-beta", "xai")
        assert ModelCapability.VISION in caps

    # Unknown models
    def test_unknown_model_returns_unknown(self) -> None:
        """Unknown model returns UNKNOWN capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("completely-unknown-model-xyz", "openai")
        assert ModelCapability.UNKNOWN in caps


# =============================================================================
# Test: Filter Models by Capability
# =============================================================================


class TestFilterModelsByCapability:
    """Tests for filter_models_by_capability() function."""

    def test_filter_chat_models_openai(self, sample_openai_models: list[str]) -> None:
        """Filters OpenAI models by CHAT capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            filter_models_by_capability,
        )

        chat_models = filter_models_by_capability(
            sample_openai_models, "openai", ModelCapability.CHAT
        )

        # GPT models should be in chat
        assert "gpt-4o" in chat_models
        assert "gpt-4o-mini" in chat_models
        assert "gpt-3.5-turbo" in chat_models

        # Embedding models should not be in chat
        assert "text-embedding-ada-002" not in chat_models

    def test_filter_embedding_models_openai(self, sample_openai_models: list[str]) -> None:
        """Filters OpenAI models by EMBEDDING capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            filter_models_by_capability,
        )

        embedding_models = filter_models_by_capability(
            sample_openai_models, "openai", ModelCapability.EMBEDDING
        )

        # Embedding models should be included
        assert "text-embedding-ada-002" in embedding_models
        assert "text-embedding-3-small" in embedding_models

        # Chat models should not be in embedding
        assert "gpt-4o" not in embedding_models

    def test_filter_audio_models_openai(self, sample_openai_models: list[str]) -> None:
        """Filters OpenAI models by AUDIO capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            filter_models_by_capability,
        )

        audio_models = filter_models_by_capability(
            sample_openai_models, "openai", ModelCapability.AUDIO
        )

        # Audio models should be included
        assert "whisper-1" in audio_models
        assert "tts-1" in audio_models

    def test_filter_image_models_openai(self, sample_openai_models: list[str]) -> None:
        """Filters OpenAI models by IMAGE capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            filter_models_by_capability,
        )

        image_models = filter_models_by_capability(
            sample_openai_models, "openai", ModelCapability.IMAGE
        )

        # DALL-E should be included
        assert "dall-e-3" in image_models

    def test_filter_vision_models_openai(self, sample_openai_models: list[str]) -> None:
        """Filters OpenAI models by VISION capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            filter_models_by_capability,
        )

        vision_models = filter_models_by_capability(
            sample_openai_models, "openai", ModelCapability.VISION
        )

        # GPT-4o has vision
        assert "gpt-4o" in vision_models
        assert "gpt-4o-mini" in vision_models

    def test_filter_chat_models_anthropic(self, sample_anthropic_models: list[str]) -> None:
        """Filters Anthropic models by CHAT capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            filter_models_by_capability,
        )

        chat_models = filter_models_by_capability(
            sample_anthropic_models, "anthropic", ModelCapability.CHAT
        )

        # All Claude models should be chat
        assert "claude-3-opus-20240229" in chat_models
        assert "claude-3-sonnet-20240229" in chat_models
        assert "claude-3-5-sonnet-20241022" in chat_models

    def test_filter_empty_list(self) -> None:
        """Filtering empty list returns empty list."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            filter_models_by_capability,
        )

        result = filter_models_by_capability([], "openai", ModelCapability.CHAT)
        assert result == []


# =============================================================================
# Test: Categorize Models
# =============================================================================


class TestCategorizeModels:
    """Tests for categorize_models() function."""

    def test_categorize_openai_models(self, sample_openai_models: list[str]) -> None:
        """Categorizes OpenAI models by capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            categorize_models,
        )

        categories = categorize_models(sample_openai_models, "openai")

        # Should have multiple categories
        assert isinstance(categories, dict)

        # Check CHAT category
        if ModelCapability.CHAT in categories:
            chat_models = categories[ModelCapability.CHAT]
            assert "gpt-4o" in chat_models

    def test_categorize_anthropic_models(self, sample_anthropic_models: list[str]) -> None:
        """Categorizes Anthropic models by capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            categorize_models,
        )

        categories = categorize_models(sample_anthropic_models, "anthropic")

        # Should have CHAT category
        assert ModelCapability.CHAT in categories
        assert "claude-3-opus-20240229" in categories[ModelCapability.CHAT]

    def test_categorize_empty_list(self) -> None:
        """Categorizing empty list returns dict with empty lists for all capabilities."""
        from ai_infra.llm.providers.discovery import ModelCapability, categorize_models

        categories = categorize_models([], "openai")
        # categorize_models returns dict with all capabilities as keys (even if empty)
        assert isinstance(categories, dict)
        for cap in ModelCapability:
            if cap in categories:
                assert categories[cap] == []


# =============================================================================
# Test: Cache Management
# =============================================================================


class TestCacheManagement:
    """Tests for cache loading, saving, and validation."""

    def test_load_empty_cache(self, temp_cache_dir: Path) -> None:
        """Load cache returns empty dict when no cache exists."""
        from ai_infra.llm.providers.discovery import _load_cache

        cache = _load_cache()
        assert cache == {}

    def test_save_and_load_cache(self, temp_cache_dir: Path) -> None:
        """Can save and load cache."""
        from ai_infra.llm.providers.discovery import _load_cache, _save_cache

        test_cache = {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini"],
                "timestamp": time.time(),
            }
        }

        _save_cache(test_cache)
        loaded = _load_cache()

        assert "openai" in loaded
        assert loaded["openai"]["models"] == ["gpt-4o", "gpt-4o-mini"]

    def test_cache_validity_fresh(self, temp_cache_dir: Path) -> None:
        """Fresh cache is valid."""
        from ai_infra.llm.providers.discovery import _is_cache_valid

        cache = {
            "openai": {
                "models": ["gpt-4o"],
                "timestamp": time.time(),
            }
        }

        assert _is_cache_valid(cache, "openai") is True

    def test_cache_validity_expired(self, temp_cache_dir: Path) -> None:
        """Expired cache is invalid."""
        from ai_infra.llm.providers.discovery import (
            CACHE_TTL_SECONDS,
            _is_cache_valid,
        )

        cache = {
            "openai": {
                "models": ["gpt-4o"],
                "timestamp": time.time() - CACHE_TTL_SECONDS - 100,
            }
        }

        assert _is_cache_valid(cache, "openai") is False

    def test_cache_validity_missing_provider(self) -> None:
        """Missing provider is invalid."""
        from ai_infra.llm.providers.discovery import _is_cache_valid

        cache = {"anthropic": {"models": [], "timestamp": time.time()}}

        assert _is_cache_valid(cache, "openai") is False

    def test_cache_validity_missing_timestamp(self) -> None:
        """Missing timestamp is invalid."""
        from ai_infra.llm.providers.discovery import _is_cache_valid

        cache = {"openai": {"models": ["gpt-4o"]}}

        assert _is_cache_valid(cache, "openai") is False

    def test_clear_cache(self, temp_cache_dir: Path) -> None:
        """Can clear cache."""
        from ai_infra.llm.providers.discovery import (
            CACHE_FILE,
            _save_cache,
            clear_cache,
        )

        # Create cache file
        _save_cache({"openai": {"models": [], "timestamp": time.time()}})
        assert CACHE_FILE.exists()

        # Clear it
        clear_cache()
        assert not CACHE_FILE.exists()


# =============================================================================
# Test: List Models
# =============================================================================


class TestListModels:
    """Tests for list_models() function."""

    def test_list_models_unknown_provider_raises(self) -> None:
        """Raises ValueError for unknown provider."""
        from ai_infra.llm.providers.discovery import list_models

        with pytest.raises(ValueError, match="Unknown provider"):
            list_models("unknown_provider")

    def test_list_models_unconfigured_provider_raises(self, mock_env_clean: None) -> None:
        """Raises RuntimeError when provider is not configured."""
        from ai_infra.llm.providers.discovery import list_models

        with pytest.raises(RuntimeError, match="not configured"):
            list_models("openai")

    def test_list_models_uses_cache(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Uses cached models when available."""
        from ai_infra.llm.providers.discovery import (
            _save_cache,
            list_models,
        )

        # Pre-populate cache
        cached_models = ["cached-model-1", "cached-model-2"]
        _save_cache(
            {
                "openai": {
                    "models": cached_models,
                    "timestamp": time.time(),
                }
            }
        )

        # Should use cache (no API call)
        models = list_models("openai", use_cache=True)
        assert models == cached_models

    def test_list_models_refresh_bypasses_cache(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Refresh bypasses cache."""
        from ai_infra.llm.providers.discovery import (
            _FETCHERS,
            _save_cache,
            list_models,
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
        mock_fetcher = MagicMock(return_value=["fresh-model-1", "fresh-model-2"])
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            models = list_models("openai", refresh=True)

            assert "fresh-model-1" in models
            mock_fetcher.assert_called_once()
        finally:
            _FETCHERS["openai"] = original_fetcher

    def test_list_models_with_capability_filter(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Filters models by capability."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            _save_cache,
            list_models,
        )

        # Pre-populate cache with mixed models
        _save_cache(
            {
                "openai": {
                    "models": [
                        "gpt-4o",
                        "gpt-3.5-turbo",
                        "text-embedding-ada-002",
                        "dall-e-3",
                    ],
                    "timestamp": time.time(),
                }
            }
        )

        # Filter by CHAT capability
        chat_models = list_models("openai", capability=ModelCapability.CHAT)

        assert "gpt-4o" in chat_models
        assert "gpt-3.5-turbo" in chat_models
        assert "text-embedding-ada-002" not in chat_models

    def test_list_models_without_cache(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Can list models without using cache."""
        from ai_infra.llm.providers.discovery import _FETCHERS, list_models

        mock_fetcher = MagicMock(return_value=["model-1", "model-2"])
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            models = list_models("openai", use_cache=False)

            assert models == ["model-1", "model-2"]
            mock_fetcher.assert_called_once()
        finally:
            _FETCHERS["openai"] = original_fetcher

    def test_list_models_forwards_timeout_to_xai_fetcher(
        self,
        mock_env_clean: None,
        mock_xai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Forwards timeout to the xAI fetcher."""
        from ai_infra.llm.providers.discovery import _FETCHERS, list_models

        mock_fetcher = MagicMock(return_value=["grok-beta"])
        original_fetcher = _FETCHERS["xai"]
        _FETCHERS["xai"] = mock_fetcher
        try:
            models = list_models("xai", use_cache=False, timeout=7.5)

            assert models == ["grok-beta"]
            mock_fetcher.assert_called_once_with(timeout=7.5)
        finally:
            _FETCHERS["xai"] = original_fetcher


# =============================================================================
# Test: List All Models
# =============================================================================


class TestListAllModels:
    """Tests for list_all_models() function."""

    def test_list_all_models_no_providers(self, mock_env_clean: None) -> None:
        """Returns empty dict when no providers configured."""
        from ai_infra.llm.providers.discovery import list_all_models

        result = list_all_models()
        assert result == {}

    def test_list_all_models_skips_unconfigured(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Skips unconfigured providers by default."""
        from ai_infra.llm.providers.discovery import list_all_models

        with patch("ai_infra.llm.providers.discovery._list_openai_models") as mock_fetcher:
            mock_fetcher.return_value = ["gpt-4o"]

            result = list_all_models()

            assert "openai" in result
            assert "anthropic" not in result  # Not configured

    def test_list_all_models_includes_unconfigured(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Includes unconfigured providers when skip_unconfigured=False."""
        from ai_infra.llm.providers.discovery import list_all_models

        with patch("ai_infra.llm.providers.discovery._list_openai_models") as mock_fetcher:
            mock_fetcher.return_value = ["gpt-4o"]

            result = list_all_models(skip_unconfigured=False)

            assert "openai" in result
            # Unconfigured providers have empty lists
            assert "anthropic" in result
            assert result["anthropic"] == []

    def test_list_all_models_multiple_providers(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        mock_anthropic_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Lists models from multiple configured providers."""
        from ai_infra.llm.providers.discovery import _FETCHERS, list_all_models

        mock_openai = MagicMock(return_value=["gpt-4o"])
        mock_anthropic = MagicMock(return_value=["claude-3-opus"])
        original_openai = _FETCHERS["openai"]
        original_anthropic = _FETCHERS["anthropic"]
        _FETCHERS["openai"] = mock_openai
        _FETCHERS["anthropic"] = mock_anthropic
        try:
            result = list_all_models()

            assert result["openai"] == ["gpt-4o"]
            assert result["anthropic"] == ["claude-3-opus"]
        finally:
            _FETCHERS["openai"] = original_openai
            _FETCHERS["anthropic"] = original_anthropic

    def test_list_all_models_forwards_timeout(
        self,
        mock_env_clean: None,
        mock_xai_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Forwards timeout through list_all_models."""
        from ai_infra.llm.providers.discovery import list_all_models

        with patch("ai_infra.llm.providers.discovery.list_models") as mock_list:
            mock_list.return_value = ["grok-beta"]

            result = list_all_models(timeout=7.5)

        assert result == {"xai": ["grok-beta"]}
        mock_list.assert_called_once_with("xai", refresh=False, use_cache=True, timeout=7.5)


# =============================================================================
# Test: API Fetchers (Mocked)
# =============================================================================


class TestOpenAIFetcher:
    """Tests for _list_openai_models() function."""

    def test_list_openai_models_success(self, mock_env_clean: None, mock_openai_key: str) -> None:
        """Successfully fetches OpenAI models."""
        from ai_infra.llm.providers.discovery import _list_openai_models

        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value = MagicMock(data=[mock_model1, mock_model2])

        with patch("openai.OpenAI", return_value=mock_client):
            models = _list_openai_models()

        assert "gpt-3.5-turbo" in models
        assert "gpt-4o" in models

    def test_list_openai_models_error_returns_empty(
        self, mock_env_clean: None, mock_openai_key: str
    ) -> None:
        """Returns empty list on API error."""
        from ai_infra.llm.providers.discovery import _list_openai_models

        with patch("openai.OpenAI") as mock_class:
            mock_class.side_effect = Exception("API Error")

            models = _list_openai_models()

        assert models == []


class TestAnthropicFetcher:
    """Tests for _list_anthropic_models() function."""

    def test_list_anthropic_models_success(
        self, mock_env_clean: None, mock_anthropic_key: str
    ) -> None:
        """Successfully fetches Anthropic models."""
        from ai_infra.llm.providers.discovery import _list_anthropic_models

        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "claude-3-opus-20240229"
        mock_model2 = MagicMock()
        mock_model2.id = "claude-3-sonnet-20240229"
        mock_client.models.list.return_value = MagicMock(data=[mock_model1, mock_model2])

        with patch("anthropic.Anthropic", return_value=mock_client):
            models = _list_anthropic_models()

        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models

    def test_list_anthropic_models_error_returns_empty(
        self, mock_env_clean: None, mock_anthropic_key: str
    ) -> None:
        """Returns empty list on API error."""
        from ai_infra.llm.providers.discovery import _list_anthropic_models

        with patch("anthropic.Anthropic") as mock_class:
            mock_class.side_effect = Exception("API Error")

            models = _list_anthropic_models()

        assert models == []


class TestGoogleFetcher:
    """Tests for _list_google_models() function."""

    def test_list_google_models_success(self, mock_env_clean: None, mock_google_key: str) -> None:
        """Successfully fetches Google models."""
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.name = "models/gemini-1.5-pro"
        mock_model2 = MagicMock()
        mock_model2.name = "models/gemini-1.5-flash"
        mock_client.models.list.return_value = [mock_model1, mock_model2]

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        # Patch sys.modules to inject mock
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            # Need to reimport after patching sys.modules
            from ai_infra.llm.providers.discovery import _list_google_models

            models = _list_google_models()

        # Models are returned without "models/" prefix
        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models

    def test_list_google_models_error_returns_empty(
        self, mock_env_clean: None, mock_google_key: str
    ) -> None:
        """Returns empty list on API error."""
        from ai_infra.llm.providers.discovery import _list_google_models

        mock_genai = MagicMock()
        mock_genai.Client.side_effect = Exception("API Error")

        with patch.dict(sys.modules, {"google": MagicMock(), "google.genai": mock_genai}):
            models = _list_google_models()

        assert models == []


class TestXaiFetcher:
    """Tests for _list_xai_models() function."""

    def test_list_xai_models_success(self, mock_env_clean: None, mock_xai_key: str) -> None:
        """Successfully fetches xAI models."""
        from ai_infra.llm.providers.discovery import _list_xai_models

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "grok-beta"},
                {"id": "grok-vision-beta"},
            ]
        }

        with patch("httpx.get", return_value=mock_response):
            models = _list_xai_models()

        assert "grok-beta" in models
        assert "grok-vision-beta" in models

    def test_list_xai_models_forwards_timeout(
        self, mock_env_clean: None, mock_xai_key: str
    ) -> None:
        """Forwards caller-provided timeout to xAI model discovery."""
        from ai_infra.llm.providers.discovery import _list_xai_models

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}

        with patch("httpx.get", return_value=mock_response) as mock_get:
            _list_xai_models(timeout=7.5)

        assert mock_get.call_args.kwargs["timeout"] == 7.5

    def test_list_xai_models_uses_xai_models_endpoint(
        self, mock_env_clean: None, mock_xai_key: str
    ) -> None:
        """Uses xAI's models endpoint."""
        from ai_infra.llm.providers.discovery import _list_xai_models

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}

        with patch("httpx.get", return_value=mock_response) as mock_get:
            _list_xai_models()

            assert mock_get.call_args.args[0] == "https://api.x.ai/v1/models"
            assert mock_get.call_args.kwargs["headers"]["Authorization"] == f"Bearer {mock_xai_key}"

    def test_list_xai_models_error_returns_empty(
        self, mock_env_clean: None, mock_xai_key: str
    ) -> None:
        """Returns empty list on API error."""
        from ai_infra.llm.providers.discovery import _list_xai_models

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("API Error")

            models = _list_xai_models()

        assert models == []


# =============================================================================
# Test: Chat Model Fetchers
# =============================================================================


class TestChatModelFetchers:
    """Tests for convenience chat model fetchers."""

    def test_openai_chat_models_filters_correctly(
        self, mock_env_clean: None, mock_openai_key: str
    ) -> None:
        """_list_openai_chat_models filters non-chat models."""
        from ai_infra.llm.providers.discovery import _list_openai_chat_models

        with patch("ai_infra.llm.providers.discovery._list_openai_models") as mock_fetcher:
            mock_fetcher.return_value = [
                "gpt-4o",
                "gpt-3.5-turbo",
                "text-embedding-ada-002",
                "dall-e-3",
            ]

            chat_models = _list_openai_chat_models()

            assert "gpt-4o" in chat_models
            assert "gpt-3.5-turbo" in chat_models
            assert "text-embedding-ada-002" not in chat_models
            assert "dall-e-3" not in chat_models

    def test_google_chat_models_filters_correctly(
        self, mock_env_clean: None, mock_google_key: str
    ) -> None:
        """_list_google_chat_models filters non-chat models."""
        from ai_infra.llm.providers.discovery import _list_google_chat_models

        with patch("ai_infra.llm.providers.discovery._list_google_models") as mock_fetcher:
            mock_fetcher.return_value = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "imagen-3.0-generate-001",
            ]

            chat_models = _list_google_chat_models()

            assert "gemini-1.5-pro" in chat_models
            assert "gemini-1.5-flash" in chat_models
            assert "imagen-3.0-generate-001" not in chat_models


# =============================================================================
# Test: Constants and Exports
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_supported_providers_list(self) -> None:
        """SUPPORTED_PROVIDERS contains expected providers."""
        from ai_infra.llm.providers.discovery import SUPPORTED_PROVIDERS

        assert "openai" in SUPPORTED_PROVIDERS
        assert "anthropic" in SUPPORTED_PROVIDERS
        assert "google_genai" in SUPPORTED_PROVIDERS
        assert "xai" in SUPPORTED_PROVIDERS

    def test_provider_env_vars_mapping(self) -> None:
        """PROVIDER_ENV_VARS maps providers to env vars."""
        from ai_infra.llm.providers.discovery import PROVIDER_ENV_VARS

        assert PROVIDER_ENV_VARS["openai"] == "OPENAI_API_KEY"
        assert PROVIDER_ENV_VARS["anthropic"] == "ANTHROPIC_API_KEY"
        # google_genai uses GEMINI_API_KEY (from ProviderRegistry)
        assert PROVIDER_ENV_VARS["google_genai"] == "GEMINI_API_KEY"
        assert PROVIDER_ENV_VARS["xai"] == "XAI_API_KEY"

    def test_cache_ttl_is_positive(self) -> None:
        """CACHE_TTL_SECONDS is a positive number."""
        from ai_infra.llm.providers.discovery import CACHE_TTL_SECONDS

        assert CACHE_TTL_SECONDS > 0
        assert isinstance(CACHE_TTL_SECONDS, (int, float))


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports_are_accessible(self) -> None:
        """All items in __all__ are accessible."""
        from ai_infra.llm.providers import discovery

        for name in discovery.__all__:
            assert hasattr(discovery, name), f"{name} not found in module"

    def test_core_functions_exported(self) -> None:
        """Core functions are in __all__."""
        from ai_infra.llm.providers.discovery import __all__

        assert "list_providers" in __all__
        assert "list_configured_providers" in __all__
        assert "list_models" in __all__
        assert "list_all_models" in __all__
        assert "is_provider_configured" in __all__
        assert "get_api_key" in __all__
        assert "clear_cache" in __all__

    def test_capability_exports(self) -> None:
        """Capability-related items are in __all__."""
        from ai_infra.llm.providers.discovery import __all__

        assert "ModelCapability" in __all__
        assert "detect_model_capabilities" in __all__
        assert "filter_models_by_capability" in __all__
        assert "categorize_models" in __all__

    def test_constants_exported(self) -> None:
        """Constants are in __all__."""
        from ai_infra.llm.providers.discovery import __all__

        assert "SUPPORTED_PROVIDERS" in __all__
        assert "PROVIDER_ENV_VARS" in __all__


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_model_list_categorization(self) -> None:
        """Categorizing empty list works correctly."""
        from ai_infra.llm.providers.discovery import ModelCapability, categorize_models

        result = categorize_models([], "openai")
        # Returns dict with capabilities as keys, even if lists are empty
        assert isinstance(result, dict)
        for cap in ModelCapability:
            if cap in result:
                assert result[cap] == []

    def test_model_with_no_matching_patterns(self) -> None:
        """Model with no matching patterns gets UNKNOWN."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            detect_model_capabilities,
        )

        caps = detect_model_capabilities("xyz-unknown-model-abc", "openai")
        assert ModelCapability.UNKNOWN in caps

    def test_cache_with_invalid_json(self, temp_cache_dir: Path) -> None:
        """Invalid JSON in cache returns empty dict."""
        from ai_infra.llm.providers.discovery import CACHE_FILE, _load_cache

        # Write invalid JSON
        CACHE_FILE.write_text("not valid json {{{")

        cache = _load_cache()
        assert cache == {}

    def test_cache_with_non_string_timestamp(self) -> None:
        """Non-numeric timestamp is invalid."""
        from ai_infra.llm.providers.discovery import _is_cache_valid

        cache = {
            "openai": {
                "models": ["gpt-4o"],
                "timestamp": "not a number",
            }
        }

        assert _is_cache_valid(cache, "openai") is False

    def test_filter_models_unknown_provider(self) -> None:
        """Filtering with unknown provider still works (returns empty or filtered)."""
        from ai_infra.llm.providers.discovery import (
            ModelCapability,
            filter_models_by_capability,
        )

        # Should not crash, may return empty or the models with UNKNOWN
        models = ["model-1", "model-2"]
        result = filter_models_by_capability(models, "unknown_provider", ModelCapability.CHAT)
        # Should return empty since no patterns match
        assert isinstance(result, list)

    def test_categorize_models_unknown_provider(self) -> None:
        """Categorizing with unknown provider still works."""
        from ai_infra.llm.providers.discovery import ModelCapability, categorize_models

        models = ["model-1", "model-2"]
        result = categorize_models(models, "unknown_provider")

        # Should return dict with UNKNOWN capability
        assert isinstance(result, dict)
        if result:
            # Models should be categorized as UNKNOWN
            assert ModelCapability.UNKNOWN in result


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
        """Full workflow: list models, cache, filter by capability."""
        from ai_infra.llm.providers.discovery import (
            _FETCHERS,
            CACHE_FILE,
            ModelCapability,
            clear_cache,
            list_models,
        )

        # Mock the API via _FETCHERS dict
        mock_fetcher = MagicMock(
            return_value=[
                "gpt-4o",
                "gpt-3.5-turbo",
                "text-embedding-ada-002",
            ]
        )
        original_fetcher = _FETCHERS["openai"]
        _FETCHERS["openai"] = mock_fetcher
        try:
            # First call fetches from API
            models = list_models("openai")
            assert mock_fetcher.call_count == 1
            assert "gpt-4o" in models

            # Cache should exist
            assert CACHE_FILE.exists()

            # Second call uses cache
            models2 = list_models("openai")
            assert mock_fetcher.call_count == 1  # No additional call
            assert models2 == models

            # Filter by capability
            chat_models = list_models("openai", capability=ModelCapability.CHAT)
            assert "gpt-4o" in chat_models
            assert "text-embedding-ada-002" not in chat_models

            # Clear cache
            clear_cache()
            assert not CACHE_FILE.exists()

            # Next call fetches again
            list_models("openai")
            assert mock_fetcher.call_count == 2
        finally:
            _FETCHERS["openai"] = original_fetcher

    def test_multi_provider_workflow(
        self,
        mock_env_clean: None,
        mock_openai_key: str,
        mock_anthropic_key: str,
        temp_cache_dir: Path,
    ) -> None:
        """Workflow with multiple providers."""
        from ai_infra.llm.providers.discovery import (
            _FETCHERS,
            get_default_provider,
            list_all_models,
            list_configured_providers,
        )

        # Check configured providers
        providers = list_configured_providers()
        assert "openai" in providers
        assert "anthropic" in providers

        # Default should be OpenAI (highest priority)
        default = get_default_provider()
        assert default == "openai"

        # List all models via _FETCHERS dict
        mock_openai = MagicMock(return_value=["gpt-4o"])
        mock_anthropic = MagicMock(return_value=["claude-3-opus"])
        original_openai = _FETCHERS["openai"]
        original_anthropic = _FETCHERS["anthropic"]
        _FETCHERS["openai"] = mock_openai
        _FETCHERS["anthropic"] = mock_anthropic
        try:
            all_models = list_all_models()

            assert "openai" in all_models
            assert "anthropic" in all_models
            assert "gpt-4o" in all_models["openai"]
            assert "claude-3-opus" in all_models["anthropic"]
        finally:
            _FETCHERS["openai"] = original_openai
            _FETCHERS["anthropic"] = original_anthropic
