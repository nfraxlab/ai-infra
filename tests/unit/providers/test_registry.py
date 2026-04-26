"""Tests for the Provider Registry.

Tests cover:
- Provider registration
- Capability lookup
- is_configured() with mocked env vars
- get_default_for_capability() with different env var combinations
"""

import os
from unittest.mock import patch

from ai_infra.providers import (
    ProviderCapability,
    ProviderRegistry,
    get_provider,
    is_provider_configured,
    list_providers,
    list_providers_for_capability,
)


class TestProviderRegistry:
    """Tests for ProviderRegistry class."""

    def test_list_all_providers(self):
        """Test listing all registered providers."""
        providers = ProviderRegistry.list_all()
        assert isinstance(providers, list)
        assert len(providers) > 0
        # Check known providers are registered
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google_genai" in providers

    def test_get_provider(self):
        """Test getting a specific provider config."""
        config = ProviderRegistry.get("openai")
        assert config is not None
        assert config.name == "openai"
        assert config.display_name == "OpenAI"
        assert config.env_var == "OPENAI_API_KEY"

    def test_get_nonexistent_provider(self):
        """Test getting a provider that doesn't exist."""
        config = ProviderRegistry.get("nonexistent_provider")
        assert config is None

    def test_list_for_capability_chat(self):
        """Test listing providers for CHAT capability."""
        providers = ProviderRegistry.list_for_capability(ProviderCapability.CHAT)
        assert isinstance(providers, list)
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google_genai" in providers
        assert "xai" in providers

    def test_list_for_capability_embeddings(self):
        """Test listing providers for EMBEDDINGS capability."""
        providers = ProviderRegistry.list_for_capability(ProviderCapability.EMBEDDINGS)
        assert "openai" in providers
        assert "voyage" in providers
        assert "cohere" in providers
        assert "google_genai" in providers

    def test_list_for_capability_tts(self):
        """Test listing providers for TTS capability."""
        providers = ProviderRegistry.list_for_capability(ProviderCapability.TTS)
        assert "openai" in providers
        assert "elevenlabs" in providers
        assert "google_genai" in providers

    def test_list_for_capability_stt(self):
        """Test listing providers for STT capability."""
        providers = ProviderRegistry.list_for_capability(ProviderCapability.STT)
        assert "openai" in providers
        assert "deepgram" in providers
        assert "google_genai" in providers

    def test_list_for_capability_imagegen(self):
        """Test listing providers for IMAGEGEN capability."""
        providers = ProviderRegistry.list_for_capability(ProviderCapability.IMAGEGEN)
        assert "openai" in providers
        assert "google_genai" in providers
        assert "xai" in providers
        assert "stability" in providers
        assert "replicate" in providers

    def test_list_for_capability_realtime(self):
        """Test listing providers for REALTIME capability."""
        providers = ProviderRegistry.list_for_capability(ProviderCapability.REALTIME)
        assert "openai" in providers
        assert "google_genai" in providers

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_is_configured_with_key(self):
        """Test is_configured returns True when env var is set."""
        assert ProviderRegistry.is_configured("openai") is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_configured_without_key(self):
        """Test is_configured returns False when env var is not set."""
        # Clear any existing env vars
        for key in list(os.environ.keys()):
            if "API_KEY" in key or "API_TOKEN" in key:
                del os.environ[key]
        assert ProviderRegistry.is_configured("openai") is False

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    def test_is_configured_with_alt_env_var(self):
        """Test is_configured checks alternate env vars."""
        # Google has alt_env_vars including GEMINI_API_KEY
        assert ProviderRegistry.is_configured("google_genai") is True

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_get_api_key(self):
        """Test get_api_key returns the key."""
        key = ProviderRegistry.get_api_key("openai")
        assert key == "test-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_not_set(self):
        """Test get_api_key returns None when not set."""
        key = ProviderRegistry.get_api_key("openai")
        assert key is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_get_default_for_capability(self):
        """Test get_default_for_capability returns first configured provider."""
        # Clear other keys to ensure openai is selected
        provider = ProviderRegistry.get_default_for_capability(ProviderCapability.CHAT)
        assert provider == "openai"


class TestProviderConfig:
    """Tests for individual provider configurations."""

    def test_openai_config(self):
        """Test OpenAI provider config."""
        config = ProviderRegistry.get("openai")
        assert config is not None

        # Check CHAT capability
        chat = config.get_capability(ProviderCapability.CHAT)
        assert chat is not None
        assert "gpt-4o" in chat.models
        assert "gpt-4o-mini" in chat.models
        assert chat.default_model is not None

        # Check TTS capability
        tts = config.get_capability(ProviderCapability.TTS)
        assert tts is not None
        assert "tts-1" in tts.models
        assert "alloy" in tts.voices
        assert tts.default_voice == "alloy"

        # Check EMBEDDINGS capability
        emb = config.get_capability(ProviderCapability.EMBEDDINGS)
        assert emb is not None
        assert "text-embedding-3-small" in emb.models

    def test_anthropic_config(self):
        """Test Anthropic provider config."""
        config = ProviderRegistry.get("anthropic")
        assert config is not None
        assert config.env_var == "ANTHROPIC_API_KEY"

        # Check CHAT capability
        chat = config.get_capability(ProviderCapability.CHAT)
        assert chat is not None
        assert "claude-sonnet-4-20250514" in chat.models or any("claude" in m for m in chat.models)

    def test_google_config(self):
        """Test Google/Gemini provider config."""
        config = ProviderRegistry.get("google_genai")
        assert config is not None
        assert config.env_var == "GEMINI_API_KEY"
        assert "GOOGLE_API_KEY" in config.alt_env_vars

        # Check multiple capabilities
        assert config.has_capability(ProviderCapability.CHAT)
        assert config.has_capability(ProviderCapability.EMBEDDINGS)
        assert config.has_capability(ProviderCapability.TTS)
        assert config.has_capability(ProviderCapability.STT)
        assert config.has_capability(ProviderCapability.IMAGEGEN)
        assert config.has_capability(ProviderCapability.REALTIME)

    def test_xai_config(self):
        """Test xAI provider config."""
        config = ProviderRegistry.get("xai")
        assert config is not None
        assert config.env_var == "XAI_API_KEY"
        assert config.has_capability(ProviderCapability.CHAT)
        assert config.has_capability(ProviderCapability.IMAGEGEN)

        imagegen = config.get_capability(ProviderCapability.IMAGEGEN)
        assert imagegen is not None
        assert "grok-imagine-image" in imagegen.models

    def test_elevenlabs_config(self):
        """Test ElevenLabs provider config (TTS only)."""
        config = ProviderRegistry.get("elevenlabs")
        assert config is not None

        # Should only have TTS
        assert config.has_capability(ProviderCapability.TTS)
        assert not config.has_capability(ProviderCapability.STT)
        assert not config.has_capability(ProviderCapability.CHAT)

    def test_deepgram_config(self):
        """Test Deepgram provider config (STT only)."""
        config = ProviderRegistry.get("deepgram")
        assert config is not None

        # Should only have STT
        assert config.has_capability(ProviderCapability.STT)
        assert not config.has_capability(ProviderCapability.TTS)
        assert not config.has_capability(ProviderCapability.CHAT)

    def test_all_providers_have_display_name(self):
        """Test all providers have a display name."""
        for name in ProviderRegistry.list_all():
            config = ProviderRegistry.get(name)
            assert config is not None
            assert config.display_name, f"Provider {name} missing display_name"

    def test_all_providers_have_env_var(self):
        """Test all providers have an env_var."""
        for name in ProviderRegistry.list_all():
            config = ProviderRegistry.get(name)
            assert config is not None
            assert config.env_var, f"Provider {name} missing env_var"

    def test_no_duplicate_provider_names(self):
        """Test there are no duplicate provider names."""
        providers = ProviderRegistry.list_all()
        assert len(providers) == len(set(providers))


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_provider_function(self):
        """Test get_provider convenience function."""
        config = get_provider("openai")
        assert config is not None
        assert config.name == "openai"

    def test_list_providers_function(self):
        """Test list_providers convenience function."""
        providers = list_providers()
        assert isinstance(providers, list)
        assert "openai" in providers

    def test_list_providers_for_capability_function(self):
        """Test list_providers_for_capability convenience function."""
        providers = list_providers_for_capability(ProviderCapability.CHAT)
        assert "openai" in providers

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False)
    def test_is_provider_configured_function(self):
        """Test is_provider_configured convenience function."""
        assert is_provider_configured("openai") is True


class TestCapabilityConfig:
    """Tests for CapabilityConfig dataclass."""

    def test_capability_with_all_fields(self):
        """Test capability config with all fields."""
        config = ProviderRegistry.get("openai")
        tts = config.get_capability(ProviderCapability.TTS)

        assert tts.models is not None
        assert tts.default_model is not None
        assert tts.voices is not None
        assert tts.default_voice is not None
        assert tts.features is not None

    def test_capability_with_extra_data(self):
        """Test capability config with extra data."""
        config = ProviderRegistry.get("openai")
        emb = config.get_capability(ProviderCapability.EMBEDDINGS)

        # OpenAI embeddings should have dimensions in extra
        assert emb.extra is not None
        assert "dimensions" in emb.extra

    def test_provider_get_models(self):
        """Test ProviderConfig.get_models helper."""
        config = ProviderRegistry.get("openai")
        models = config.get_models(ProviderCapability.CHAT)
        assert isinstance(models, list)
        assert "gpt-4o" in models

    def test_provider_get_voices(self):
        """Test ProviderConfig.get_voices helper."""
        config = ProviderRegistry.get("openai")
        voices = config.get_voices(ProviderCapability.TTS)
        assert isinstance(voices, list)
        assert "alloy" in voices
