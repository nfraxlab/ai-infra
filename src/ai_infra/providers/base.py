"""Base data structures for provider configuration.

This module defines the core data structures used by the provider registry:
- ProviderCapability: Enum of capabilities a provider can support
- CapabilityConfig: Configuration for a specific capability
- ProviderConfig: Complete configuration for a provider
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProviderCapability(StrEnum):
    """Capabilities that a provider can support.

    Each capability represents a distinct AI service that providers may offer.
    Not all providers support all capabilities.

    Example:
        >>> from ai_infra.providers import ProviderCapability
        >>> ProviderCapability.CHAT
        'chat'
        >>> ProviderCapability.TTS
        'tts'
    """

    CHAT = "chat"  # LLM chat completions
    EMBEDDINGS = "embeddings"  # Text embeddings
    MULTIMODAL_EMBEDDINGS = "multimodal_embeddings"  # Text + image embeddings
    TTS = "tts"  # Text-to-speech
    STT = "stt"  # Speech-to-text
    IMAGEGEN = "imagegen"  # Image generation
    REALTIME = "realtime"  # Realtime voice API (WebSocket-based)


@dataclass
class CapabilityConfig:
    """Configuration for a specific capability within a provider.

    Attributes:
        models: List of available model names for this capability.
        default_model: The default model to use if none specified.
        voices: List of available voice names (for TTS/Realtime).
        default_voice: The default voice to use if none specified.
        features: List of supported features (e.g., ["streaming", "timestamps"]).
        extra: Provider-specific configuration that doesn't fit elsewhere.

    Example:
        >>> config = CapabilityConfig(
        ...     models=["gpt-4o", "gpt-4o-mini"],
        ...     default_model="gpt-4o-mini",
        ...     features=["streaming", "function_calling"],
        ... )
    """

    models: list[str] = field(default_factory=list)
    default_model: str | None = None
    voices: list[str] = field(default_factory=list)
    default_voice: str | None = None
    features: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default_model to first model if not specified."""
        if self.default_model is None and self.models:
            self.default_model = self.models[0]
        if self.default_voice is None and self.voices:
            self.default_voice = self.voices[0]


@dataclass
class ProviderConfig:
    """Complete configuration for an AI provider.

    This is the central configuration object that contains all information
    about a provider's capabilities, authentication, and settings.

    Attributes:
        name: Internal provider name (e.g., "openai", "anthropic").
        display_name: Human-readable name (e.g., "OpenAI", "Anthropic").
        env_var: Primary environment variable for API key.
        alt_env_vars: Alternative environment variables to check.
        capabilities: Dict mapping capability to its configuration.
        base_url: Optional custom base URL for API requests.

    Example:
        >>> from ai_infra.providers import ProviderConfig, ProviderCapability, CapabilityConfig
        >>> config = ProviderConfig(
        ...     name="openai",
        ...     display_name="OpenAI",
        ...     env_var="OPENAI_API_KEY",
        ...     capabilities={
        ...         ProviderCapability.CHAT: CapabilityConfig(
        ...             models=["gpt-4o", "gpt-4o-mini"],
        ...             default_model="gpt-4o-mini",
        ...         ),
        ...     },
        ... )
        >>> config.has_capability(ProviderCapability.CHAT)
        True
    """

    name: str
    display_name: str
    env_var: str
    alt_env_vars: list[str] = field(default_factory=list)
    capabilities: dict[ProviderCapability, CapabilityConfig] = field(default_factory=dict)
    base_url: str | None = None

    def has_capability(self, cap: ProviderCapability) -> bool:
        """Check if this provider supports a given capability.

        Args:
            cap: The capability to check for.

        Returns:
            True if the provider has this capability configured.
        """
        return cap in self.capabilities

    def get_capability(self, cap: ProviderCapability) -> CapabilityConfig | None:
        """Get the configuration for a specific capability.

        Args:
            cap: The capability to get configuration for.

        Returns:
            CapabilityConfig if the capability is supported, None otherwise.
        """
        return self.capabilities.get(cap)

    def get_models(self, cap: ProviderCapability) -> list[str]:
        """Get list of models for a capability.

        Args:
            cap: The capability to get models for.

        Returns:
            List of model names, empty if capability not supported.
        """
        config = self.get_capability(cap)
        return config.models if config else []

    def get_default_model(self, cap: ProviderCapability) -> str | None:
        """Get the default model for a capability.

        Args:
            cap: The capability to get the default model for.

        Returns:
            Default model name, None if capability not supported.
        """
        config = self.get_capability(cap)
        return config.default_model if config else None

    def get_voices(self, cap: ProviderCapability) -> list[str]:
        """Get list of voices for a capability (TTS/Realtime).

        Args:
            cap: The capability to get voices for.

        Returns:
            List of voice names, empty if capability not supported or has no voices.
        """
        config = self.get_capability(cap)
        return config.voices if config else []

    def get_default_voice(self, cap: ProviderCapability) -> str | None:
        """Get the default voice for a capability.

        Args:
            cap: The capability to get the default voice for.

        Returns:
            Default voice name, None if capability not supported or has no voices.
        """
        config = self.get_capability(cap)
        return config.default_voice if config else None
