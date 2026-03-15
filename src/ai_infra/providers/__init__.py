"""Centralized provider registry for ai-infra.

This module provides a single source of truth for all provider configurations
including API keys, models, voices, and capabilities.

Example:
    >>> from ai_infra.providers import ProviderRegistry, ProviderCapability
    >>>
    >>> # List all providers
    >>> ProviderRegistry.list_all()
    ['anthropic', 'cohere', 'deepgram', 'elevenlabs', 'google_genai', ...]
    >>>
    >>> # List providers that support a capability
    >>> ProviderRegistry.list_for_capability(ProviderCapability.TTS)
    ['elevenlabs', 'google_genai', 'openai']
    >>>
    >>> # Get default provider for a capability
    >>> ProviderRegistry.get_default_for_capability(ProviderCapability.CHAT)
    'openai'  # If OPENAI_API_KEY is set
    >>>
    >>> # Get models for a provider
    >>> ProviderRegistry.get_models("openai", ProviderCapability.CHAT)
    ['gpt-4o', 'gpt-4o-mini', ...]
    >>>
    >>> # Get API key
    >>> ProviderRegistry.get_api_key("openai")
    'sk-...'

Convenience functions are also available at module level:
    >>> from ai_infra.providers import (
    ...     list_providers,
    ...     list_providers_for_capability,
    ...     is_provider_configured,
    ...     get_default_provider,
    ...     get_api_key,
    ... )
"""

# Import provider modules to register them
# These are imported for their side effects (calling ProviderRegistry.register)
from ai_infra.providers import (  # noqa: F401
    amazon,
    anthropic,
    cohere,
    deepgram,
    elevenlabs,
    google,
    openai,
    replicate,
    stability,
    voyage,
    xai,
)

# Base data structures
from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig

# Registry class and convenience functions
from ai_infra.providers.registry import (
    ProviderRegistry,
    get_api_key,
    get_default_provider,
    get_provider,
    get_provider_config,
    is_provider_configured,
    list_configured_providers,
    list_providers,
    list_providers_for_capability,
)

__all__ = [
    # Classes
    "ProviderRegistry",
    "ProviderConfig",
    "CapabilityConfig",
    "ProviderCapability",
    # Convenience functions
    "get_provider",
    "get_provider_config",
    "list_providers",
    "list_configured_providers",
    "list_providers_for_capability",
    "is_provider_configured",
    "get_default_provider",
    "get_api_key",
]
