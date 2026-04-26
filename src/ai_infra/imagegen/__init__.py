"""Image generation module with provider-agnostic API.

Supports multiple providers:
- OpenAI (GPT Image 1.5/1/mini, DALL-E 2/3)
- Google (Gemini 3.1 Flash Image Preview, Gemini 3 Pro Image Preview, Imagen 3/4)
- xAI (Grok Imagine image generation and editing)
- Stability AI (Stable Diffusion)
- Replicate (SDXL, Flux, etc.)

Example:
    ```python
    from ai_infra import ImageGen

    # Zero-config: auto-detects provider from env vars
    gen = ImageGen()

    # Generate an image
    images = gen.generate("A sunset over mountains")

    # With specific provider (default is imagen-4.0-fast-generate-001)
    gen = ImageGen(provider="google")
    images = gen.generate("A futuristic city", size="1024x1024", n=2)

    # List live models from the provider API
    from ai_infra.imagegen import list_models
    models = list_models("google")
    ```
"""

from ai_infra.imagegen.discovery import (
    PROVIDER_ENV_VARS,
    SUPPORTED_PROVIDERS,
    clear_cache,
    get_api_key,
    is_provider_configured,
    list_all_available_models,
    list_all_models,
    list_available_models,
    list_configured_providers,
    list_known_models,
    list_models,
    list_providers,
)
from ai_infra.imagegen.imagegen import ImageGen
from ai_infra.imagegen.models import GeneratedImage, ImageGenProvider

__all__ = [
    "ImageGen",
    "GeneratedImage",
    "ImageGenProvider",
    # Discovery
    "list_providers",
    "list_configured_providers",
    "list_known_models",
    "list_models",
    "list_available_models",
    "list_all_models",
    "list_all_available_models",
    "is_provider_configured",
    "get_api_key",
    "clear_cache",
    "SUPPORTED_PROVIDERS",
    "PROVIDER_ENV_VARS",
]
