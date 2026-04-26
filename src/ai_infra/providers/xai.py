"""xAI (Grok) provider configuration.

xAI supports:
- Chat: Grok-3, Grok-2, Grok-beta models
- ImageGen: Grok Imagine image generation and editing

xAI uses an OpenAI-compatible API for chat, and the official xAI SDK for image generation.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

XAI = ProviderConfig(
    name="xai",
    display_name="xAI (Grok)",
    env_var="XAI_API_KEY",
    base_url="https://api.x.ai/v1",
    capabilities={
        ProviderCapability.CHAT: CapabilityConfig(
            models=[
                "grok-3",
                "grok-3-mini",
                "grok-2",
                "grok-2-mini",
                "grok-beta",
            ],
            default_model="grok-code-fast-1",
            features=["streaming", "function_calling"],
            extra={
                "openai_compatible": True,
            },
        ),
        ProviderCapability.IMAGEGEN: CapabilityConfig(
            models=["grok-imagine-image"],
            default_model="grok-imagine-image",
            features=["edit", "multi_image_edit", "base64_output", "async"],
            extra={
                "aspect_ratios": [
                    "1:1",
                    "16:9",
                    "9:16",
                    "4:3",
                    "3:4",
                    "3:2",
                    "2:3",
                    "2:1",
                    "1:2",
                    "19.5:9",
                    "9:19.5",
                    "20:9",
                    "9:20",
                    "auto",
                ],
                "resolutions": ["1k", "2k"],
                "max_images": 10,
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(XAI)
