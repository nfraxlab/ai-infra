"""Models and types for image generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ImageGenProvider(StrEnum):
    """Supported image generation providers."""

    OPENAI = "openai"
    GOOGLE = "google"
    XAI = "xai"
    STABILITY = "stability"
    REPLICATE = "replicate"


@dataclass
class GeneratedImage:
    """Represents a generated image.

    Attributes:
        data: Image data as bytes, or None if URL-only.
        url: URL to the image (for providers that return URLs).
        revised_prompt: The prompt as revised by the model (if applicable).
        model: The model used to generate the image.
        provider: The provider that generated the image.
        metadata: Additional provider-specific metadata.
    """

    data: bytes | None = None
    url: str | None = None
    revised_prompt: str | None = None
    model: str | None = None
    provider: ImageGenProvider | None = None
    metadata: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save the image to a file.

        Args:
            path: File path to save to.

        Raises:
            ValueError: If no image data is available.
        """
        if self.data is None:
            raise ValueError("No image data available. Try fetching from URL first.")

        with open(path, "wb") as f:
            f.write(self.data)

    async def fetch(self) -> bytes:
        """Fetch image data from URL if not already loaded.

        Returns:
            Image data as bytes.

        Raises:
            ValueError: If no URL is available.
        """
        if self.data is not None:
            return self.data

        if self.url is None:
            raise ValueError("No URL available to fetch image from.")

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(self.url)
            response.raise_for_status()
            self.data = response.content
            return self.data


# Default models per provider
DEFAULT_MODELS = {
    ImageGenProvider.OPENAI: "gpt-image-1.5",
    ImageGenProvider.GOOGLE: "imagen-4.0-fast-generate-001",  # Dedicated image generation API
    ImageGenProvider.XAI: "grok-imagine-image",
    ImageGenProvider.STABILITY: "stable-diffusion-xl-1024-v1-0",
    ImageGenProvider.REPLICATE: "black-forest-labs/flux-schnell",
}

# Available models per provider
AVAILABLE_MODELS = {
    ImageGenProvider.OPENAI: [
        "gpt-image-1.5",
        "gpt-image-1",
        "gpt-image-1-mini",
        "dall-e-2",
        "dall-e-3",
    ],
    ImageGenProvider.GOOGLE: [
        # Gemini multimodal image generation (uses generate_content API)
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-exp-image-generation",
        "gemini-3-pro-image-preview",  # Nano Banana Pro
        # Imagen models (uses generate_images API, requires billing)
        "imagen-4.0-ultra-generate-001",
        "imagen-3.0-generate-002",
        "imagen-3.0-generate-001",
        "imagen-3.0-fast-generate-001",
        "imagen-4.0-generate-001",
        "imagen-4.0-fast-generate-001",
    ],
    ImageGenProvider.XAI: ["grok-imagine-image"],
    ImageGenProvider.STABILITY: [
        "stable-diffusion-xl-1024-v1-0",
        "stable-diffusion-v1-6",
    ],
    ImageGenProvider.REPLICATE: [
        "black-forest-labs/flux-schnell",
        "black-forest-labs/flux-dev",
        "stability-ai/sdxl",
    ],
}

# Gemini models use generate_content API, Imagen models use generate_images API
GEMINI_IMAGE_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image",
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.5-flash-image-preview",
    "gemini-3-pro-image-preview",  # Nano Banana Pro
}
