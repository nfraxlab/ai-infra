"""Main ImageGen class with provider-agnostic API."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import math
import time
from typing import Any, BinaryIO, Literal

from ai_infra.imagegen.models import (
    DEFAULT_MODELS,
    GeneratedImage,
    ImageGenProvider,
)
from ai_infra.providers import ProviderCapability, ProviderRegistry

# Provider aliases for backwards compatibility
_PROVIDER_ALIASES = {"google": "google_genai"}
_REVERSE_ALIASES = {"google_genai": "google"}
_GOOGLE_IMAGE_REQUEST_TIMEOUT_MS = 180_000
_OPENAI_IMAGE_RETRY_ATTEMPTS = 3
_OPENAI_IMAGE_RETRY_DELAY_SECONDS = 1.0
_IMAGE_MAGIC_HEADERS: tuple[bytes, ...] = (
    b"\xff\xd8\xff",
    b"\x89PNG\r\n\x1a\n",
    b"GIF87a",
    b"GIF89a",
    b"RIFF",
    b"\x00\x00\x00\x0cjP  ",
)
_GOOGLE_GEMINI_SUPPORTED_ASPECT_RATIOS = {
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
}
_GOOGLE_GEMINI_SUPPORTED_IMAGE_SIZES = {"512", "1K", "2K", "4K"}
_GOOGLE_IMAGEN_SUPPORTED_ASPECT_RATIOS = {"1:1", "3:4", "4:3", "9:16", "16:9"}
_GOOGLE_IMAGEN_SUPPORTED_IMAGE_SIZES = {"1K", "2K"}
_XAI_SUPPORTED_ASPECT_RATIOS = {
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
}


def _looks_like_image_bytes(data: bytes) -> bool:
    """Return whether the payload already looks like binary image data."""
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return True
    return any(data.startswith(header) for header in _IMAGE_MAGIC_HEADERS)


def _normalize_provider_image_bytes(image_bytes: Any) -> bytes | None:
    """Normalize provider image payloads that may be raw bytes or base64 text."""
    if image_bytes is None:
        return None

    raw: bytes
    if isinstance(image_bytes, memoryview):
        raw = image_bytes.tobytes()
    elif isinstance(image_bytes, bytearray):
        raw = bytes(image_bytes)
    elif isinstance(image_bytes, bytes):
        raw = image_bytes
    elif isinstance(image_bytes, str):
        payload = image_bytes.strip()
        if payload.startswith("data:"):
            _, _, payload = payload.partition(",")
        raw = payload.encode("ascii")
    else:
        try:
            raw = bytes(image_bytes)
        except (TypeError, ValueError):
            return None

    stripped = raw.strip()
    if _looks_like_image_bytes(stripped):
        return stripped

    try:
        decoded = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError):
        return stripped

    return decoded or stripped


def _is_google_negative_prompt_unsupported_error(exc: Exception) -> bool:
    """Return True when the Gemini API rejects Imagen negative prompts."""
    message = str(exc).lower()
    return "negative_prompt" in message and "not supported" in message and "gemini api" in message


def _safe_optional_attr(obj: Any, name: str) -> Any:
    """Return an optional response attribute without propagating provider property errors."""
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _parse_size(size: str) -> tuple[int, int] | None:
    """Parse an image size string like "1024x1024"."""
    try:
        width_text, height_text = size.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except (AttributeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    return width, height


def _aspect_ratio_from_size(
    size: str,
    *,
    supported_ratios: set[str],
) -> str | None:
    """Return a provider-supported aspect ratio derived from a WxH size."""
    parsed = _parse_size(size)
    if parsed is None:
        return None

    width, height = parsed
    divisor = math.gcd(width, height)
    aspect_ratio = f"{width // divisor}:{height // divisor}"
    return aspect_ratio if aspect_ratio in supported_ratios else None


def _resolution_from_size(
    size: str,
    *,
    supported_sizes: set[str],
    lowercase: bool = False,
) -> str | None:
    """Return the closest provider resolution label for a WxH size."""
    parsed = _parse_size(size)
    if parsed is None:
        return None

    max_dimension = max(parsed)
    ordered_sizes = [
        (512, "512"),
        (1024, "1K"),
        (2048, "2K"),
        (4096, "4K"),
    ]
    for threshold, label in ordered_sizes:
        if max_dimension <= threshold and label in supported_sizes:
            return label.lower() if lowercase else label

    for label in ("4K", "2K", "1K", "512"):
        if label in supported_sizes:
            return label.lower() if lowercase else label

    return None


def _normalize_kwargs_aliases(kwargs: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    """Map external or camelCase kwargs to the provider SDK's preferred names."""
    normalized = dict(kwargs)
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized.pop(source)
    return normalized


def _infer_image_mime_type(data: bytes) -> str:
    """Infer a MIME type from image bytes."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"\x00\x00\x00\x0cjP  "):
        return "image/jp2"
    return "image/png"


def _load_image_input_data(image: str | bytes) -> bytes:
    """Load image bytes from a file path or in-memory bytes."""
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return image_file.read()
    return image


def _image_input_to_data_uri(image: str | bytes) -> str:
    """Convert an image path or bytes into a base64 data URI."""
    image_data = _load_image_input_data(image)
    mime_type = _infer_image_mime_type(image_data)
    payload = base64.b64encode(image_data).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _iter_google_response_parts(response: Any) -> list[Any]:
    """Return content parts from Google GenAI responses across SDK versions."""
    parts = getattr(response, "parts", None)
    if parts is not None:
        return list(parts)

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None)
    return list(parts) if parts is not None else []


def _build_google_generated_images(
    response: Any,
    *,
    model: str | None,
) -> list[GeneratedImage]:
    """Extract final image outputs from Google content generation responses."""
    results: list[GeneratedImage] = []
    for part in _iter_google_response_parts(response):
        if getattr(part, "thought", False):
            continue

        inline_data = getattr(part, "inline_data", None)
        if inline_data is None:
            continue

        results.append(
            GeneratedImage(
                data=_normalize_provider_image_bytes(getattr(inline_data, "data", None)),
                model=model,
                provider=ImageGenProvider.GOOGLE,
            )
        )

    return results


def _build_google_image_part(image: str | bytes) -> Any:
    """Build a Google GenAI image input part from bytes or a local file."""
    from google.genai import types

    image_data = _load_image_input_data(image)
    mime_type = _infer_image_mime_type(image_data)

    part_class = getattr(types, "Part", None)
    part_from_bytes = getattr(part_class, "from_bytes", None) if part_class is not None else None
    if callable(part_from_bytes):
        return part_from_bytes(data=image_data, mime_type=mime_type)

    return {
        "inline_data": {
            "data": base64.b64encode(image_data).decode("ascii"),
            "mime_type": mime_type,
        }
    }


def _is_retryable_openai_image_error(exc: Exception) -> bool:
    """Return whether an OpenAI image request error is worth retrying."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code in {408, 409, 429} or status_code >= 500):
        return True

    if exc.__class__.__name__ in {
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
    }:
        return True

    message = str(exc).lower()
    return (
        "timed out" in message
        or "timeout" in message
        or "did not complete within" in message
        or "temporarily unavailable" in message
        or "server_error" in message
        or "internal server error" in message
        or "rate limit" in message
        or "rate_limit" in message
    )


def _is_openai_gpt_image_model(model: str | None) -> bool:
    """Return whether the OpenAI image model uses GPT Image response semantics."""
    return bool(model and model.startswith("gpt-image-"))


class ImageGen:
    """Provider-agnostic image generation.

    Supports OpenAI, Google, xAI, Stability AI, and Replicate.
    Auto-detects provider from environment variables if not specified.

    Example:
        ```python
        # Zero-config: auto-detects from env vars
        gen = ImageGen()
        images = gen.generate("A sunset over mountains")

        # Explicit provider and model
        gen = ImageGen(provider="google", model="imagen-4.0-generate-001")
        images = gen.generate("A futuristic city", n=2)

        # Save generated image
        images[0].save("output.png")
        ```

    Environment Variables:
        - OPENAI_API_KEY: For OpenAI GPT Image / DALL-E
        - GOOGLE_API_KEY or GEMINI_API_KEY: For Google Gemini / Imagen
        - XAI_API_KEY: For xAI Grok Imagine
        - STABILITY_API_KEY: For Stability AI
        - REPLICATE_API_TOKEN: For Replicate
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ImageGen.

        Args:
            provider: Provider name ("openai", "google", "stability", "replicate").
                     Auto-detected from env vars if not specified.
            model: Model name. Uses provider default if not specified.
            api_key: API key. Uses env var if not specified.
            **kwargs: Additional provider-specific options.
        """
        self._provider, self._api_key = self._resolve_provider_and_key(provider, api_key)
        self._model = model or self._get_default_model(self._provider)
        self._kwargs = kwargs
        self._client: Any = None

    @property
    def provider(self) -> ImageGenProvider:
        """Get the current provider."""
        return self._provider

    @property
    def model(self) -> str | None:
        """Get the current model."""
        return self._model

    def _get_default_model(self, provider: ImageGenProvider) -> str | None:
        """Get default model for provider from registry."""
        # Map provider enum to registry name
        registry_name = provider.value
        if registry_name == "google":
            registry_name = "google_genai"

        config = ProviderRegistry.get(registry_name)
        if config:
            cap = config.get_capability(ProviderCapability.IMAGEGEN)
            if cap and cap.default_model:
                return cap.default_model

        # Fallback to legacy constant
        return DEFAULT_MODELS.get(provider)

    def _resolve_provider_and_key(
        self,
        provider: str | None,
        api_key: str | None,
    ) -> tuple[ImageGenProvider, str]:
        """Resolve provider and API key from args or environment."""

        if provider is not None:
            # Explicit provider
            normalized_provider = _REVERSE_ALIASES.get(provider.lower(), provider.lower())
            p = ImageGenProvider(normalized_provider)
            key = api_key or self._get_env_key(p)
            if not key:
                raise ValueError(f"No API key found for provider '{provider}'")
            return p, key

        # Auto-detect from registry
        # Provider priority order
        priority = ["openai", "google_genai", "xai", "stability", "replicate"]
        for name in priority:
            if ProviderRegistry.is_configured(name):
                # Map to ImageGenProvider enum
                enum_name = _REVERSE_ALIASES.get(name, name)
                key = ProviderRegistry.get_api_key(name)
                if key:
                    return ImageGenProvider(enum_name), key

        raise ValueError(
            "No API key found. Set one of: OPENAI_API_KEY, GOOGLE_API_KEY, "
            "XAI_API_KEY, STABILITY_API_KEY, or REPLICATE_API_TOKEN"
        )

    def _get_env_key(self, provider: ImageGenProvider) -> str | None:
        """Get the environment variable key for a provider."""
        # Map to registry name
        registry_name = provider.value
        if registry_name == "google":
            registry_name = "google_genai"

        return ProviderRegistry.get_api_key(registry_name)

    def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        quality: Literal["standard", "hd", "low", "medium", "high", "auto"] = "standard",
        style: Literal["vivid", "natural"] | None = None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            size: Image size (e.g., "1024x1024", "1792x1024").
            n: Number of images to generate.
            quality: Image quality ("standard" or "hd"). OpenAI only.
            style: Image style ("vivid" or "natural"). OpenAI DALL-E 3 only.
            **kwargs: Additional provider-specific options.

        Returns:
            List of GeneratedImage objects.

        Example:
            ```python
            images = gen.generate(
                "A cat wearing a hat",
                size="1024x1024",
                n=2,
                quality="hd",
            )
            for img in images:
                print(img.url)
            ```
        """
        if self._provider == ImageGenProvider.OPENAI:
            return self._generate_openai(
                prompt, size=size, n=n, quality=quality, style=style, **kwargs
            )
        elif self._provider == ImageGenProvider.GOOGLE:
            return self._generate_google(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.XAI:
            return self._generate_xai(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.STABILITY:
            return self._generate_stability(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.REPLICATE:
            return self._generate_replicate(prompt, size=size, n=n, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    async def agenerate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        quality: Literal["standard", "hd", "low", "medium", "high", "auto"] = "standard",
        style: Literal["vivid", "natural"] | None = None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async version of generate().

        Args:
            prompt: Text description of the image to generate.
            size: Image size (e.g., "1024x1024", "1792x1024").
            n: Number of images to generate.
            quality: Image quality ("standard" or "hd"). OpenAI only.
            style: Image style ("vivid" or "natural"). OpenAI DALL-E 3 only.
            **kwargs: Additional provider-specific options.

        Returns:
            List of GeneratedImage objects.
        """
        if self._provider == ImageGenProvider.OPENAI:
            return await self._agenerate_openai(
                prompt, size=size, n=n, quality=quality, style=style, **kwargs
            )
        elif self._provider == ImageGenProvider.GOOGLE:
            return await self._agenerate_google(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.XAI:
            return await self._agenerate_xai(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.STABILITY:
            return await self._agenerate_stability(prompt, size=size, n=n, **kwargs)
        elif self._provider == ImageGenProvider.REPLICATE:
            return await self._agenerate_replicate(prompt, size=size, n=n, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    # -------------------------------------------------------------------------
    # OpenAI Implementation
    # -------------------------------------------------------------------------

    def _get_openai_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _run_openai_image_operation(self, operation: Any) -> Any:
        """Retry transient OpenAI image operation failures."""
        attempts = max(
            int(self._kwargs.get("openai_image_retry_attempts", _OPENAI_IMAGE_RETRY_ATTEMPTS)), 1
        )
        delay = max(
            float(
                self._kwargs.get(
                    "openai_image_retry_delay_seconds", _OPENAI_IMAGE_RETRY_DELAY_SECONDS
                )
            ),
            0.0,
        )

        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_openai_image_error(exc):
                    raise
                time.sleep(delay * attempt)

        raise RuntimeError("OpenAI image operation exhausted retries")

    async def _arun_openai_image_operation(self, operation: Any) -> Any:
        """Retry transient async OpenAI image operation failures."""
        attempts = max(
            int(self._kwargs.get("openai_image_retry_attempts", _OPENAI_IMAGE_RETRY_ATTEMPTS)), 1
        )
        delay = max(
            float(
                self._kwargs.get(
                    "openai_image_retry_delay_seconds", _OPENAI_IMAGE_RETRY_DELAY_SECONDS
                )
            ),
            0.0,
        )

        for attempt in range(1, attempts + 1):
            try:
                return await operation()
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_openai_image_error(exc):
                    raise
                await asyncio.sleep(delay * attempt)

        raise RuntimeError("OpenAI image operation exhausted retries")

    def _openai_generate_quality(self, quality: str) -> str | None:
        """Normalize quality values across legacy DALL-E and GPT image models."""
        if _is_openai_gpt_image_model(self._model):
            if quality == "hd":
                return "high"
            if quality in {"low", "medium", "high", "auto"}:
                return quality
            return None
        return quality

    def _openai_image_response_format(self) -> str | None:
        """Get the appropriate OpenAI image response format for the current model."""
        return None if _is_openai_gpt_image_model(self._model) else "url"

    def _build_openai_generated_images(
        self,
        data: list[Any] | None,
        *,
        model: str | None = None,
    ) -> list[GeneratedImage]:
        """Convert OpenAI image API results into GeneratedImage objects."""
        results: list[GeneratedImage] = []
        resolved_model = model or self._model
        for img in data or []:
            image_data: bytes | None = None
            b64_json = getattr(img, "b64_json", None)
            if isinstance(b64_json, (str, bytes)):
                image_data = base64.b64decode(b64_json)
            results.append(
                GeneratedImage(
                    data=image_data,
                    url=getattr(img, "url", None),
                    revised_prompt=getattr(img, "revised_prompt", None),
                    model=resolved_model,
                    provider=ImageGenProvider.OPENAI,
                )
            )
        return results

    def _generate_openai(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        quality: str,
        style: str | None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using OpenAI DALL-E."""
        client = self._get_openai_client()

        params: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "size": size,
            "n": n,
        }
        response_format = self._openai_image_response_format()
        if response_format is not None:
            params["response_format"] = response_format

        # DALL-E 3 specific options
        if self._model == "dall-e-3":
            params["quality"] = quality
            if style:
                params["style"] = style
        else:
            normalized_quality = self._openai_generate_quality(quality)
            if normalized_quality is not None:
                params["quality"] = normalized_quality

        params.update(kwargs)

        response = self._run_openai_image_operation(lambda: client.images.generate(**params))

        return self._build_openai_generated_images(response.data)

    async def _agenerate_openai(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        quality: str,
        style: str | None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using OpenAI DALL-E."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key)

        params: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "size": size,
            "n": n,
        }
        response_format = self._openai_image_response_format()
        if response_format is not None:
            params["response_format"] = response_format

        if self._model == "dall-e-3":
            params["quality"] = quality
            if style:
                params["style"] = style
        else:
            normalized_quality = self._openai_generate_quality(quality)
            if normalized_quality is not None:
                params["quality"] = normalized_quality

        params.update(kwargs)

        response = await self._arun_openai_image_operation(lambda: client.images.generate(**params))

        return self._build_openai_generated_images(response.data)

    # -------------------------------------------------------------------------
    # Google Implementation
    # -------------------------------------------------------------------------

    def _build_google_generate_content_config(self, size: str, **kwargs: Any) -> Any:
        """Build GenerateContentConfig for Google Gemini image models."""
        from google.genai import types

        config_kwargs = _normalize_kwargs_aliases(
            dict(kwargs),
            {
                "aspectRatio": "aspect_ratio",
                "imageSize": "image_size",
                "responseModalities": "response_modalities",
            },
        )
        response_modalities = config_kwargs.pop("response_modalities", ["IMAGE"])
        image_config = config_kwargs.pop("image_config", None)
        image_config_type = getattr(types, "ImageConfig", None)
        generate_content_config_type = types.GenerateContentConfig

        if isinstance(image_config, dict) and image_config_type is not None:
            image_config = image_config_type(**image_config)

        if image_config is None:
            aspect_ratio = config_kwargs.pop("aspect_ratio", None)
            if aspect_ratio is None:
                aspect_ratio = _aspect_ratio_from_size(
                    size,
                    supported_ratios=_GOOGLE_GEMINI_SUPPORTED_ASPECT_RATIOS,
                )

            image_size = config_kwargs.pop("image_size", None)
            if image_size is None:
                image_size = config_kwargs.pop("resolution", None)
            if image_size is None and (self._model or "").startswith("gemini-3"):
                image_size = _resolution_from_size(
                    size,
                    supported_sizes=_GOOGLE_GEMINI_SUPPORTED_IMAGE_SIZES,
                )

            if aspect_ratio or image_size:
                image_config_kwargs: dict[str, Any] = {}
                if aspect_ratio:
                    image_config_kwargs["aspect_ratio"] = aspect_ratio
                if image_size:
                    image_config_kwargs["image_size"] = image_size
                if image_config_type is not None:
                    image_config = image_config_type(**image_config_kwargs)
                else:
                    image_config = image_config_kwargs

        final_kwargs: dict[str, Any] = {}
        if response_modalities is not None:
            final_kwargs["response_modalities"] = response_modalities
        if image_config is not None:
            final_kwargs["image_config"] = image_config
        final_kwargs.update(config_kwargs)
        return generate_content_config_type(**final_kwargs)

    def _build_google_generate_images_config(self, size: str, n: int, **kwargs: Any) -> Any:
        """Build GenerateImagesConfig for Google Imagen models."""
        from google.genai import types

        config_kwargs = _normalize_kwargs_aliases(
            dict(kwargs),
            {
                "numberOfImages": "number_of_images",
                "outputMimeType": "output_mime_type",
                "aspectRatio": "aspect_ratio",
                "imageSize": "image_size",
                "personGeneration": "person_generation",
            },
        )
        config_kwargs.setdefault("number_of_images", n)
        config_kwargs.setdefault("output_mime_type", "image/png")

        if "aspect_ratio" not in config_kwargs:
            aspect_ratio = _aspect_ratio_from_size(
                size,
                supported_ratios=_GOOGLE_IMAGEN_SUPPORTED_ASPECT_RATIOS,
            )
            if aspect_ratio is not None:
                config_kwargs["aspect_ratio"] = aspect_ratio

        generate_images_config_type = getattr(types, "GenerateImagesConfig", None)
        if generate_images_config_type is None:
            return config_kwargs
        return generate_images_config_type(**config_kwargs)

    def _generate_google_content_images(
        self,
        client: Any,
        contents: Any,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images through Gemini's generate_content API."""
        results: list[GeneratedImage] = []
        for _ in range(n):
            response = client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._build_google_generate_content_config(size, **kwargs),
            )
            results.extend(_build_google_generated_images(response, model=self._model))
            if len(results) >= n:
                break
        return results[:n]

    async def _agenerate_google_content_images(
        self,
        client: Any,
        contents: Any,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async image generation through Gemini's generate_content API."""
        results: list[GeneratedImage] = []
        for _ in range(n):
            response = await client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._build_google_generate_content_config(size, **kwargs),
            )
            results.extend(_build_google_generated_images(response, model=self._model))
            if len(results) >= n:
                break
        return results[:n]

    def _get_google_client(self) -> Any:
        """Get or create Google GenAI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                api_key=self._api_key,
                http_options={"timeout": _GOOGLE_IMAGE_REQUEST_TIMEOUT_MS},
            )
        return self._client

    def _is_gemini_model(self) -> bool:
        """Check if current model is a Gemini multimodal model."""
        from .models import GEMINI_IMAGE_MODELS

        model = self._model or ""
        return model in GEMINI_IMAGE_MODELS or model.startswith("gemini-")

    def _generate_google(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Google (Gemini or Imagen)."""
        client = self._get_google_client()

        if self._is_gemini_model():
            # Use generate_content API for Gemini models
            return self._generate_google_gemini(client, prompt, size=size, n=n, **kwargs)
        else:
            # Use generate_images API for Imagen models
            return self._generate_google_imagen(client, prompt, size=size, n=n, **kwargs)

    def _generate_google_gemini(
        self,
        client: Any,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Gemini multimodal API."""
        return self._generate_google_content_images(
            client,
            [prompt],
            size=size,
            n=n,
            **kwargs,
        )

    def _generate_google_imagen(
        self,
        client: Any,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Google Imagen API."""
        config_kwargs = dict(kwargs)

        try:
            response = client.models.generate_images(
                model=self._model,
                prompt=prompt,
                config=self._build_google_generate_images_config(size, n, **config_kwargs),
            )
        except Exception as exc:
            if "negative_prompt" in config_kwargs and _is_google_negative_prompt_unsupported_error(
                exc
            ):
                retry_kwargs = dict(config_kwargs)
                retry_kwargs.pop("negative_prompt", None)
                response = client.models.generate_images(
                    model=self._model,
                    prompt=prompt,
                    config=self._build_google_generate_images_config(size, n, **retry_kwargs),
                )
            else:
                raise

        return [
            GeneratedImage(
                data=(
                    _normalize_provider_image_bytes(img.image.image_bytes)
                    if hasattr(img, "image")
                    else None
                ),
                model=self._model,
                provider=ImageGenProvider.GOOGLE,
            )
            for img in response.generated_images
        ]

    async def _agenerate_google(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Google (Gemini or Imagen)."""
        from google import genai

        client = genai.Client(
            api_key=self._api_key,
            http_options={"timeout": _GOOGLE_IMAGE_REQUEST_TIMEOUT_MS},
        )

        if self._is_gemini_model():
            # Use generate_content API for Gemini models
            return await self._agenerate_google_gemini(client, prompt, size=size, n=n, **kwargs)
        else:
            # Use generate_images API for Imagen models
            return await self._agenerate_google_imagen(client, prompt, size=size, n=n, **kwargs)

    async def _agenerate_google_gemini(
        self,
        client: Any,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Gemini multimodal API."""
        return await self._agenerate_google_content_images(
            client,
            [prompt],
            size=size,
            n=n,
            **kwargs,
        )

    async def _agenerate_google_imagen(
        self,
        client: Any,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Google Imagen API."""
        config_kwargs = dict(kwargs)

        try:
            response = await client.aio.models.generate_images(
                model=self._model,
                prompt=prompt,
                config=self._build_google_generate_images_config(size, n, **config_kwargs),
            )
        except Exception as exc:
            if "negative_prompt" in config_kwargs and _is_google_negative_prompt_unsupported_error(
                exc
            ):
                retry_kwargs = dict(config_kwargs)
                retry_kwargs.pop("negative_prompt", None)
                response = await client.aio.models.generate_images(
                    model=self._model,
                    prompt=prompt,
                    config=self._build_google_generate_images_config(size, n, **retry_kwargs),
                )
            else:
                raise

        return [
            GeneratedImage(
                data=(
                    _normalize_provider_image_bytes(img.image.image_bytes)
                    if hasattr(img, "image")
                    else None
                ),
                model=self._model,
                provider=ImageGenProvider.GOOGLE,
            )
            for img in response.generated_images
        ]

    # -------------------------------------------------------------------------
    # xAI Implementation
    # -------------------------------------------------------------------------

    def _get_xai_client(self) -> Any:
        """Get or create xAI SDK client."""
        if self._client is None:
            try:
                from xai_sdk import Client
            except ImportError as exc:
                raise ImportError(
                    "xai-sdk is required for xAI image generation. Install ai-infra[xai]."
                ) from exc

            self._client = Client(api_key=self._api_key)
        return self._client

    def _build_xai_request_params(self, size: str, **kwargs: Any) -> dict[str, Any]:
        """Build xAI image generation parameters with ai-infra friendly defaults."""
        params = _normalize_kwargs_aliases(
            dict(kwargs),
            {
                "aspectRatio": "aspect_ratio",
                "imageFormat": "image_format",
            },
        )
        params.setdefault("image_format", "base64")

        if "aspect_ratio" not in params:
            aspect_ratio = _aspect_ratio_from_size(
                size,
                supported_ratios=_XAI_SUPPORTED_ASPECT_RATIOS,
            )
            if aspect_ratio is not None:
                params["aspect_ratio"] = aspect_ratio

        if "resolution" not in params:
            resolution = _resolution_from_size(
                size,
                supported_sizes={"1K", "2K"},
                lowercase=True,
            )
            if resolution is not None:
                params["resolution"] = resolution

        return params

    def _build_xai_generated_image(self, response: Any) -> GeneratedImage:
        """Convert xAI SDK responses into GeneratedImage objects."""
        metadata: dict[str, Any] = {}
        respect_moderation = _safe_optional_attr(response, "respect_moderation")
        if respect_moderation is not None:
            metadata["respect_moderation"] = respect_moderation

        image_data = _normalize_provider_image_bytes(_safe_optional_attr(response, "image"))
        image_url = None if image_data is not None else _safe_optional_attr(response, "url")

        return GeneratedImage(
            data=image_data,
            url=image_url,
            model=_safe_optional_attr(response, "model") or self._model,
            provider=ImageGenProvider.XAI,
            metadata=metadata,
        )

    def _generate_xai(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using xAI Grok Imagine."""
        client = self._get_xai_client()
        params = self._build_xai_request_params(size, **kwargs)

        if n > 1:
            responses = client.image.sample_batch(
                prompt=prompt,
                model=self._model,
                n=n,
                **params,
            )
            return [self._build_xai_generated_image(response) for response in responses]

        response = client.image.sample(
            prompt=prompt,
            model=self._model,
            **params,
        )
        return [self._build_xai_generated_image(response)]

    async def _agenerate_xai(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using xAI Grok Imagine."""
        try:
            from xai_sdk import AsyncClient
        except ImportError as exc:
            raise ImportError(
                "xai-sdk is required for xAI image generation. Install ai-infra[xai]."
            ) from exc

        client = AsyncClient(api_key=self._api_key)
        params = self._build_xai_request_params(size, **kwargs)

        if n > 1:
            responses = await client.image.sample_batch(
                prompt=prompt,
                model=self._model,
                n=n,
                **params,
            )
            return [self._build_xai_generated_image(response) for response in responses]

        response = await client.image.sample(
            prompt=prompt,
            model=self._model,
            **params,
        )
        return [self._build_xai_generated_image(response)]

    # -------------------------------------------------------------------------
    # Stability AI Implementation
    # -------------------------------------------------------------------------

    def _generate_stability(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Stability AI."""
        import httpx

        width, height = map(int, size.split("x"))

        response = httpx.post(
            f"https://api.stability.ai/v1/generation/{self._model}/text-to-image",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "text_prompts": [{"text": prompt, "weight": 1.0}],
                "width": width,
                "height": height,
                "samples": n,
                **kwargs,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        return [
            GeneratedImage(
                data=base64.b64decode(artifact["base64"]),
                model=self._model,
                provider=ImageGenProvider.STABILITY,
                metadata={"seed": artifact.get("seed")},
            )
            for artifact in data.get("artifacts", [])
        ]

    async def _agenerate_stability(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Stability AI."""
        import httpx

        width, height = map(int, size.split("x"))

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.stability.ai/v1/generation/{self._model}/text-to-image",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "text_prompts": [{"text": prompt, "weight": 1.0}],
                    "width": width,
                    "height": height,
                    "samples": n,
                    **kwargs,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        return [
            GeneratedImage(
                data=base64.b64decode(artifact["base64"]),
                model=self._model,
                provider=ImageGenProvider.STABILITY,
                metadata={"seed": artifact.get("seed")},
            )
            for artifact in data.get("artifacts", [])
        ]

    # -------------------------------------------------------------------------
    # Replicate Implementation
    # -------------------------------------------------------------------------

    def _generate_replicate(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate images using Replicate."""
        import replicate

        width, height = map(int, size.split("x"))

        # Run the model
        output = replicate.run(
            self._model,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": n,
                **kwargs,
            },
        )

        # Output is typically a list of URLs
        if isinstance(output, list):
            return [
                GeneratedImage(
                    url=str(url),
                    model=self._model,
                    provider=ImageGenProvider.REPLICATE,
                )
                for url in output
            ]
        else:
            return [
                GeneratedImage(
                    url=str(output),
                    model=self._model,
                    provider=ImageGenProvider.REPLICATE,
                )
            ]

    async def _agenerate_replicate(
        self,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Async generate images using Replicate."""
        import replicate

        width, height = map(int, size.split("x"))

        # Replicate's async API
        output = await replicate.async_run(
            self._model,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": n,
                **kwargs,
            },
        )

        if isinstance(output, list):
            return [
                GeneratedImage(
                    url=str(url),
                    model=self._model,
                    provider=ImageGenProvider.REPLICATE,
                )
                for url in output
            ]
        else:
            return [
                GeneratedImage(
                    url=str(output),
                    model=self._model,
                    provider=ImageGenProvider.REPLICATE,
                )
            ]

    # -------------------------------------------------------------------------
    # Edit and Variations (OpenAI-specific)
    # -------------------------------------------------------------------------

    def edit(
        self,
        image: str | bytes,
        prompt: str,
        *,
        mask: str | bytes | None = None,
        size: str = "1024x1024",
        n: int = 1,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Edit an existing image based on a prompt.

        Note: Supported by OpenAI image-edit models.

        Args:
            image: Path to image file or image bytes.
            prompt: Description of the edit to make.
            mask: Optional mask image (transparent areas will be edited).
            size: Output image size.
            n: Number of variations to generate.
            **kwargs: Additional options.

        Returns:
            List of GeneratedImage objects.
        """
        if self._provider == ImageGenProvider.GOOGLE:
            if mask is not None:
                raise NotImplementedError("mask editing is only supported for OpenAI")
            return self._edit_google(image, prompt, size=size, n=n, **kwargs)

        if self._provider == ImageGenProvider.XAI:
            if mask is not None:
                raise NotImplementedError("mask editing is only supported for OpenAI")
            return self._edit_xai(image, prompt, size=size, n=n, **kwargs)

        if self._provider != ImageGenProvider.OPENAI:
            raise NotImplementedError(f"edit() not supported for {self._provider}")

        client = self._get_openai_client()

        # Handle image input
        image_file: BinaryIO
        if isinstance(image, str):
            image_file = open(image, "rb")
        else:
            image_file = io.BytesIO(image)
            image_file.name = "image.png"

        # Handle mask input
        mask_file: BinaryIO | None = None
        if mask is not None:
            if isinstance(mask, str):
                mask_file = open(mask, "rb")
            else:
                mask_file = io.BytesIO(mask)
                mask_file.name = "mask.png"

        try:
            request_model = self._model if _is_openai_gpt_image_model(self._model) else "dall-e-2"
            params: dict[str, Any] = {
                "model": request_model,
                "image": image_file,
                "prompt": prompt,
                "size": size,
                "n": n,
            }
            if _is_openai_gpt_image_model(self._model):
                normalized_quality = self._openai_generate_quality(
                    kwargs.pop("quality", "standard")
                )
                if normalized_quality is not None:
                    params["quality"] = normalized_quality
            if mask_file:
                params["mask"] = mask_file
            params.update(kwargs)

            response = self._run_openai_image_operation(lambda: client.images.edit(**params))

            return self._build_openai_generated_images(response.data, model=request_model)
        finally:
            if isinstance(image, str):
                image_file.close()
            if mask is not None and isinstance(mask, str):
                if mask_file is not None:
                    mask_file.close()

    def _edit_google(
        self,
        image: str | bytes,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Edit an image using Google Gemini image models."""
        if not self._is_gemini_model():
            raise NotImplementedError(
                "Google image editing requires a Gemini image model, such as "
                "gemini-3.1-flash-image-preview or gemini-3-pro-image-preview"
            )

        client = self._get_google_client()
        additional_images = kwargs.pop("reference_images", None)
        if additional_images is None:
            additional_images = kwargs.pop("input_images", None)
        if additional_images is None:
            additional_images = kwargs.pop("images", None)

        contents = [prompt, _build_google_image_part(image)]
        for extra_image in additional_images or []:
            contents.append(_build_google_image_part(extra_image))

        return self._generate_google_content_images(
            client,
            contents,
            size=size,
            n=n,
            **kwargs,
        )

    def _edit_xai(
        self,
        image: str | bytes,
        prompt: str,
        *,
        size: str,
        n: int,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Edit an image using xAI Grok Imagine."""
        client = self._get_xai_client()
        params = self._build_xai_request_params(size, **kwargs)

        image_urls = params.get("image_urls")
        if image_urls is None:
            params["image_url"] = _image_input_to_data_uri(image)

        if n > 1:
            responses = client.image.sample_batch(
                prompt=prompt,
                model=self._model,
                n=n,
                **params,
            )
            return [self._build_xai_generated_image(response) for response in responses]

        response = client.image.sample(
            prompt=prompt,
            model=self._model,
            **params,
        )
        return [self._build_xai_generated_image(response)]

    def variations(
        self,
        image: str | bytes,
        *,
        size: str = "1024x1024",
        n: int = 1,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Generate variations of an existing image.

        Note: Currently only supported by OpenAI (DALL-E 2).

        Args:
            image: Path to image file or image bytes.
            size: Output image size.
            n: Number of variations to generate.
            **kwargs: Additional options.

        Returns:
            List of GeneratedImage objects.
        """
        if self._provider != ImageGenProvider.OPENAI:
            raise NotImplementedError(f"variations() not supported for {self._provider}")

        client = self._get_openai_client()

        # Handle image input
        image_file: BinaryIO
        if isinstance(image, str):
            image_file = open(image, "rb")
        else:
            image_file = io.BytesIO(image)
            image_file.name = "image.png"

        try:
            response = self._run_openai_image_operation(
                lambda: client.images.create_variation(
                    model="dall-e-2",  # Only DALL-E 2 supports variations
                    image=image_file,
                    size=size,
                    n=n,
                    **kwargs,
                )
            )

            return [
                GeneratedImage(
                    url=img.url,
                    model="dall-e-2",
                    provider=ImageGenProvider.OPENAI,
                )
                for img in response.data
            ]
        finally:
            if isinstance(image, str):
                image_file.close()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def list_providers() -> list[str]:
        """List all available providers."""
        return [p.value for p in ImageGenProvider]

    @staticmethod
    def list_models(
        provider: str,
        *,
        refresh: bool = False,
        timeout: float | None = None,
    ) -> list[str]:
        """List live models for a provider by querying the provider API.

        Args:
            provider: Provider name.
            refresh: Force refresh from API, bypassing cache.
            timeout: Optional request timeout in seconds for HTTP-based fetchers.

        Returns:
            List of live model IDs available from the provider.
        """
        from ai_infra.imagegen.discovery import list_models

        return list_models(provider, refresh=refresh, timeout=timeout)

    @staticmethod
    def list_known_models(provider: str) -> list[str]:
        """List the built-in fallback model catalog for a provider.

        Args:
            provider: Provider name.

        Returns:
            List of statically known model names.
        """
        from ai_infra.imagegen.discovery import list_known_models

        return list_known_models(provider)

    @staticmethod
    def list_available_models(
        provider: str,
        *,
        refresh: bool = False,
        timeout: float | None = None,
    ) -> list[str]:
        """List live models for a provider by querying the provider API.

        This explicit alias is kept for backwards compatibility.

        Args:
            provider: Provider name.
            refresh: Force refresh from API, bypassing cache.
            timeout: Optional request timeout in seconds for HTTP-based fetchers.

        Returns:
            List of live model IDs available from the provider.
        """
        from ai_infra.imagegen.discovery import list_available_models

        return list_available_models(provider, refresh=refresh, timeout=timeout)

    @staticmethod
    def list_all_models(
        *,
        refresh: bool = False,
        timeout: float | None = None,
    ) -> dict[str, list[str]]:
        """List live models for all configured image generation providers.

        Args:
            refresh: Force refresh from API, bypassing cache.
            timeout: Optional request timeout in seconds for HTTP-based fetchers.

        Returns:
            Dict mapping provider names to lists of live model IDs.
        """
        from ai_infra.imagegen.discovery import list_all_models

        return list_all_models(refresh=refresh, timeout=timeout)

    @staticmethod
    def list_configured_providers() -> list[str]:
        """List configured image generation providers."""
        from ai_infra.imagegen.discovery import list_configured_providers

        return list_configured_providers()

    @staticmethod
    def is_provider_configured(provider: str) -> bool:
        """Check whether an image generation provider is configured."""
        from ai_infra.imagegen.discovery import is_provider_configured

        return is_provider_configured(provider)
