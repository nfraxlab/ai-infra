"""Vision support for LLM - simple image input for any provider.

This module provides a simple, provider-agnostic API for vision (image input)
that works with any LLM provider that supports vision. It uses LangChain's
standardized content block format internally for maximum compatibility.

Supported providers (automatic):
- OpenAI (GPT-4o, GPT-4V)
- Anthropic (Claude 3, Claude 3.5)
- Google (Gemini 1.5, Gemini 2.0)
- Any other LangChain-supported vision model

Example:
    ```python
    from ai_infra.llm import LLM

    llm = LLM()

    # Simple: Just pass image URLs or paths
    response = llm.chat(
        "What's in this image?",
        images=["https://example.com/photo.jpg"]
    )

    # Multiple images
    response = llm.chat(
        "Compare these two images",
        images=["image1.png", "image2.png"]
    )

    # Mix URLs and local files
    response = llm.chat(
        "What do these have in common?",
        images=[
            "https://example.com/cat.jpg",
            "/path/to/dog.png",
        ]
    )

    # Raw bytes (e.g., from camera, screenshot)
    screenshot = take_screenshot()
    response = llm.chat("Describe this screen", images=[screenshot])
    ```
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

# Type alias for image inputs we accept
ImageInput = str | bytes | Path


# =============================================================================
# Public API - Simple and Clean
# =============================================================================


def create_vision_message(
    text: str,
    images: list[ImageInput],
) -> HumanMessage:
    """Create a LangChain HumanMessage with text and images.

    This is the recommended way to create vision messages. It returns
    a LangChain HumanMessage that works with any vision-capable model.

    Args:
        text: The text prompt/question about the image(s).
        images: List of images - can be URLs, file paths, or raw bytes.

    Returns:
        HumanMessage ready for model.invoke().

    Example:
        ```python
        from ai_infra.llm.multimodal import create_vision_message
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o")
        msg = create_vision_message("What's in this image?", ["photo.jpg"])
        response = model.invoke([msg])
        ```
    """
    content = build_vision_content(text, images)
    # HumanMessage expects list[str | dict[Any, Any]] but we have list[dict[str, Any]]
    # These are effectively compatible - dict[str, Any] is more specific
    return HumanMessage(content=content)  # type: ignore[arg-type]


def build_vision_content(
    text: str,
    images: list[ImageInput],
) -> list[dict[str, Any]]:
    """Build content blocks for a vision message.

    Creates a list of content blocks in LangChain's standard format.
    This format is automatically converted to provider-specific formats
    by LangChain's chat model adapters.

    Args:
        text: The text prompt/question.
        images: List of images (URLs, paths, or bytes).

    Returns:
        List of content blocks ready for HumanMessage.

    Example:
        ```python
        content = build_vision_content(
            "Describe these",
            ["https://example.com/a.jpg", "local.png"]
        )
        # [
        #     {"type": "text", "text": "Describe these"},
        #     {"type": "image_url", "image_url": {"url": "https://..."}},
        #     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        # ]
        ```
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]

    for image in images:
        content.append(encode_image(image))

    return content


def encode_image(image: ImageInput) -> dict[str, Any]:
    """Encode a single image to LangChain's standard format.

    Automatically detects the image type and encodes appropriately:
    - URLs are passed through directly
    - File paths are read and base64 encoded with correct MIME type
    - Raw bytes are base64 encoded

    Args:
        image: URL string, file path, Path object, or raw bytes.

    Returns:
        Content block dict in LangChain's image_url format.

    Example:
        ```python
        # URL
        encode_image("https://example.com/cat.jpg")
        # {"type": "image_url", "image_url": {"url": "https://..."}}

        # File path
        encode_image("photo.png")
        # {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

        # Bytes
        encode_image(screenshot_bytes)
        # {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ```
    """
    if isinstance(image, str):
        return _encode_string_image(image)
    elif isinstance(image, bytes):
        return _encode_bytes_image(image)
    elif isinstance(image, Path):
        return _encode_path_image(image)
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. Expected str (URL or path), bytes, or Path."
        )


# =============================================================================
# Internal Helpers
# =============================================================================


def _is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith(("http://", "https://", "data:"))


def _get_mime_type(path: Path) -> str:
    """Get MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "image/png"


def _encode_string_image(image: str) -> dict[str, Any]:
    """Encode a string image (URL or file path)."""
    if _is_url(image):
        # URLs pass through directly
        return {"type": "image_url", "image_url": {"url": image}}
    else:
        # Treat as file path
        return _encode_path_image(Path(image))


_MAGIC_MIME: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP — checked further below
    (b"\x00\x00\x00\x0cjP  ", "image/jp2"),
]


def _detect_mime_type(data: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    for magic, mime in _MAGIC_MIME:
        if data[: len(magic)] == magic:
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return "image/png"


def _encode_bytes_image(image: bytes, mime_type: str | None = None) -> dict[str, Any]:
    """Encode raw bytes to base64 data URL."""
    resolved_mime = mime_type or _detect_mime_type(image)
    b64 = base64.b64encode(image).decode("utf-8")
    data_url = f"data:{resolved_mime};base64,{b64}"
    return {"type": "image_url", "image_url": {"url": data_url}}


def _encode_path_image(path: Path) -> dict[str, Any]:
    """Encode a file path to base64 data URL."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type = _get_mime_type(path)
    image_bytes = path.read_bytes()
    return _encode_bytes_image(image_bytes, mime_type)


# =============================================================================
# Backwards Compatibility (deprecated - use encode_image instead)
# =============================================================================


def encode_image_for_openai(image: ImageInput) -> dict[str, Any]:
    """Encode image for OpenAI API.

    .. deprecated::
        Use `encode_image()` instead - it works for all providers.
    """
    return encode_image(image)


def encode_image_for_anthropic(image: ImageInput) -> dict[str, Any]:
    """Encode image for Anthropic API.

    .. deprecated::
        Use `encode_image()` instead - it works for all providers.
    """
    return encode_image(image)


def encode_image_for_google(image: ImageInput) -> dict[str, Any]:
    """Encode image for Google API.

    .. deprecated::
        Use `encode_image()` instead - it works for all providers.
    """
    return encode_image(image)


def make_vision_message(
    text: str,
    images: list[ImageInput],
    provider: str = "",  # Ignored - kept for backwards compatibility
) -> list[dict[str, Any]]:
    """Create message content with text and images.

    .. deprecated::
        Use `build_vision_content()` or `create_vision_message()` instead.
        The `provider` parameter is no longer needed.
    """
    return build_vision_content(text, images)


__all__ = [
    # New API (recommended)
    "create_vision_message",
    "build_vision_content",
    "encode_image",
    "ImageInput",
    # Backwards compatibility (deprecated)
    "encode_image_for_openai",
    "encode_image_for_anthropic",
    "encode_image_for_google",
    "make_vision_message",
]
