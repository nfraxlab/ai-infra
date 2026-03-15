from __future__ import annotations

from pathlib import Path
from typing import Any

ImageInput = str | bytes | Path
AudioInput = str | bytes | Path


DocumentInput = str | bytes | Path


def make_messages(
    user: str,
    system: str | None = None,
    extras: list[dict[str, Any]] | None = None,
    images: list[ImageInput] | None = None,
    audio: AudioInput | None = None,
    documents: list[DocumentInput] | None = None,
    provider: str | None = None,  # Kept for backwards compatibility, but ignored
    history: list[dict[str, Any]] | None = None,
):
    """Create a list of messages for LLM chat.

    Args:
        user: The user message.
        system: Optional system message.
        extras: Optional additional messages.
        images: Optional list of images (URLs, bytes, or file paths).
        audio: Optional audio input (URL, bytes, or file path).
        documents: Optional list of document files (PDFs, text files) for native file understanding.
        provider: Deprecated - no longer needed. Kept for backwards compatibility.
        history: Optional conversation history inserted before the current user message.
            Each entry is a dict with ``role`` and ``content`` keys. Content may be a
            string or a list of content blocks (e.g. image + text from a prior turn).

    Returns:
        List of message dicts.
    """
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if history:
        msgs.extend(history)

    # Handle multimodal content in user message
    if images or audio or documents:
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]

        # Add images
        if images:
            from ai_infra.llm.multimodal.vision import build_vision_content

            # build_vision_content returns list including text, we just want image blocks
            vision_content = build_vision_content("", images)
            # Skip the empty text block and get just image blocks
            content.extend([c for c in vision_content if c.get("type") != "text"])

        # Add audio
        if audio:
            from ai_infra.llm.multimodal.audio import encode_audio

            content.append(encode_audio(audio))

        # Add documents
        if documents:
            from ai_infra.llm.multimodal.documents import encode_document

            for doc in documents:
                content.append(encode_document(doc))

        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user", "content": user})

    if extras:
        msgs.extend(extras)
    return msgs


def is_valid_response(res: Any) -> bool:
    """Generic 'did we get something usable?' check."""
    content = getattr(res, "content", None)
    if content is not None:
        return str(content).strip() != ""
    if isinstance(res, dict) and isinstance(res.get("messages"), list) and res["messages"]:
        last = res["messages"][-1]
        if hasattr(last, "content"):
            return str(getattr(last, "content", "")).strip() != ""
        if isinstance(last, dict):
            return str(last.get("content", "")).strip() != ""
    return res is not None
