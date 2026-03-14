"""Document input support for LLM - native file understanding without text extraction.

This module provides a simple, provider-agnostic API for passing documents
directly to multimodal LLMs. Unlike the Retriever (which extracts text and
does RAG), this sends the full file binary as a native content block so the
model can see the original layout, formatting, and structure.

Supported providers:
- Anthropic (Claude 3.5+) - native PDF understanding
- Google (Gemini 1.5+, Gemini 2.0) - native PDF understanding
- OpenAI (GPT-4o with Files API) - via base64 with mime_type

Supported file types:
- PDF (.pdf) - full layout/table/structure preservation
- Plain text (.txt, .md) - passed as text-plain blocks for efficiency
- Images (.jpg, .png, .gif, .webp) - delegated to vision module

Example:
    ```python
    from ai_infra import LLM

    llm = LLM()

    # Single document
    response = llm.chat(
        "Summarize this document",
        files=["report.pdf"]
    )

    # Multiple documents
    response = llm.chat(
        "Compare these two contracts",
        files=["contract_a.pdf", "contract_b.pdf"]
    )

    # Mix documents and images in one message
    response = llm.chat(
        "What does the chart in this report show?",
        files=["report.pdf"],
        images=["chart.png"]
    )

    # Raw bytes (e.g., from a file upload handler)
    with open("report.pdf", "rb") as f:
        pdf_bytes = f.read()
    response = llm.chat("Summarize", files=[pdf_bytes])
    ```
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

# Type alias for document inputs we accept
DocumentInput = str | bytes | Path

# MIME types that receive native file-block treatment
_DOCUMENT_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
}

# Plain text types sent as text-plain blocks (more efficient than base64)
_TEXT_MIME_TYPES: dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

# Image extensions - delegated to vision module rather than encoded here
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})


# =============================================================================
# Public API
# =============================================================================


def create_document_message(
    text: str,
    documents: list[DocumentInput],
) -> HumanMessage:
    """Create a LangChain HumanMessage with text and document file attachments.

    This is the recommended way to create messages with file attachments.
    The model receives the files as native binary content blocks, preserving
    layout and formatting without any text extraction step.

    Args:
        text: The text prompt / question about the document(s).
        documents: List of documents - can be file paths, URLs, or raw bytes.

    Returns:
        HumanMessage ready for model.invoke().

    Example:
        ```python
        from ai_infra.llm.multimodal.documents import create_document_message

        msg = create_document_message("Summarize this", ["report.pdf"])
        response = llm.invoke([msg])
        ```
    """
    content = build_document_content(text, documents)
    return HumanMessage(content=content)  # type: ignore[arg-type]


def build_document_content(
    text: str,
    documents: list[DocumentInput],
) -> list[dict[str, Any]]:
    """Build content blocks for a message containing document file attachments.

    Creates a list of content blocks in LangChain's standard format.
    Each document becomes a ``{"type": "file", ...}`` content block,
    which LangChain's provider adapters translate to the native format
    (e.g., Anthropic ``document`` blocks, Gemini ``fileData`` parts).

    Plain-text files (.txt, .md) are emitted as ``{"type": "text-plain"}``
    blocks, which avoids unnecessary base64 overhead.

    Args:
        text: The text prompt / question.
        documents: List of document inputs (paths, bytes).

    Returns:
        List of content block dicts ready for HumanMessage.

    Example:
        ```python
        content = build_document_content("Summarize this", ["report.pdf"])
        # [
        #     {"type": "text", "text": "Summarize this"},
        #     {
        #         "type": "file",
        #         "source_type": "base64",
        #         "data": "<base64>",
        #         "mime_type": "application/pdf",
        #     },
        # ]
        ```
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for document in documents:
        content.append(encode_document(document))
    return content


def encode_document(document: DocumentInput) -> dict[str, Any]:
    """Encode a single document to LangChain's standard file content block format.

    Automatically selects the appropriate encoding strategy:
    - URLs are passed through as ``source_type: "url"`` blocks.
    - PDF paths/bytes are base64-encoded and sent as ``source_type: "base64"``
      blocks with ``mime_type: "application/pdf"``.
    - Plain-text paths are read and returned as ``type: "text-plain"`` blocks.
    - Image paths are delegated to :func:`ai_infra.llm.multimodal.vision.encode_image`.

    Args:
        document: URL string, file path string, Path object, or raw bytes.

    Returns:
        Content block dict suitable for inclusion in a HumanMessage content list.

    Raises:
        FileNotFoundError: If a file path does not exist.
        TypeError: If the input type is not supported.
        ValueError: If the file extension is not supported.

    Example:
        ```python
        # PDF file
        encode_document("report.pdf")
        # {
        #     "type": "file",
        #     "source_type": "base64",
        #     "data": "JVBERi0x...",
        #     "mime_type": "application/pdf",
        # }

        # Plain text
        encode_document("notes.txt")
        # {"type": "text-plain", "text": "...", "mime_type": "text/plain"}

        # PDF bytes
        encode_document(pdf_bytes)
        # {"type": "file", "source_type": "base64", "data": "...", "mime_type": "application/pdf"}
        ```
    """
    if isinstance(document, bytes):
        return _encode_bytes_document(document)
    elif isinstance(document, Path):
        return _encode_path_document(document)
    elif isinstance(document, str):
        return _encode_string_document(document)
    else:
        raise TypeError(
            f"Unsupported document type: {type(document)}. "
            "Expected str (URL or path), bytes, or Path."
        )


# =============================================================================
# Internal helpers
# =============================================================================


def _is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith(("http://", "https://", "data:"))


def _encode_string_document(document: str) -> dict[str, Any]:
    """Encode a string document (URL or file path)."""
    if _is_url(document):
        return _encode_url_document(document)
    return _encode_path_document(Path(document))


def _encode_url_document(url: str) -> dict[str, Any]:
    """Encode a URL document as a file content block with source_type 'url'."""
    # Attempt to infer mime_type from the URL path
    path_part = url.split("?")[0]
    ext = Path(path_part).suffix.lower()
    mime_type = _DOCUMENT_MIME_TYPES.get(ext) or _TEXT_MIME_TYPES.get(ext) or "application/pdf"
    return {
        "type": "file",
        "source_type": "url",
        "url": url,
        "mime_type": mime_type,
    }


def _encode_path_document(path: Path) -> dict[str, Any]:
    """Encode a file path as the appropriate content block."""
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    ext = path.suffix.lower()

    # Plain-text files: emit as text-plain block (no base64 needed)
    if ext in _TEXT_MIME_TYPES:
        text = path.read_text(encoding="utf-8")
        return {
            "type": "text-plain",
            "text": text,
            "mime_type": _TEXT_MIME_TYPES[ext],
        }

    # Image files: delegate to vision module
    if ext in _IMAGE_EXTENSIONS:
        from ai_infra.llm.multimodal.vision import encode_image

        return encode_image(path)

    # PDF and other binary documents: base64 encode
    if ext in _DOCUMENT_MIME_TYPES:
        data = path.read_bytes()
        return _encode_bytes_document(data, mime_type=_DOCUMENT_MIME_TYPES[ext])

    raise ValueError(
        f"Unsupported document file type: '{ext}'. "
        f"Supported types: {', '.join(sorted(_DOCUMENT_MIME_TYPES) + sorted(_TEXT_MIME_TYPES))}. "
        "For images, use the images= parameter instead."
    )


def _encode_bytes_document(
    data: bytes,
    mime_type: str = "application/pdf",
) -> dict[str, Any]:
    """Encode raw document bytes as a base64 file content block."""
    b64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "file",
        "source_type": "base64",
        "data": b64,
        "mime_type": mime_type,
    }


__all__ = [
    "create_document_message",
    "build_document_content",
    "encode_document",
    "DocumentInput",
]
