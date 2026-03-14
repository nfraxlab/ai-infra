"""Tests for LLM document input (native file content block) functionality."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pytest

from ai_infra.llm.multimodal.documents import (
    DocumentInput,
    build_document_content,
    create_document_message,
    encode_document,
)


class TestEncodeDocument:
    """Tests for encode_document function."""

    def test_encode_pdf_bytes(self):
        """PDF bytes produce a base64 file block with application/pdf mime type."""
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        result = encode_document(pdf_bytes)

        assert result["type"] == "file"
        assert result["source_type"] == "base64"
        assert result["mime_type"] == "application/pdf"
        assert "data" in result
        # Verify base64 round-trips correctly
        assert base64.b64decode(result["data"]) == pdf_bytes

    def test_encode_pdf_path_string(self):
        """PDF file path string produces a base64 file block."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            path = f.name

        try:
            result = encode_document(path)
            assert result["type"] == "file"
            assert result["source_type"] == "base64"
            assert result["mime_type"] == "application/pdf"
            assert base64.b64decode(result["data"]) == b"%PDF-1.4 test"
        finally:
            Path(path).unlink()

    def test_encode_pdf_path_object(self):
        """Path object for a PDF produces a base64 file block."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            path = Path(f.name)

        try:
            result = encode_document(path)
            assert result["type"] == "file"
            assert result["mime_type"] == "application/pdf"
        finally:
            path.unlink()

    def test_encode_txt_file(self):
        """Plain text file produces a text-plain block (no base64 overhead)."""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("Hello, world!")
            path = f.name

        try:
            result = encode_document(path)
            assert result["type"] == "text-plain"
            assert result["text"] == "Hello, world!"
            assert result["mime_type"] == "text/plain"
        finally:
            Path(path).unlink()

    def test_encode_markdown_file(self):
        """Markdown file produces a text-plain block with text/markdown mime type."""
        with tempfile.NamedTemporaryFile(
            suffix=".md", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("# Title\n\nBody text.")
            path = f.name

        try:
            result = encode_document(path)
            assert result["type"] == "text-plain"
            assert "# Title" in result["text"]
            assert result["mime_type"] == "text/markdown"
        finally:
            Path(path).unlink()

    def test_encode_url(self):
        """HTTP URL produces a file block with source_type 'url'."""
        url = "https://example.com/report.pdf"
        result = encode_document(url)

        assert result["type"] == "file"
        assert result["source_type"] == "url"
        assert result["url"] == url
        assert result["mime_type"] == "application/pdf"

    def test_encode_url_infers_mime_type(self):
        """URL mime type is inferred from the path extension."""
        url = "https://example.com/notes.txt"
        result = encode_document(url)

        assert result["type"] == "file"
        assert result["source_type"] == "url"
        assert result["mime_type"] == "text/plain"

    def test_image_path_delegated_to_vision(self):
        """Image file paths are delegated to the vision encoder (image_url block)."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Minimal 1x1 PNG bytes
            f.write(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
                b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            path = f.name

        try:
            result = encode_document(path)
            # Vision module returns image_url blocks
            assert result["type"] == "image_url"
        finally:
            Path(path).unlink()

    def test_file_not_found(self):
        """Missing file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Document file not found"):
            encode_document("/nonexistent/path/report.pdf")

    def test_unsupported_extension_raises(self):
        """Unsupported file type raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"binary data")
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported document file type"):
                encode_document(path)
        finally:
            Path(path).unlink()

    def test_unsupported_type_raises(self):
        """Passing an unsupported Python type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported document type"):
            encode_document(12345)  # type: ignore[arg-type]


class TestBuildDocumentContent:
    """Tests for build_document_content function."""

    def test_single_document(self):
        """Single document produces text block followed by file block."""
        pdf_bytes = b"%PDF-1.4 test"
        content = build_document_content("Summarize this", [pdf_bytes])

        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Summarize this"}
        assert content[1]["type"] == "file"

    def test_multiple_documents(self):
        """Multiple documents each become their own content block."""
        content = build_document_content(
            "Compare these",
            [b"%PDF-1.4 doc1", b"%PDF-1.4 doc2"],
        )

        assert len(content) == 3  # 1 text + 2 file blocks
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "file"
        assert content[2]["type"] == "file"

    def test_empty_text_with_document(self):
        """Empty string text still produces a text block first."""
        content = build_document_content("", [b"%PDF-1.4 test"])

        assert content[0] == {"type": "text", "text": ""}
        assert content[1]["type"] == "file"


class TestCreateDocumentMessage:
    """Tests for create_document_message function."""

    def test_creates_human_message(self):
        """Creates a LangChain HumanMessage with the correct content."""
        msg = create_document_message("Summarize this", [b"%PDF-1.4 test"])

        assert msg.__class__.__name__ == "HumanMessage"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "file"

    def test_message_with_text_file(self):
        """Plain text file produces a HumanMessage with text-plain block."""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("Some notes")
            path = f.name

        try:
            msg = create_document_message("What are these notes about?", [path])
            assert msg.__class__.__name__ == "HumanMessage"
            assert isinstance(msg.content, list)
            assert msg.content[1]["type"] == "text-plain"
            assert msg.content[1]["text"] == "Some notes"
        finally:
            Path(path).unlink()


class TestDocumentInputTypeAlias:
    """Tests that DocumentInput type alias accepts all expected types."""

    def test_str_is_valid_document_input(self):
        """Strings (paths or URLs) are valid DocumentInput values."""
        # This is a compile-time check; at runtime str is always valid
        val: DocumentInput = "report.pdf"
        assert isinstance(val, str)

    def test_bytes_is_valid_document_input(self):
        val: DocumentInput = b"%PDF-1.4"
        assert isinstance(val, bytes)

    def test_path_is_valid_document_input(self):
        val: DocumentInput = Path("report.pdf")
        assert isinstance(val, Path)
