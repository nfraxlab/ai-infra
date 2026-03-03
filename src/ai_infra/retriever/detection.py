"""Input type detection for the Retriever module.

Automatically detects whether a string input is:
- Raw text content
- A file path
- A directory path
"""

from __future__ import annotations

import os
from typing import Literal

# Common file extensions we handle
FILE_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".docx",
    ".doc",
    ".csv",
    ".json",
    ".html",
    ".htm",
    ".xml",
    ".rst",
    ".xlsx",
    ".xls",
}


def detect_input_type(value: str) -> Literal["text", "file", "directory"]:
    """Detect if input is raw text, a file path, or a directory.

    Detection logic:
    1. If it looks like a path (has separators or starts with . / ~)
       AND exists on the filesystem, return "file" or "directory"
    2. If it looks like a path but doesn't exist, raise FileNotFoundError
    3. Otherwise, treat as raw text

    Args:
        value: The input string to analyze.

    Returns:
        "text", "file", or "directory"

    Raises:
        FileNotFoundError: If input looks like a path but doesn't exist.

    Examples:
        >>> detect_input_type("Hello world")
        'text'
        >>> detect_input_type("./README.md")  # if file exists
        'file'
        >>> detect_input_type("./docs/")  # if directory exists
        'directory'
    """
    # Check if it looks like a path
    looks_like_path = _looks_like_path(value)

    if looks_like_path:
        # Expand user home directory (~)
        expanded = os.path.expanduser(value)

        # Check if it exists
        if os.path.isdir(expanded):
            return "directory"
        if os.path.isfile(expanded):
            return "file"

        # Looks like a path but doesn't exist
        # Check if it has a file extension we recognize
        _, ext = os.path.splitext(value.lower())
        if ext in FILE_EXTENSIONS:
            raise FileNotFoundError(
                f"File not found: {value}\n"
                f"The path looks like a file but doesn't exist. "
                f"If you meant to add raw text, use retriever.add_text() instead."
            )

        # Check if it ends with / (clearly meant to be a directory)
        if value.endswith("/") or value.endswith("\\"):
            raise FileNotFoundError(
                f"Directory not found: {value}\n"
                f"The path looks like a directory but doesn't exist. "
                f"If you meant to add raw text, use retriever.add_text() instead."
            )

    # Default: treat as raw text
    return "text"


def _looks_like_path(value: str) -> bool:
    """Check if a string looks like a file or directory path.

    Args:
        value: The string to check.

    Returns:
        True if it looks like a path, False otherwise.
    """
    # Empty or whitespace-only strings are not paths
    if not value or not value.strip():
        return False

    # Very long strings are probably text content, not paths
    # (Most file paths are under 260 characters on Windows, 4096 on Unix)
    if len(value) > 500:
        return False

    # Newlines in the string means it's definitely text content
    if "\n" in value:
        return False

    # Starts with path indicators
    if value.startswith(("./", "../", "/", "~", "~/")):
        return True

    # Windows paths
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        return True

    # Contains path separators AND has a file extension we recognize
    if "/" in value or "\\" in value:
        _, ext = os.path.splitext(value.lower())
        if ext in FILE_EXTENSIONS:
            return True

    # Just a filename with extension (e.g., "document.pdf")
    # Only treat as path if it actually exists
    _, ext = os.path.splitext(value.lower())
    if ext in FILE_EXTENSIONS and os.path.exists(value):
        return True

    return False
