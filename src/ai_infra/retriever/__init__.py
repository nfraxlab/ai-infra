"""Retriever module for ai-infra.

Provides dead-simple document retrieval - add documents, search with natural language.
No external dependencies exposed - just use ai-infra!

Example - Basic usage:
    ```python
    from ai_infra import Retriever

    # Zero-config: uses memory backend, auto-detects embedding provider
    retriever = Retriever()

    # Add anything - auto-detects type
    retriever.add("This is some text about refunds...")     # Raw text
    retriever.add("./docs/policy.pdf")                       # Single file
    retriever.add("./docs/")                                 # Whole directory

    # Search - returns list of strings
    results = retriever.search("What is the refund policy?")
    # ["Most relevant chunk", "Second most relevant", ...]
    ```

Example - With detailed results:
    ```python
    from ai_infra import Retriever

    retriever = Retriever()
    retriever.add("./company_docs/")

    # Get detailed results (scores, sources, metadata)
    results = retriever.search("refund policy", detailed=True)
    for r in results:
        print(f"{r.score:.2f}: {r.text[:50]}... (from {r.source})")
    ```

Example - Production config:
    ```python
    from ai_infra import Retriever

    retriever = Retriever(
        provider="openai",
        model="text-embedding-3-large",
        backend="postgres",
        connection_string="postgresql://user:pass@host:5432/db",
    )
    ```
"""

# Re-export from llm/tools/custom for convenience
from ai_infra.llm.tools.custom.retriever import (
    create_retriever_tool,
    create_retriever_tool_async,
)
from ai_infra.retriever.loaders import (
    load_excel_bytes,
    ocr_page_from_bytes,
)
from ai_infra.retriever.models import Chunk, SearchResult
from ai_infra.retriever.retriever import Retriever

__all__ = [
    "Chunk",
    "Retriever",
    "SearchResult",
    "create_retriever_tool",
    "create_retriever_tool_async",
    "load_excel_bytes",
    "ocr_page_from_bytes",
]
