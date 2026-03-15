"""Embeddings module for ai-infra.

Provides a simple, provider-agnostic interface for text embeddings.
No external dependencies exposed - just use ai-infra!

Example - Basic usage:
    ```python
    from ai_infra import Embeddings

    # Zero-config: uses first available provider
    embeddings = Embeddings()

    # Single text
    vector = embeddings.embed("Hello, world!")

    # Batch embedding
    vectors = embeddings.embed_batch(["Hello", "World", "!"])
    ```

Example - With VectorStore:
    ```python
    from ai_infra import Embeddings, VectorStore

    embeddings = Embeddings()
    store = VectorStore(embeddings=embeddings)

    # Add documents
    store.add_texts(["doc1", "doc2", "doc3"])

    # Search
    results = store.search("query", k=3)
    for doc, score in results:
        print(f"{score:.2f}: {doc}")
    ```
"""

from ai_infra.embeddings.embeddings import Embeddings
from ai_infra.embeddings.multimodal import MultimodalEmbeddings
from ai_infra.embeddings.vectorstore import VectorStore

__all__ = [
    "Embeddings",
    "MultimodalEmbeddings",
    "VectorStore",
]
