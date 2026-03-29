"""Memory management for ai-infra agents.

This module provides:
1. `fit_context()` - The primary API for fitting messages into token budgets
2. Long-term memory store (MemoryStore with semantic search)
3. Conversation history RAG (ConversationMemory)

Context Management (Primary API):
    ```python
    from ai_infra.memory import fit_context

    # Simple: fit messages into budget (trims oldest)
    result = fit_context(messages, max_tokens=4000)

    # With summarization: compress old messages instead of dropping
    result = fit_context(messages, max_tokens=4000, summarize=True)

    # Rolling summary: extend existing summary (for stateless APIs)
    result = fit_context(
        messages,
        max_tokens=4000,
        summarize=True,
        summary="Previous summary...",
    )
    ```

Long-term Memory (across sessions):
    ```python
    from ai_infra.memory import MemoryStore

    # In-memory (dev)
    store = MemoryStore()

    # SQLite (single-instance)
    store = MemoryStore.sqlite("./memories.db")

    # PostgreSQL (production)
    store = MemoryStore.postgres(os.environ["DATABASE_URL"])

    # Store and search memories
    store.put(("user_123", "preferences"), "lang", {"value": "Python"})
    results = store.search(("user_123", "preferences"), "programming language")
    ```

Conversation History RAG:
    ```python
    from ai_infra.memory import ConversationMemory, create_memory_tool

    memory = ConversationMemory(backend="sqlite", path="./conversations.db")

    # Index a conversation
    memory.index_conversation(
        user_id="user_123",
        session_id="session_456",
        messages=[...],
    )

    # Search past conversations
    results = memory.search(user_id="user_123", query="authentication bug")

    # Create tool for agent
    recall_tool = create_memory_tool(memory)
    agent = Agent(tools=[recall_tool])
    ```
"""

# Primary API (6.5)
from ai_infra.llm.memory.context import ContextResult, afit_context, fit_context

# Long-term memory store (6.4.2)
from ai_infra.llm.memory.store import MemoryItem, MemoryStore

# Summarization utilities (6.5+)
from ai_infra.llm.memory.summarize import (
    SummarizationMiddleware,
    SummarizeResult,
    asummarize_messages,
    summarize_messages,
)

# Token utilities (internal, but useful for advanced users)
from ai_infra.llm.memory.tokens import count_tokens, count_tokens_approximate

# Re-export ConversationMemory from tools/custom for convenience
from ai_infra.llm.tools.custom.memory import (
    ConversationChunk,
    ConversationMemory,
    SearchResult,
    create_memory_tool,
    create_memory_tool_async,
)

__all__ = [
    # Primary API (6.5) - This is what most users need
    "fit_context",
    "afit_context",
    "ContextResult",
    # Token utilities (for advanced use)
    "count_tokens",
    "count_tokens_approximate",
    # Summarization
    "summarize_messages",
    "asummarize_messages",
    "SummarizeResult",
    "SummarizationMiddleware",
    # Long-term memory store (6.4.2)
    "MemoryStore",
    "MemoryItem",
    # Conversation memory (6.4.3)
    "ConversationMemory",
    "ConversationChunk",
    "SearchResult",
    "create_memory_tool",
    "create_memory_tool_async",
]
