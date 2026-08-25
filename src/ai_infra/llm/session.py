"""Session management for agent state persistence and pause/resume.

This module provides a simple API for:
1. Persisting agent conversations across requests (short-term memory)
2. Pausing agent execution and resuming later (HITL workflows)
3. Cross-session memory (long-term memory)

Example - Simple conversation memory:
    ```python
    from ai_infra.llm import Agent
    from ai_infra.llm.session import memory, Session

    # In-memory session (for dev/testing)
    agent = Agent(tools=[...], session=memory())

    # Run with session ID
    result = agent.run("Hi, I'm Bob", session_id="user-123")
    result = agent.run("What's my name?", session_id="user-123")  # Remembers "Bob"
    ```

Example - Persistent sessions with Postgres:
    ```python
    from ai_infra.llm.session import postgres

    # Production: use Postgres
    agent = Agent(
        tools=[...],
        session=postgres("postgresql://..."),
    )
    ```

Example - Pause and resume:
    ```python
    from ai_infra.llm import Agent
    from ai_infra.llm.session import memory

    agent = Agent(
        tools=[dangerous_tool],
        session=memory(),
        pause_before=["dangerous_tool"],  # Pause before this tool
    )

    # Run until pause
    result = agent.run("Delete file.txt", session_id="task-1")

    if result.paused:
        # Agent is paused, waiting for approval
        print(f"Waiting for: {result.pending_action}")

        # Later, resume with approval
        result = agent.resume(
            session_id="task-1",
            approved=True,
        )
    ```
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel

# =============================================================================
# Session Result Models
# =============================================================================


class PendingAction(BaseModel):
    """Information about a paused action waiting for approval."""

    id: str
    """Unique ID for this pending action."""

    action_type: Literal["tool_call", "output_review"]
    """Type of action waiting."""

    tool_name: str | None = None
    """Tool name if action_type is tool_call."""

    args: dict[str, Any] = {}
    """Arguments for the tool call."""

    context: dict[str, Any] = {}
    """Additional context."""

    message: str | None = None
    """Human-readable description of what's pending."""


class SessionResult(BaseModel):
    """Result from an agent run with session support."""

    content: str
    """The agent's response content."""

    paused: bool = False
    """Whether the agent is paused waiting for input."""

    pending_action: PendingAction | None = None
    """Details about pending action if paused."""

    session_id: str
    """The session ID used for this run."""

    messages: list[Any] = []
    """Full message history."""


class ResumeDecision(BaseModel):
    """Decision for resuming a paused agent."""

    approved: bool = True
    """Whether to approve the pending action."""

    modified_args: dict[str, Any] | None = None
    """Modified arguments (if approved with changes)."""

    reason: str | None = None
    """Reason for decision."""


# =============================================================================
# Session Storage Protocol
# =============================================================================


class SessionStorage(Protocol):
    """Protocol for session storage backends.

    Implementations must provide both sync and async methods.
    The simplest implementation is in-memory (for dev/testing).
    Production should use Postgres or Redis.
    """

    def get_checkpointer(self) -> Any:
        """Get the LangGraph checkpointer instance."""
        ...

    def get_store(self) -> Any | None:
        """Get the LangGraph store instance (for cross-session memory)."""
        ...


# =============================================================================
# Built-in Storage Backends
# =============================================================================


class MemoryStorage:
    """In-memory session storage for development and testing.

    State is lost when the process exits. Use for:
    - Local development
    - Unit tests
    - Prototyping

    Example:
        ```python
        from ai_infra.llm.session import memory

        agent = Agent(tools=[...], session=memory())
        ```
    """

    def __init__(self):
        from langgraph.checkpoint.memory import MemorySaver

        self._checkpointer = MemorySaver()
        self._store = None  # Could add InMemoryStore for long-term memory

    def get_checkpointer(self) -> Any:
        return self._checkpointer

    def get_store(self) -> Any | None:
        return self._store


class PostgresStorage:
    """PostgreSQL session storage for production.

    Persists state across process restarts. Use for:
    - Production deployments
    - Multi-instance apps
    - Long-running conversations

    Example:
        ```python
        from ai_infra.llm.session import postgres

        agent = Agent(
            tools=[...],
            session=postgres("postgresql://user:pass@host:5432/db"),
        )
        ```

    Note:
        Requires `langgraph-checkpoint-postgres` package:
        `pip install langgraph-checkpoint-postgres`
    """

    def __init__(
        self,
        connection_string: str,
        *,
        pool_size: int = 5,
        enable_store: bool = False,
    ):
        """Initialize Postgres storage.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            enable_store: Enable cross-session memory store
        """
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PostgreSQL storage requires 'langgraph-checkpoint-postgres'. "
                "Install with: pip install langgraph-checkpoint-postgres"
            ) from e

        self._connection_string = connection_string
        self._pool_size = pool_size
        self._enable_store = enable_store
        self._checkpointer = None
        self._store = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize connections."""
        if self._initialized:
            return

        from langgraph.checkpoint.postgres import PostgresSaver

        self._checkpointer = PostgresSaver.from_conn_string(
            self._connection_string,
        )
        # Call setup to create tables if needed
        self._checkpointer.setup()  # type: ignore[attr-defined]

        if self._enable_store:
            try:
                from langgraph.store.postgres import PostgresStore

                self._store = PostgresStore.from_conn_string(self._connection_string)
                self._store.setup()  # type: ignore[attr-defined]
            except ImportError:
                pass  # Store not available

        self._initialized = True

    def get_checkpointer(self) -> Any:
        self._ensure_initialized()
        return self._checkpointer

    def get_store(self) -> Any | None:
        self._ensure_initialized()
        return self._store


class SQLiteStorage:
    """SQLite session storage for single-instance deployments.

    Persists state to a local file. Use for:
    - Desktop applications
    - Single-server deployments
    - Offline-capable apps

    Example:
        ```python
        from ai_infra.llm.session import sqlite

        agent = Agent(
            tools=[...],
            session=sqlite("./agent_sessions.db"),
        )
        ```

    Note:
        Requires `langgraph-checkpoint-sqlite` package:
        `pip install langgraph-checkpoint-sqlite`
    """

    def __init__(self, path: str = ":memory:"):
        """Initialize SQLite storage.

        Args:
            path: Path to SQLite database file, or ":memory:" for in-memory
        """
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as e:
            raise ImportError(
                "SQLite storage requires 'langgraph-checkpoint-sqlite'. "
                "Install with: pip install langgraph-checkpoint-sqlite"
            ) from e

        import sqlite3

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._conn)
        self._store = None

    def get_checkpointer(self) -> Any:
        return self._checkpointer

    def get_store(self) -> Any | None:
        return self._store


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def memory() -> MemoryStorage:
    """Create an in-memory session storage.

    Use for development, testing, and prototyping.
    State is lost when the process exits.

    Returns:
        MemoryStorage instance

    Example:
        ```python
        from ai_infra.llm.session import memory

        agent = Agent(tools=[...], session=memory())
        result = agent.run("Hello", session_id="user-123")
        ```
    """
    return MemoryStorage()


def postgres(
    connection_string: str,
    *,
    pool_size: int = 5,
    enable_store: bool = False,
) -> PostgresStorage:
    """Create a PostgreSQL session storage.

    Use for production deployments with persistence.

    Args:
        connection_string: PostgreSQL connection string
        pool_size: Connection pool size
        enable_store: Enable cross-session memory store

    Returns:
        PostgresStorage instance

    Example:
        ```python
        from ai_infra.llm.session import postgres

        agent = Agent(
            tools=[...],
            session=postgres("postgresql://user:pass@host:5432/db"),
        )
        ```
    """
    return PostgresStorage(
        connection_string,
        pool_size=pool_size,
        enable_store=enable_store,
    )


def sqlite(path: str = ":memory:") -> SQLiteStorage:
    """Create a SQLite session storage.

    Use for single-instance deployments or desktop apps.

    Args:
        path: Path to SQLite database file, or ":memory:" for in-memory

    Returns:
        SQLiteStorage instance

    Example:
        ```python
        from ai_infra.llm.session import sqlite

        agent = Agent(
            tools=[...],
            session=sqlite("./sessions.db"),
        )
        ```
    """
    return SQLiteStorage(path)


# =============================================================================
# Session Configuration
# =============================================================================


@dataclass
class SessionConfig:
    """Configuration for session-aware agent execution.

    This is used internally by the Agent class to manage sessions.

    The max_messages parameter prevents unbounded memory growth by trimming
    old messages when the limit is exceeded.
    """

    storage: SessionStorage
    """Storage backend for session state."""

    pause_before: list[str] = field(default_factory=list)
    """Tool names to pause before executing."""

    pause_after: list[str] = field(default_factory=list)
    """Tool names to pause after executing."""

    max_messages: int | None = 100
    """Maximum number of messages to retain in session history.

    When exceeded, oldest messages are trimmed (keeping system message).
    Set to None for unlimited (not recommended for production).
    Default: 100 messages.
    """

    max_tokens: int | None = None
    """Maximum total tokens to retain in session history.

    When exceeded, oldest messages are trimmed until under limit.
    Set to None for unlimited (uses max_messages limit only).
    Note: Requires token counting which may add latency.
    """

    trim_strategy: Literal["last", "summarize"] = "last"
    """Strategy for trimming when limits are exceeded.

    - "last": Keep the most recent messages (default, fast)
    - "summarize": Summarize old messages before dropping (slower, preserves context)
    """

    def get_config(self, session_id: str) -> dict[str, Any]:
        """Get LangGraph config for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Config dict with thread_id for LangGraph
        """
        return {"configurable": {"thread_id": session_id}}


# =============================================================================
# Session State Utilities
# =============================================================================


def generate_session_id() -> str:
    """Generate a unique session ID.

    Returns:
        UUID string for use as session_id
    """
    return str(uuid.uuid4())


def is_paused(result: Any) -> bool:
    """Check if an agent result indicates a paused state.

    Args:
        result: Result from agent.run() or agent.arun()

    Returns:
        True if agent is paused waiting for input
    """
    if isinstance(result, SessionResult):
        return result.paused

    # Check for LangGraph interrupt indicator
    if isinstance(result, dict) and "__interrupt__" in result:
        return True

    return False


def get_pending_action(result: Any) -> PendingAction | None:
    """Extract pending action from a paused result.

    Args:
        result: Result from agent.run() or agent.arun()

    Returns:
        PendingAction if paused, None otherwise
    """
    if isinstance(result, SessionResult):
        return result.pending_action

    # Parse LangGraph interrupt
    if isinstance(result, dict) and "__interrupt__" in result:
        interrupt_data = result["__interrupt__"]
        if interrupt_data:
            first = interrupt_data[0]
            value = getattr(first, "value", first) if hasattr(first, "value") else first
            if isinstance(value, dict):
                action_requests = value.get("action_requests")
                if isinstance(action_requests, list) and action_requests:
                    action_request = action_requests[0]
                    if isinstance(action_request, dict):
                        return PendingAction(
                            id=getattr(first, "id", str(uuid.uuid4())),
                            action_type="tool_call",
                            tool_name=action_request.get("name"),
                            args=action_request.get("args", {}),
                            message=action_request.get("description"),
                        )
                return PendingAction(
                    id=value.get("id", str(uuid.uuid4())),
                    action_type=value.get("action_type", "tool_call"),
                    tool_name=value.get("tool_name"),
                    args=value.get("args", {}),
                    context=value.get("context", {}),
                    message=value.get("message"),
                )
    return None


def trim_messages(
    messages: list[dict[str, Any]],
    max_messages: int | None = 100,
    keep_system: bool = True,
) -> list[dict[str, Any]]:
    """Trim messages to prevent unbounded history growth.

    Removes oldest messages (except system) when limit is exceeded.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_messages: Maximum messages to keep (None = unlimited)
        keep_system: Whether to always preserve system message at start

    Returns:
        Trimmed list of messages

    Example:
        >>> messages = [{"role": "system", "content": "..."}, ...]
        >>> trimmed = trim_messages(messages, max_messages=50)
    """
    if max_messages is None or len(messages) <= max_messages:
        return messages

    # Separate system message if present and keep_system is True
    system_msg = None
    if keep_system and messages and messages[0].get("role") == "system":
        system_msg = messages[0]
        messages = messages[1:]
        # Adjust limit to account for system message
        max_messages = max_messages - 1

    # Keep the most recent messages
    trimmed = messages[-max_messages:] if max_messages > 0 else []

    # Prepend system message if it was preserved
    if system_msg is not None:
        trimmed = [system_msg] + trimmed

    return trimmed
