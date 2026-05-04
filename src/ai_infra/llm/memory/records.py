"""Generic memory records and token-bounded context packing.

Applications decide where memories are stored and how they are curated. This
module provides the provider-agnostic shape and ranking/packing behavior used
to pass durable memory back into an LLM context window.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ai_infra.llm.memory.tokens import count_tokens_approximate

MemoryKind = Literal["fact", "decision", "preference", "instruction", "summary", "context", "other"]
MemorySource = Literal["user", "agent", "conversation", "import", "system"]


class MemoryRecord(BaseModel):
    """Portable representation of a durable memory item.

    The model is intentionally storage-neutral. Database IDs, workspace IDs,
    project IDs, conversation IDs, or application-specific fields should be
    carried in ``metadata`` rather than becoming SDK concepts.
    """

    id: str
    namespace: tuple[str, ...] = Field(default_factory=tuple)
    title: str
    body: str
    kind: MemoryKind = "context"
    source: MemorySource = "agent"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    expires_at: float | None = None
    score: float | None = None

    @field_validator("title", "body")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("memory text fields cannot be empty")
        return cleaned

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_tags = value.split(",")
        else:
            raw_tags = list(value)
        tags: list[str] = []
        for tag in raw_tags:
            cleaned = str(tag).strip().lower()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        return tags

    @property
    def token_count(self) -> int:
        """Approximate token count for the record as context."""
        return count_tokens_approximate([self.as_context_text(include_metadata=False)])

    def as_context_text(self, *, include_metadata: bool = True) -> str:
        """Render the memory as compact Markdown-like context."""
        parts = [f"### {self.title}", self.body]
        labels: list[str] = [self.kind, self.source]
        if self.tags:
            labels.append("tags=" + ",".join(self.tags))
        if include_metadata and self.metadata:
            scope_parts = []
            for key in ("workspace_id", "project_id", "conversation_id", "source_ref"):
                value = self.metadata.get(key)
                if value:
                    scope_parts.append(f"{key}={value}")
            if scope_parts:
                labels.append(" ".join(scope_parts))
        parts.append(f"[{'; '.join(labels)}]")
        return "\n".join(parts)


class MemoryContextPolicy(BaseModel):
    """Controls how memory records are ranked and packed into context."""

    max_records: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=1600, ge=1)
    min_score: float | None = Field(default=None, ge=0.0)
    include_header: bool = True
    header: str = "Relevant durable memory"
    include_metadata: bool = True


class MemoryContextPack(BaseModel):
    """Token-bounded memory context selected for an LLM turn."""

    records: list[MemoryRecord]
    text: str
    tokens: int
    skipped_count: int = 0


_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")


def _query_terms(query: str) -> set[str]:
    return {term.lower() for term in _TERM_RE.findall(query) if len(term) > 1}


def _lexical_score(query: str, record: MemoryRecord) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0

    title = record.title.lower()
    body = record.body.lower()
    tags = {tag.lower() for tag in record.tags}
    weighted_hits = 0.0
    for term in terms:
        if term in title:
            weighted_hits += 2.0
        if term in body:
            weighted_hits += 1.0
        if term in tags:
            weighted_hits += 1.5
    return weighted_hits / max(1.0, len(terms) * 2.0)


def rank_memory_records(query: str, records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Rank records using existing scores plus deterministic lexical fallback."""
    now = time.time()
    ranked: list[MemoryRecord] = []
    for record in records:
        if record.expires_at is not None and record.expires_at <= now:
            continue
        lexical = _lexical_score(query, record)
        base_score = record.score if record.score is not None else lexical
        score = max(base_score, lexical)
        ranked.append(record.model_copy(update={"score": score}))
    ranked.sort(key=lambda item: (item.score or 0.0, item.updated_at), reverse=True)
    return ranked


def pack_memory_context(
    query: str,
    records: list[MemoryRecord],
    *,
    policy: MemoryContextPolicy | None = None,
) -> MemoryContextPack:
    """Rank records and pack them into a token-bounded context block."""
    active_policy = policy or MemoryContextPolicy()
    selected: list[MemoryRecord] = []
    lines: list[str] = []
    if active_policy.include_header:
        lines.append(f"## {active_policy.header}")

    budget = active_policy.max_tokens
    used_tokens = count_tokens_approximate(lines) if lines else 0
    skipped = 0

    for record in rank_memory_records(query, records):
        if len(selected) >= active_policy.max_records:
            skipped += 1
            continue
        if active_policy.min_score is not None and (record.score or 0.0) < active_policy.min_score:
            skipped += 1
            continue
        block = record.as_context_text(include_metadata=active_policy.include_metadata)
        block_tokens = count_tokens_approximate([block])
        if selected and used_tokens + block_tokens > budget:
            skipped += 1
            continue
        if not selected and used_tokens + block_tokens > budget:
            skipped += 1
            continue
        selected.append(record)
        lines.append(block)
        used_tokens += block_tokens

    text = "\n\n".join(lines).strip() if selected else ""
    return MemoryContextPack(
        records=selected,
        text=text,
        tokens=count_tokens_approximate([text]) if text else 0,
        skipped_count=skipped,
    )
