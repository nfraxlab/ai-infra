from __future__ import annotations

import pytest

from ai_infra.llm.memory.records import (
    MemoryContextPolicy,
    MemoryRecord,
    pack_memory_context,
    rank_memory_records,
)


def test_memory_record_rejects_empty_body() -> None:
    with pytest.raises(ValueError):
        MemoryRecord(id="1", title="Valid", body=" ")


def test_rank_memory_records_uses_lexical_fallback() -> None:
    records = [
        MemoryRecord(id="1", title="Billing design", body="Invoices use Stripe webhooks"),
        MemoryRecord(id="2", title="Editor state", body="Selection state lives in AppState"),
    ]

    ranked = rank_memory_records("invoice webhook", records)

    assert ranked[0].id == "1"
    assert ranked[0].score is not None


def test_pack_memory_context_respects_record_and_token_budget() -> None:
    records = [
        MemoryRecord(
            id="1",
            title="Billing source of truth",
            body="Billing state is reconciled from invoice events.",
            tags=["billing"],
        ),
        MemoryRecord(
            id="2",
            title="Unrelated editor detail",
            body="The editor preview is debounced before rendering markdown.",
        ),
        MemoryRecord(
            id="3",
            title="Billing retry policy",
            body="Webhook retries should be idempotent and observable.",
            tags=["billing"],
        ),
    ]

    pack = pack_memory_context(
        "billing webhook",
        records,
        policy=MemoryContextPolicy(max_records=1, max_tokens=200),
    )

    assert len(pack.records) == 1
    assert pack.records[0].id in {"1", "3"}
    assert "Relevant durable memory" in pack.text
    assert pack.skipped_count == 2


def test_pack_memory_context_returns_empty_when_budget_too_small() -> None:
    records = [MemoryRecord(id="1", title="Large", body="x" * 1000)]

    pack = pack_memory_context(
        "large",
        records,
        policy=MemoryContextPolicy(max_records=3, max_tokens=1),
    )

    assert pack.records == []
    assert pack.text == ""
