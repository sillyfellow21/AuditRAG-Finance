from __future__ import annotations

from datetime import datetime, timezone

from core.models import DocumentRecord, ExtractedFields, LineItem, RetrievedContext
from reasoning.engine import ReasoningEngine


class FakeRetrieval:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve_with_diagnostics(
        self,
        document_id: str,
        question: str,
        fallback_chunks: list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedContext], dict[str, object]]:
        self.calls += 1
        contexts = [RetrievedContext(text="Total: 140.00", score=0.88, metadata={"doc": document_id})]
        diagnostics: dict[str, object] = {
            "source": "vector",
            "count": 1,
            "requested_k": top_k or 5,
            "cache_hit": False,
            "top_score": 0.88,
        }
        return contexts, diagnostics

    def build_evidence(self, contexts: list[RetrievedContext], max_chars: int = 280) -> list[str]:
        return [ctx.text[:max_chars] for ctx in contexts]


class FakeRepository:
    def __init__(self, record: DocumentRecord) -> None:
        self.record = record

    def get_document(self, document_id: str) -> DocumentRecord | None:
        if self.record.document_id == document_id:
            return self.record
        return None

    def list_recent_totals(self, limit: int = 250) -> list[float]:
        return [90.0, 95.0, 100.0, 105.0, 98.0, 102.0, 97.0][:limit]

    def list_recent_charge_signatures(self, limit: int = 250) -> list[dict[str, object]]:
        return [
            {"vendor": "Acme", "total": 140.0, "date": "2026-03-10", "document_id": "doc-0"},
            {"vendor": "Acme", "total": 140.0, "date": "2026-02-10", "document_id": "doc-x"},
        ][:limit]


def test_reasoning_engine_response_cache_and_explainability(
    test_settings,
    observability,
) -> None:
    record = DocumentRecord(
        document_id="doc-1",
        filename="invoice.pdf",
        file_path="invoice.pdf",
        text_path="invoice.txt",
        uploaded_at=datetime.now(timezone.utc),
        parser_used="pypdf",
        extracted_data=ExtractedFields(
            vendor="Acme",
            date="2026-04-10",
            total=140.0,
            tax=45.0,
            line_items=[
                LineItem(description="Item A", amount=30.0),
                LineItem(description="Item B", amount=20.0),
            ],
        ),
        chunks=["Total: 140.00"],
    )

    retrieval = FakeRetrieval()
    repository = FakeRepository(record)
    engine = ReasoningEngine(
        settings=test_settings,
        retrieval=retrieval,
        repository=repository,
        observability=observability,
    )

    first = engine.answer_question("doc-1", "Is this a suspicious charge?")
    second = engine.answer_question("doc-1", "Is this a suspicious charge?")

    assert first.category == "anomaly"
    assert first.anomaly is not None
    assert "suspicious" in first.flags
    assert first.explainability.cache["answer_cache_hit"] is False
    assert second.explainability.cache["answer_cache_hit"] is True
    assert retrieval.calls == 1
