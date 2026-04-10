from __future__ import annotations

from core.models import RetrievedContext
from retrieval.pipeline import RetrievalPipeline


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, document_id: str, question: str, top_k: int) -> list[RetrievedContext]:
        self.calls += 1
        return [
            RetrievedContext(
                text="Grand total is 120.00",
                score=0.91,
                metadata={"document_id": document_id, "top_k": top_k},
            )
        ]


def test_retrieval_cache_hits(observability) -> None:
    vector_store = FakeVectorStore()
    pipeline = RetrievalPipeline(
        vector_store=vector_store,
        observability=observability,
        default_top_k=3,
        cache_enabled=True,
        cache_ttl_seconds=60,
        cache_max_entries=10,
    )

    first_contexts, first_diag = pipeline.retrieve_with_diagnostics(
        document_id="doc-1",
        question="what is the total",
        fallback_chunks=["fallback chunk"],
        top_k=3,
    )
    second_contexts, second_diag = pipeline.retrieve_with_diagnostics(
        document_id="doc-1",
        question="what is the total",
        fallback_chunks=["fallback chunk"],
        top_k=3,
    )

    assert len(first_contexts) == 1
    assert len(second_contexts) == 1
    assert first_diag["cache_hit"] is False
    assert second_diag["cache_hit"] is True
    assert vector_store.calls == 1
