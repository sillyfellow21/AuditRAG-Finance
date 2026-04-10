from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from core.cache import TTLCache
from core.models import RetrievedContext
from embeddings.store import ChromaVectorStore
from monitoring.observability import Observability


class RetrievalPipeline:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        observability: Observability,
        default_top_k: int = 5,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 180,
        cache_max_entries: int = 1024,
    ) -> None:
        self.vector_store = vector_store
        self.observability = observability
        self.default_top_k = default_top_k
        self.cache_enabled = cache_enabled
        self._cache: Optional[TTLCache[tuple[str, str, int], List[RetrievedContext]]] = None
        if cache_enabled:
            self._cache = TTLCache(
                max_entries=cache_max_entries,
                ttl_seconds=cache_ttl_seconds,
            )

    def retrieve_context(
        self,
        document_id: str,
        question: str,
        fallback_chunks: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedContext]:
        contexts, _ = self.retrieve_with_diagnostics(
            document_id=document_id,
            question=question,
            fallback_chunks=fallback_chunks,
            top_k=top_k,
        )
        return contexts

    def retrieve_with_diagnostics(
        self,
        document_id: str,
        question: str,
        fallback_chunks: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievedContext], Dict[str, Any]]:
        requested_k = top_k or self.default_top_k
        cache_key = (document_id, question.strip().lower(), requested_k)

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.observability.log_event(
                    "retrieval.cache_hit",
                    {"document_id": document_id, "count": len(cached)},
                )
                return cached, {
                    "source": "cache",
                    "count": len(cached),
                    "requested_k": requested_k,
                    "cache_hit": True,
                    "top_score": max((ctx.score for ctx in cached), default=0.0),
                }

        vector_hits = self.vector_store.retrieve(
            document_id=document_id,
            question=question,
            top_k=requested_k,
        )
        if vector_hits:
            if self._cache is not None:
                self._cache.set(cache_key, vector_hits)
            self.observability.log_event(
                "retrieval.vector_hits",
                {"document_id": document_id, "count": len(vector_hits)},
            )
            return vector_hits, {
                "source": "vector",
                "count": len(vector_hits),
                "requested_k": requested_k,
                "cache_hit": False,
                "top_score": max((ctx.score for ctx in vector_hits), default=0.0),
            }

        fallback_hits: List[RetrievedContext] = []
        if fallback_chunks:
            fallback_hits = self._lexical_fallback(question, fallback_chunks, requested_k)
            self.observability.log_event(
                "retrieval.lexical_hits",
                {"document_id": document_id, "count": len(fallback_hits)},
            )
            if self._cache is not None:
                self._cache.set(cache_key, fallback_hits)

        return fallback_hits, {
            "source": "lexical_fallback",
            "count": len(fallback_hits),
            "requested_k": requested_k,
            "cache_hit": False,
            "top_score": max((ctx.score for ctx in fallback_hits), default=0.0),
        }

    def build_evidence(self, contexts: List[RetrievedContext], max_chars: int = 280) -> List[str]:
        evidence: List[str] = []
        for ctx in contexts:
            snippet = ctx.text.strip().replace("\n", " ")
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3].rstrip() + "..."
            evidence.append(snippet)
        return evidence

    def _lexical_fallback(
        self,
        question: str,
        chunks: List[str],
        top_k: int,
    ) -> List[RetrievedContext]:
        query_tokens = set(self._tokenize(question))
        scored: List[RetrievedContext] = []

        for idx, chunk in enumerate(chunks):
            chunk_tokens = set(self._tokenize(chunk))
            if not chunk_tokens:
                continue

            overlap = len(query_tokens.intersection(chunk_tokens))
            score = overlap / max(len(query_tokens), 1)
            if score <= 0:
                continue

            scored.append(
                RetrievedContext(
                    text=chunk,
                    score=score,
                    metadata={"chunk_index": str(idx), "source": "lexical_fallback"},
                )
            )

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())
