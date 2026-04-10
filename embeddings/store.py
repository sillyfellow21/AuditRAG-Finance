from __future__ import annotations

from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import Settings
from core.models import RetrievedContext
from monitoring.observability import Observability


class ChromaVectorStore:
    def __init__(self, settings: Settings, observability: Observability) -> None:
        self.settings = settings
        self.observability = observability

        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.db = Chroma(
            collection_name="financial_documents",
            embedding_function=self.embedding_fn,
            persist_directory=str(settings.chroma_dir),
        )

    def index_document(
        self,
        document_id: str,
        chunks: List[str],
        base_metadata: Optional[Dict[str, str]] = None,
    ) -> int:
        documents: List[Document] = []
        ids: List[str] = []

        metadata_template: Dict[str, str] = base_metadata.copy() if base_metadata else {}
        metadata_template["document_id"] = document_id

        for idx, chunk in enumerate(chunks):
            metadata = metadata_template.copy()
            metadata["chunk_index"] = str(idx)
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
            ids.append(f"{document_id}-{idx}")

        if documents:
            self.db.add_documents(documents=documents, ids=ids)
            self.observability.log_event(
                "embeddings.indexed",
                {"document_id": document_id, "chunk_count": len(documents)},
            )

        return len(documents)

    def retrieve(
        self,
        document_id: str,
        question: str,
        top_k: int,
    ) -> List[RetrievedContext]:
        try:
            pairs = self.db.similarity_search_with_relevance_scores(
                question,
                k=top_k,
                filter={"document_id": document_id},
            )
            results = [
                RetrievedContext(
                    text=doc.page_content,
                    score=float(max(0.0, min(1.0, score))),
                    metadata=doc.metadata,
                )
                for doc, score in pairs
            ]
            return results
        except Exception as exc:
            self.observability.log_warning(
                "retrieval.relevance_failed",
                {"document_id": document_id, "error": str(exc)},
            )

        # Backward-compatible fallback where lower distance means closer.
        try:
            pairs = self.db.similarity_search_with_score(
                question,
                k=top_k,
                filter={"document_id": document_id},
            )
            normalized: List[RetrievedContext] = []
            for doc, distance in pairs:
                score = 1.0 / (1.0 + float(distance))
                normalized.append(
                    RetrievedContext(
                        text=doc.page_content,
                        score=score,
                        metadata=doc.metadata,
                    )
                )
            return normalized
        except Exception as exc:
            self.observability.log_warning(
                "retrieval.score_failed",
                {"document_id": document_id, "error": str(exc)},
            )
            return []

    def delete_document(self, document_id: str) -> int:
        payload = self.db.get(where={"document_id": document_id})
        ids = payload.get("ids", []) if isinstance(payload, dict) else []
        if ids:
            self.db.delete(ids=ids)
        deleted = len(ids)
        self.observability.log_event(
            "embeddings.deleted",
            {"document_id": document_id, "deleted": deleted},
        )
        return deleted
