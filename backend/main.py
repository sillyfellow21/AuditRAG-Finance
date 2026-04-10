from __future__ import annotations

import asyncio

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.models import (
    AnswerResponse,
    AskRequest,
    AuditLogEntry,
    AuditLogResponse,
    DocumentRecord,
    ErrorResponse,
    UploadResponse,
)
from embeddings.store import ChromaVectorStore
from extraction.extractor import FinancialExtractor
from guardrails.finance import block_non_finance, finance_classifier
from ingestion.service import FileDocumentRepository, IngestionService
from monitoring.observability import Observability
from parsing.parser import FinancialDocumentParser
from reasoning.engine import ReasoningEngine
from retrieval.pipeline import RetrievalPipeline

settings = get_settings()
observability = Observability(settings=settings)

parser = FinancialDocumentParser(settings=settings, observability=observability)
extractor = FinancialExtractor(settings=settings, observability=observability)
vector_store = ChromaVectorStore(settings=settings, observability=observability)
repository = FileDocumentRepository(settings=settings, observability=observability)
retrieval = RetrievalPipeline(
    vector_store=vector_store,
    observability=observability,
    default_top_k=settings.rag_top_k,
    cache_enabled=settings.retrieval_cache_enabled,
    cache_ttl_seconds=settings.retrieval_cache_ttl_seconds,
    cache_max_entries=settings.retrieval_cache_max_entries,
)
ingestion_service = IngestionService(
    settings=settings,
    parser=parser,
    extractor=extractor,
    vector_store=vector_store,
    repository=repository,
    observability=observability,
)
reasoning_engine = ReasoningEngine(
    settings=settings,
    retrieval=retrieval,
    repository=repository,
    observability=observability,
)

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    request_id = observability.new_request_id()
    try:
        observability.audit_event(
            actor="api",
            action="document_upload",
            resource=file.filename or "unknown",
            outcome="started",
            request_id=request_id,
        )
        result = await ingestion_service.ingest_upload(file)
        observability.audit_event(
            actor="api",
            action="document_upload",
            resource=result.document_id,
            outcome="success",
            request_id=request_id,
            payload={
                "filename": result.filename,
                "indexed_chunks": result.indexed_chunks,
                "anomaly_risk": result.anomaly.risk_level if result.anomaly else None,
            },
        )
        return result
    except ValueError as exc:
        observability.audit_event(
            actor="api",
            action="document_upload",
            resource=file.filename or "unknown",
            outcome="failed",
            request_id=request_id,
            payload={"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        observability.log_exception("api.upload.failed", exc)
        observability.audit_event(
            actor="api",
            action="document_upload",
            resource=file.filename or "unknown",
            outcome="failed",
            request_id=request_id,
            payload={"error": str(exc)},
        )
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc
    finally:
        await file.close()


@app.post("/api/ask", response_model=AnswerResponse | ErrorResponse)
async def ask_question(payload: AskRequest) -> AnswerResponse | ErrorResponse:
    request_id = observability.new_request_id()
    try:
        if settings.finance_guardrails_enabled:
            guardrail_error = block_non_finance(payload.question)
            if guardrail_error is not None:
                observability.audit_event(
                    actor="guardrails",
                    action="question_blocked",
                    resource=payload.document_id,
                    outcome="blocked",
                    request_id=request_id,
                    payload={
                        "question": payload.question,
                        "classifier": finance_classifier(payload.question),
                    },
                )
                return ErrorResponse.model_validate(guardrail_error)

        observability.audit_event(
            actor="api",
            action="question_answer",
            resource=payload.document_id,
            outcome="started",
            request_id=request_id,
            payload={"question": payload.question},
        )
        result = await asyncio.to_thread(
            reasoning_engine.answer_question,
            payload.document_id,
            payload.question,
        )
        observability.audit_event(
            actor="api",
            action="question_answer",
            resource=payload.document_id,
            outcome="success",
            request_id=request_id,
            payload={
                "category": result.category,
                "confidence": result.confidence,
                "anomaly_risk": result.anomaly.risk_level if result.anomaly else None,
                "flags": result.flags,
            },
        )
        return result
    except ValueError as exc:
        observability.audit_event(
            actor="api",
            action="question_answer",
            resource=payload.document_id,
            outcome="failed",
            request_id=request_id,
            payload={"error": str(exc)},
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        observability.log_exception("api.ask.failed", exc, {"document_id": payload.document_id})
        observability.audit_event(
            actor="api",
            action="question_answer",
            resource=payload.document_id,
            outcome="failed",
            request_id=request_id,
            payload={"error": str(exc)},
        )
        raise HTTPException(status_code=500, detail="Question answering failed.") from exc


@app.get("/api/document/{document_id}", response_model=DocumentRecord)
async def get_document(document_id: str) -> DocumentRecord:
    record = await ingestion_service.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    observability.audit_event(
        actor="api",
        action="document_read",
        resource=document_id,
        outcome="success",
    )
    return record


@app.get("/api/audit-logs", response_model=AuditLogResponse)
async def get_audit_logs(limit: int = Query(default=50, ge=1, le=500)) -> AuditLogResponse:
    rows = await asyncio.to_thread(observability.read_audit_logs, limit)
    logs = [AuditLogEntry.model_validate(row) for row in rows]
    return AuditLogResponse(count=len(logs), logs=logs)
