from __future__ import annotations

import asyncio
from tempfile import SpooledTemporaryFile
from typing import Any, Dict

import requests
from fastapi import UploadFile

from core.models import ErrorResponse
from guardrails.finance import block_non_finance, finance_classifier


class BackendClient:
    def __init__(self, base_url: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def upload_document(self, filename: str, content: bytes, mime_type: str) -> Dict[str, Any]:
        files = {"file": (filename, content, mime_type)}
        response = requests.post(
            f"{self.base_url}/api/upload",
            files=files,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def ask_question(self, document_id: str, question: str) -> Dict[str, Any]:
        payload = {"document_id": document_id, "question": question}
        response = requests.post(
            f"{self.base_url}/api/ask",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def get_document(self, document_id: str) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/document/{document_id}",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def get_audit_logs(self, limit: int = 50) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/audit-logs",
            params={"limit": limit},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


class InProcessBackendClient:
    """Run the backend pipeline in-process for single-service Streamlit deployments."""

    def __init__(self) -> None:
        # Import lazily so this client can coexist with HTTP mode.
        from backend.main import ingestion_service, observability, reasoning_engine, settings

        self.settings = settings
        self.ingestion_service = ingestion_service
        self.reasoning_engine = reasoning_engine
        self.observability = observability

    @staticmethod
    def _run_async(coro: Any) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "environment": self.settings.app_env,
            "mode": "in_process",
        }

    def upload_document(self, filename: str, content: bytes, mime_type: str) -> Dict[str, Any]:
        _ = mime_type  # Reserved for parity with HTTP client signature.
        request_id = self.observability.new_request_id()
        self.observability.audit_event(
            actor="streamlit",
            action="document_upload",
            resource=filename or "unknown",
            outcome="started",
            request_id=request_id,
        )

        file_handle = SpooledTemporaryFile(max_size=8 * 1024 * 1024)
        file_handle.write(content)
        file_handle.seek(0)
        upload_file = UploadFile(file=file_handle, filename=filename)

        try:
            result = self._run_async(self.ingestion_service.ingest_upload(upload_file))
            self.observability.audit_event(
                actor="streamlit",
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
            return result.model_dump(mode="json")
        except Exception as exc:
            self.observability.audit_event(
                actor="streamlit",
                action="document_upload",
                resource=filename or "unknown",
                outcome="failed",
                request_id=request_id,
                payload={"error": str(exc)},
            )
            raise
        finally:
            file_handle.close()

    def ask_question(self, document_id: str, question: str) -> Dict[str, Any]:
        request_id = self.observability.new_request_id()
        if self.settings.finance_guardrails_enabled:
            guardrail_error = block_non_finance(question)
            if guardrail_error is not None:
                self.observability.audit_event(
                    actor="guardrails",
                    action="question_blocked",
                    resource=document_id,
                    outcome="blocked",
                    request_id=request_id,
                    payload={"question": question, "classifier": finance_classifier(question)},
                )
                return ErrorResponse.model_validate(guardrail_error).model_dump(mode="json")

        self.observability.audit_event(
            actor="streamlit",
            action="question_answer",
            resource=document_id,
            outcome="started",
            request_id=request_id,
            payload={"question": question},
        )
        try:
            response = self.reasoning_engine.answer_question(document_id, question)
            self.observability.audit_event(
                actor="streamlit",
                action="question_answer",
                resource=document_id,
                outcome="success",
                request_id=request_id,
                payload={
                    "category": response.category,
                    "confidence": response.confidence,
                    "anomaly_risk": response.anomaly.risk_level if response.anomaly else None,
                    "flags": response.flags,
                },
            )
            return response.model_dump(mode="json")
        except Exception as exc:
            self.observability.audit_event(
                actor="streamlit",
                action="question_answer",
                resource=document_id,
                outcome="failed",
                request_id=request_id,
                payload={"error": str(exc)},
            )
            raise

    def get_document(self, document_id: str) -> Dict[str, Any]:
        record = self._run_async(self.ingestion_service.get_document(document_id))
        if record is None:
            raise ValueError("Document not found.")
        self.observability.audit_event(
            actor="streamlit",
            action="document_read",
            resource=document_id,
            outcome="success",
        )
        return record.model_dump(mode="json")

    def get_audit_logs(self, limit: int = 50) -> Dict[str, Any]:
        rows = self.observability.read_audit_logs(limit=limit)
        return {"count": len(rows), "logs": rows}
