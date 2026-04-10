from __future__ import annotations

from typing import Any, Dict

import requests


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
