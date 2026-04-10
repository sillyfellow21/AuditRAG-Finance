from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Generator, Optional

from core.config import Settings


@dataclass
class TokenUsage:
    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    metadata: Dict[str, Any]


class Observability:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger("financial_assistant")
        self.logger.setLevel(settings.log_level.upper())
        self.log_path = settings.data_dir / "app.log"
        self.usage_path = settings.data_dir / "token_usage.jsonl"
        self.audit_path = settings.data_dir / "audit_log.jsonl"
        self._audit_lock = Lock()
        self._configure_logger()
        self._configure_langsmith()

    def _configure_logger(self) -> None:
        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def _configure_langsmith(self) -> None:
        tracing_enabled = (
            self.settings.langchain_tracing_v2
            and self.settings.has_langsmith
            and self.settings.langchain_api_key is not None
        )
        os.environ["LANGCHAIN_TRACING_V2"] = str(tracing_enabled).lower()
        os.environ["LANGCHAIN_PROJECT"] = self.settings.langchain_project
        os.environ["LANGSMITH_ENDPOINT"] = self.settings.langsmith_endpoint
        if tracing_enabled:
            os.environ["LANGCHAIN_API_KEY"] = self.settings.langchain_api_key.get_secret_value()
        else:
            os.environ.pop("LANGCHAIN_API_KEY", None)

    def new_request_id(self) -> str:
        return str(uuid.uuid4())

    def log_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        data = payload or {}
        self.logger.info("event=%s payload=%s", event, json.dumps(data, ensure_ascii=True))

    def log_warning(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        data = payload or {}
        self.logger.warning("event=%s payload=%s", event, json.dumps(data, ensure_ascii=True))

    def log_exception(self, event: str, exc: Exception, payload: Optional[Dict[str, Any]] = None) -> None:
        data = payload or {}
        self.logger.exception(
            "event=%s error=%s payload=%s",
            event,
            str(exc),
            json.dumps(data, ensure_ascii=True),
        )

    def record_token_usage(self, usage: TokenUsage) -> None:
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(usage)
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.usage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    def audit_event(
        self,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        request_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        event_id = str(uuid.uuid4())
        row = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "resource": resource,
            "outcome": outcome,
            "request_id": request_id,
            "payload": payload or {},
        }
        with self._audit_lock:
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        return event_id

    def read_audit_logs(self, limit: int = 50) -> list[Dict[str, Any]]:
        safe_limit = max(1, min(limit, 500))
        if not self.audit_path.exists():
            return []

        with self._audit_lock:
            lines = self.audit_path.read_text(encoding="utf-8").splitlines()

        rows: list[Dict[str, Any]] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            if len(rows) >= safe_limit:
                break

        rows.reverse()
        return rows

    @contextmanager
    def track_latency(
        self,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        start = time.perf_counter()
        state: Dict[str, Any] = payload.copy() if payload else {}
        try:
            yield state
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            state["latency_ms"] = round(elapsed_ms, 2)
            self.log_event(event=event, payload=state)

    @staticmethod
    def extract_token_usage(message: Any) -> Dict[str, int]:
        usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if message is None:
            return usage

        usage_meta = getattr(message, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            usage["prompt_tokens"] = int(usage_meta.get("input_tokens", 0))
            usage["completion_tokens"] = int(usage_meta.get("output_tokens", 0))
            usage["total_tokens"] = int(usage_meta.get("total_tokens", 0))
            return usage

        response_meta = getattr(message, "response_metadata", None)
        if isinstance(response_meta, dict):
            token_usage = response_meta.get("token_usage", {})
            if isinstance(token_usage, dict):
                usage["prompt_tokens"] = int(token_usage.get("prompt_tokens", 0))
                usage["completion_tokens"] = int(token_usage.get("completion_tokens", 0))
                usage["total_tokens"] = int(token_usage.get("total_tokens", 0))

        return usage
