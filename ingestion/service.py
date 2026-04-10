from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.cache import TTLCache
from core.config import Settings
from core.models import DocumentRecord, ExtractedFields, UploadResponse
from embeddings.store import ChromaVectorStore
from extraction.extractor import FinancialExtractor
from monitoring.observability import Observability
from parsing.parser import FinancialDocumentParser
from reasoning.anomaly import ChargeAnomalyDetector


class FileDocumentRepository:
    def __init__(self, settings: Settings, observability: Observability) -> None:
        self.settings = settings
        self.observability = observability
        self._document_cache: TTLCache[str, DocumentRecord] = TTLCache(
            max_entries=settings.repository_cache_max_entries,
            ttl_seconds=settings.repository_cache_ttl_seconds,
        )

    def metadata_path(self, document_id: str) -> Path:
        return self.settings.document_dir / f"{document_id}.json"

    def text_path(self, document_id: str) -> Path:
        return self.settings.document_dir / f"{document_id}.txt"

    def save_document(self, record: DocumentRecord, text: str) -> None:
        metadata_path = self.metadata_path(record.document_id)
        text_path = self.text_path(record.document_id)

        text_path.write_text(text, encoding="utf-8")

        payload = record.model_dump(mode="json")
        metadata_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        self._document_cache.set(record.document_id, record)

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        cached = self._document_cache.get(document_id)
        if cached is not None:
            return cached

        metadata_path = self.metadata_path(document_id)
        if not metadata_path.exists():
            return None

        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = DocumentRecord.model_validate(data)
        self._document_cache.set(document_id, record)
        return record

    def list_recent_totals(self, limit: int = 250) -> list[float]:
        totals: list[float] = []
        metadata_files = sorted(
            self.settings.document_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for metadata_path in metadata_files[: max(limit, 1)]:
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            extracted = payload.get("extracted_data", {}) if isinstance(payload, dict) else {}
            total = extracted.get("total") if isinstance(extracted, dict) else None
            if isinstance(total, (int, float)) and total > 0:
                totals.append(float(total))
        return totals

    def list_recent_charge_signatures(self, limit: int = 250) -> list[dict[str, object]]:
        signatures: list[dict[str, object]] = []
        metadata_files = sorted(
            self.settings.document_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for metadata_path in metadata_files[: max(limit, 1)]:
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            extracted = payload.get("extracted_data")
            if not isinstance(extracted, dict):
                continue

            total = extracted.get("total")
            if not isinstance(total, (int, float)):
                continue

            signatures.append(
                {
                    "vendor": extracted.get("vendor"),
                    "total": float(total),
                    "date": extracted.get("date"),
                    "document_id": payload.get("document_id"),
                }
            )
        return signatures


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        parser: FinancialDocumentParser,
        extractor: FinancialExtractor,
        vector_store: ChromaVectorStore,
        repository: FileDocumentRepository,
        observability: Observability,
    ) -> None:
        self.settings = settings
        self.parser = parser
        self.extractor = extractor
        self.vector_store = vector_store
        self.repository = repository
        self.observability = observability
        self.anomaly_detector = ChargeAnomalyDetector(
            high_total_threshold=settings.anomaly_high_total_threshold,
            high_tax_ratio=settings.anomaly_high_tax_ratio,
            mismatch_ratio_threshold=settings.anomaly_line_total_mismatch_ratio,
            robust_outlier_z_threshold=settings.anomaly_robust_z_threshold,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def ingest_upload(self, upload_file: UploadFile) -> UploadResponse:
        request_id = self.observability.new_request_id()
        with self.observability.track_latency(
            "ingestion.completed",
            {"request_id": request_id, "filename": upload_file.filename or "unknown"},
        ):
            if not upload_file.filename:
                raise ValueError("Uploaded file must include a filename.")

            original_name = self._sanitize_filename(upload_file.filename)
            extension = Path(original_name).suffix.lower()
            if extension not in FinancialDocumentParser.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"File type {extension} is not supported. "
                    f"Supported: {sorted(FinancialDocumentParser.SUPPORTED_EXTENSIONS)}"
                )

            document_id = str(uuid.uuid4())
            destination = self.settings.upload_dir / f"{document_id}_{original_name}"

            raw_bytes = await upload_file.read()
            if not raw_bytes:
                raise ValueError("Uploaded file is empty.")
            max_bytes = self.settings.max_upload_size_mb * 1024 * 1024
            if len(raw_bytes) > max_bytes:
                raise ValueError(
                    f"Uploaded file exceeds {self.settings.max_upload_size_mb} MB limit."
                )

            destination.write_bytes(raw_bytes)

            parsed = await asyncio.to_thread(self.parser.parse, document_id, destination)
            if not parsed.text:
                raise ValueError("No text could be extracted from the document.")

            extracted = await asyncio.to_thread(self.extractor.extract_fields, parsed.text)
            history_limit = self.settings.anomaly_history_limit
            historical_totals = await asyncio.to_thread(
                self.repository.list_recent_totals,
                history_limit,
            )
            historical_records = await asyncio.to_thread(
                self.repository.list_recent_charge_signatures,
                history_limit,
            )
            anomaly = await asyncio.to_thread(
                self.anomaly_detector.assess,
                extracted,
                historical_totals,
                historical_records,
            )
            chunks = await asyncio.to_thread(self._split_text, parsed.text)

            indexed_chunks = await asyncio.to_thread(
                self.vector_store.index_document,
                document_id,
                chunks,
                {
                    "filename": original_name,
                    "parser": parsed.parser_used,
                },
            )

            record = DocumentRecord(
                document_id=document_id,
                filename=original_name,
                file_path=str(destination),
                text_path=str(self.repository.text_path(document_id)),
                uploaded_at=datetime.now(timezone.utc),
                parser_used=parsed.parser_used,
                extracted_data=extracted,
                anomaly=anomaly,
                chunks=chunks,
                metadata={
                    "parse_warnings": parsed.parse_warnings,
                    "request_id": request_id,
                },
            )
            await asyncio.to_thread(self.repository.save_document, record, parsed.text)

            summary = self._build_summary(extracted)
            self.observability.log_event(
                "ingestion.indexed",
                {
                    "document_id": document_id,
                    "chunks": indexed_chunks,
                    "parser": parsed.parser_used,
                },
            )

            return UploadResponse(
                document_id=document_id,
                filename=original_name,
                parser_used=parsed.parser_used,
                extracted_data=extracted,
                indexed_chunks=indexed_chunks,
                summary=summary,
                anomaly=anomaly,
            )

    async def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        return await asyncio.to_thread(self.repository.get_document, document_id)

    def _split_text(self, text: str) -> list[str]:
        chunks = [
            chunk.strip()
            for chunk in self.splitter.split_text(text)
            if chunk.strip()
        ]
        if not chunks:
            return [text.strip()]

        max_chunks = max(1, self.settings.max_chunks_per_document)
        if len(chunks) <= max_chunks:
            return chunks

        # Condense adjacent chunks to cap embedding work on very large files.
        group_size = (len(chunks) + max_chunks - 1) // max_chunks
        condensed: list[str] = []
        for start in range(0, len(chunks), group_size):
            group = chunks[start : start + group_size]
            merged = "\n".join(group).strip()
            if merged:
                condensed.append(merged)
            if len(condensed) >= max_chunks:
                break

        self.observability.log_event(
            "ingestion.chunks_condensed",
            {
                "original": len(chunks),
                "condensed": len(condensed),
                "max_chunks": max_chunks,
            },
        )

        if condensed:
            return condensed
        return chunks

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        cleaned = Path(filename).name
        cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", cleaned)
        return cleaned or "uploaded_document"

    @staticmethod
    def _build_summary(extracted: ExtractedFields | object) -> str:
        if not hasattr(extracted, "vendor"):
            return "Document parsed and indexed successfully."

        vendor = getattr(extracted, "vendor", None) or "unknown vendor"
        date = getattr(extracted, "date", None) or "unknown date"
        total = getattr(extracted, "total", None)
        total_str = f"{total:.2f}" if isinstance(total, (float, int)) else "unknown total"
        return f"Parsed receipt/invoice from {vendor} dated {date} with total {total_str}."
