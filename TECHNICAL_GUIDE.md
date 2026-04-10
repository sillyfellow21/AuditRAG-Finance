# Technical Guide: Financial Multimodal AI Assistant

## 1) System Summary
This project is a modular FastAPI + Streamlit system for document-centric question answering over financial documents (PDF/images). It performs ingestion, parsing/OCR, structured extraction, vector indexing, retrieval, reasoning, anomaly analysis, auditing, and evaluation.

Primary runtime components:
- API server: FastAPI
- UI: Streamlit
- Vector DB: local Chroma
- Embeddings: sentence-transformers (local)
- LLM: Groq (optional at ingest, used at answer time if configured)
- OCR/parsing: Docling, pypdf, pytesseract fallback

## 2) Project Structure
Top-level modules:
- backend: API routes and application wiring
- frontend: Streamlit app and backend client
- ingestion: upload pipeline, persistence, chunking/indexing orchestration
- parsing: document parsing/OCR logic
- extraction: structured field extraction (heuristic + optional LLM)
- embeddings: Chroma vector index operations
- retrieval: context retrieval and lexical fallback, with cache
- reasoning: answer generation, confidence, explainability, anomaly integration
- guardrails: optional finance classifier/blocking logic
- monitoring: logs, token usage, audit logs
- evaluation: Ragas evaluation runner
- core: shared config, models, and cache utility
- tests: pytest suite

## 3) Core Runtime Data Flow
### Upload flow (`POST /api/upload`)
1. API receives multipart upload.
2. Ingestion validates extension and file size.
3. File persisted to `data/uploads`.
4. Parser extracts text:
   - PDF: pypdf by default (`DOCLING_FOR_PDF=false`)
   - images: Docling/OCR path
5. Extractor builds structured fields (heuristics + optional LLM merge).
6. Anomaly detector computes risk/signals using extracted fields + historical stats.
7. Text split into chunks with cap/condensation (`MAX_CHUNKS_PER_DOCUMENT`).
8. Chunks embedded + stored in Chroma.
9. Metadata saved in `data/documents/{document_id}.json` and text in `.txt`.
10. Upload response returned with summary, extracted fields, anomaly block.

### Question flow (`POST /api/ask`)
1. Optional guardrail check (controlled by `FINANCE_GUARDRAILS_ENABLED`).
2. Document loaded from repository/cache.
3. Retrieval pipeline fetches vector context (or lexical fallback).
4. Reasoning engine produces answer:
   - LLM path if available
   - rule-based fallback otherwise
5. Confidence, flags, evidence, explainability, anomaly details assembled.
6. Response cached for repeat question/doc pairs.
7. Audit events written for start/success/failure (and block if enabled).

## 4) Configuration Model (`core/config.py`)
High-impact knobs:
- `FINANCE_GUARDRAILS_ENABLED` (bool): gate non-finance questions when true.
- `GROQ_API_KEY`, `GROQ_MODEL`: LLM credentials/model.
- `EXTRACTION_USE_LLM` (bool): whether ingestion extraction uses LLM merge.
- `MAX_UPLOAD_SIZE_MB` (int): hard upload size cap.
- `DOCLING_ENABLED`, `DOCLING_FOR_PDF`: parser routing controls.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: splitter behavior.
- `MAX_CHUNKS_PER_DOCUMENT`: upper bound on indexing workload.
- `ANOMALY_HISTORY_LIMIT`: max historical records scanned for anomaly features.
- cache controls for response/retrieval/repository TTL caches.

## 5) API Contracts
### `GET /health`
Returns environment and service status.

### `POST /api/upload`
Input: multipart file
Output highlights:
- `document_id`
- `parser_used`
- `extracted_data`
- `indexed_chunks`
- `summary`
- `anomaly`

### `POST /api/ask`
Input: `{ document_id, question }`
Output (success):
- `answer`
- `confidence`
- `category` (`charge_explanation | anomaly | summary`)
- `evidence`
- `flags` (`duplicate | suspicious | recurring`)
- `missing_info`
- plus explainability fields (`structured_data`, `anomaly`, `explainability`, `reasoning`)

Output (when guardrails enabled and blocked):
- `{ "error": "This assistant only handles finance-related queries." }`

### `GET /api/document/{document_id}`
Returns persisted document record.

### `GET /api/audit-logs?limit=n`
Returns recent audit entries.

## 6) Persistence and Artifacts
Data paths:
- `data/uploads`: raw uploaded files
- `data/documents`: persisted document metadata/text
- `data/chroma`: vector index
- `data/app.log`: event log
- `data/token_usage.jsonl`: token telemetry
- `data/audit_log.jsonl`: audit trail

## 7) Caching Design
### Repository cache (`FileDocumentRepository`)
- key: `document_id`
- payload: parsed `DocumentRecord`
- reduces disk JSON reads

### Retrieval cache (`RetrievalPipeline`)
- key: `(document_id, normalized_question, top_k)`
- payload: retrieved contexts
- includes diagnostics (`cache_hit`, source, count, top_score)

### Answer cache (`ReasoningEngine`)
- key: `(document_id, normalized_question)`
- payload: full answer response
- response explainability indicates cache hit status

## 8) Anomaly Detector (`reasoning/anomaly.py`)
Signals include:
- missing/non-positive totals
- high absolute total
- high tax ratio
- line-item sum mismatch
- historical outlier (MAD robust z)
- repeated line items
- duplicate charge (same vendor/amount/date patterns)
- recurring payment pattern

Final output:
- `is_suspicious`
- `risk_level` (`low|medium|high`)
- normalized `score`
- signal list

## 9) Guardrails (`guardrails/finance.py`)
Functions:
- `finance_classifier(query)` -> `finance_related` or `non_finance`
- `block_non_finance(query)` -> error payload or `None`

Runtime behavior:
- Guardrails are now optional and controlled in API via config.

## 10) Observability and Audit
### App/event logging
- Structured event messages with payload JSON.

### Audit logging
Each event has:
- `event_id`, `timestamp`, `actor`, `action`, `resource`, `outcome`, `request_id`, `payload`

### Token logging
Tracks prompt/completion/total tokens and latency for LLM calls.

## 11) Performance Notes
Current optimizations:
- upload LLM extraction can be disabled (`EXTRACTION_USE_LLM=false`)
- historical anomaly scan capped (`ANOMALY_HISTORY_LIMIT`)
- chunk explosion controlled via `MAX_CHUNKS_PER_DOCUMENT`
- optional parser routing away from unstable PDF Docling path by default
- tracing auto-disabled when LangSmith key missing

Recommended tuning for faster ingest:
- increase `CHUNK_SIZE`
- lower `CHUNK_OVERLAP`
- reduce `MAX_CHUNKS_PER_DOCUMENT`
- keep `EXTRACTION_USE_LLM=false`

## 12) Testing
Run:
- `d:/ai new/.venv/Scripts/python.exe -m pytest -q`

Suite covers:
- anomaly logic
- cache behavior
- retrieval caching
- audit log roundtrip
- reasoning response/explainability cache path
- guardrail classifier behavior

## 13) Operational Runbook
### Start backend
- `d:/ai new/.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`

### Start frontend
- `d:/ai new/.venv/Scripts/python.exe -m streamlit run frontend/app.py --server.port 8502 --server.headless true`

### Quick health checks
- `GET /health`
- upload a known sample doc
- call `POST /api/ask`
- inspect `GET /api/audit-logs`

## 14) Common Failure Modes
- `ConnectionResetError` during upload: parser crash/path instability; mitigated by PDF parser routing and upload cap.
- slow ingest: large chunk count; tune chunking cap and overlap.
- Groq model errors: update `GROQ_MODEL` to active model.
- LangSmith warnings: expected if tracing on without key; disable tracing or set key.

## 15) Security and Secrets
- keep secrets only in `.env`
- never commit `.env` with real keys
- rotate Groq key if exposed

## 16) Change Log Highlights
Recent major changes:
- optional guardrails (`FINANCE_GUARDRAILS_ENABLED`)
- upload stability controls (`MAX_UPLOAD_SIZE_MB`, parser toggles)
- ingest performance knobs (`EXTRACTION_USE_LLM`, chunk cap, history cap)
- strict answer schema (`category`, `flags`) + explainability payloads
- audit log API and UI viewer
