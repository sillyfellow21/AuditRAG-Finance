# AuditRAG-Finance

Production-oriented multimodal AI assistant for financial document understanding, charge explanation, and anomaly detection.

This project is designed for practical, local-first operation with configurable guardrails, explainability outputs, and auditability.

Repository name: `AuditRAG-Finance`

## Table of Contents
- [What This Project Solves](#what-this-project-solves)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Key Capabilities](#key-capabilities)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Operational Modes](#operational-modes)
- [Performance Tuning](#performance-tuning)
- [Observability and Audit](#observability-and-audit)
- [Evaluation](#evaluation)
- [Testing and Quality](#testing-and-quality)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Project Guides and Governance](#project-guides-and-governance)

## What This Project Solves
Financial teams frequently need to answer questions like:
- Why was this charge deducted?
- Is this charge suspicious or duplicated?
- Is this a recurring payment pattern?

This system ingests invoices/receipts/statements, extracts structure, indexes context, and returns grounded answers with evidence and anomaly signals.

## Architecture at a Glance
1. Upload file in Streamlit.
2. FastAPI ingestion validates and stores the file.
3. Parser extracts text from PDF/image.
4. Extractor derives structured fields (heuristic + optional LLM assist).
5. Chunks are embedded and indexed in Chroma.
6. Retrieval + reasoning produce answer, confidence, flags, and explainability.
7. Audit and observability logs are recorded.

Core stack:
- Backend: FastAPI
- Frontend: Streamlit
- LLM: Groq (configurable)
- Embeddings: sentence-transformers
- Vector DB: Chroma (local)
- Parsing/OCR: pypdf, Docling, pytesseract fallback
- Evaluation: Ragas

## Key Capabilities
- PDF/image upload and parsing.
- Structured extraction: vendor, date, total, tax, line items.
- Retrieval-augmented answering with evidence snippets.
- Anomaly scoring:
  - suspicious signals
  - duplicate charge indicators
  - recurring payment indicators
- Strict answer schema for API consumers.
- Explainability block with retrieval/cache/anomaly diagnostics.
- Audit API for governance and operations.
- Caching for repository, retrieval, and answer paths.

## Repository Structure
Top-level modules:
- `backend/`: API routes and dependency wiring
- `frontend/`: Streamlit interface and API client
- `ingestion/`: upload pipeline and persistence
- `parsing/`: PDF/image parsing and OCR
- `extraction/`: field extraction logic
- `embeddings/`: Chroma index operations
- `retrieval/`: vector + lexical retrieval flow
- `reasoning/`: answer generation and anomaly logic
- `guardrails/`: optional finance classifier/blocking
- `monitoring/`: logs, token reports, audit logs
- `evaluation/`: Ragas evaluation runner
- `core/`: shared models/config/cache utility
- `tests/`: automated tests

## Quick Start
### Prerequisites
- Python 3.11 (recommended for deployment) or 3.12
- pip
- (Optional) Tesseract binary for OCR-heavy image workflows

For Streamlit Community Cloud deployments, this repo includes `runtime.txt` to pin Python to 3.11.

### Install
```bash
pip install -r requirements.txt
```

### Configure
1. Copy `.env.example` to `.env`.
2. Set at minimum:
   - `GROQ_API_KEY` (if you want LLM-powered responses)

### Run Backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Frontend
```bash
streamlit run frontend/app.py
```

If port 8501 is occupied:
```bash
streamlit run frontend/app.py --server.port 8502
```

## Configuration Reference
The following environment variables are the most operationally important.

### Core
- `APP_NAME`, `APP_ENV`, `LOG_LEVEL`

### LLM and Extraction
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `EXTRACTION_USE_LLM`

### Parsing/Upload Safety
- `MAX_UPLOAD_SIZE_MB`
- `DOCLING_ENABLED`
- `DOCLING_FOR_PDF`

### Retrieval and Chunking
- `RAG_TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `MAX_CHUNKS_PER_DOCUMENT`

### Caching
- `RESPONSE_CACHE_ENABLED`
- `RESPONSE_CACHE_TTL_SECONDS`
- `RESPONSE_CACHE_MAX_ENTRIES`
- `RETRIEVAL_CACHE_ENABLED`
- `RETRIEVAL_CACHE_TTL_SECONDS`
- `RETRIEVAL_CACHE_MAX_ENTRIES`
- `REPOSITORY_CACHE_TTL_SECONDS`
- `REPOSITORY_CACHE_MAX_ENTRIES`

### Anomaly Detection
- `ANOMALY_HIGH_TOTAL_THRESHOLD`
- `ANOMALY_HIGH_TAX_RATIO`
- `ANOMALY_LINE_TOTAL_MISMATCH_RATIO`
- `ANOMALY_ROBUST_Z_THRESHOLD`
- `ANOMALY_HISTORY_LIMIT`

### Guardrails
- `FINANCE_GUARDRAILS_ENABLED`

### Monitoring
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_PROJECT`
- `LANGSMITH_ENDPOINT`

## API Reference
### `GET /health`
Returns service health and environment.

### `POST /api/upload`
Uploads and processes a financial document.

Response includes:
- `document_id`
- `parser_used`
- `extracted_data`
- `indexed_chunks`
- `summary`
- `anomaly`

### `POST /api/ask`
Request:
```json
{
  "document_id": "<id>",
  "question": "Why was this charge deducted?"
}
```

Success response includes:
```json
{
  "answer": "...",
  "confidence": 0.82,
  "category": "charge_explanation",
  "evidence": ["..."],
  "flags": ["suspicious"],
  "missing_info": [],
  "structured_data": {},
  "anomaly": {},
  "explainability": {},
  "reasoning": "..."
}
```

When guardrails are enabled and blocked:
```json
{
  "error": "This assistant only handles finance-related queries."
}
```

### `GET /api/document/{document_id}`
Fetches persisted document record.

### `GET /api/audit-logs?limit=50`
Returns latest audit entries.

## Operational Modes
### Unrestricted Q&A (default)
- `FINANCE_GUARDRAILS_ENABLED=false`
- Allows non-finance questions.

### Finance-only guardrail mode
- `FINANCE_GUARDRAILS_ENABLED=true`
- Non-finance questions are blocked with an explicit error payload.

## Performance Tuning
If upload processing is slow, prioritize these knobs:
1. Keep `EXTRACTION_USE_LLM=false` for fastest ingest.
2. Increase `CHUNK_SIZE` and reduce `CHUNK_OVERLAP`.
3. Lower `MAX_CHUNKS_PER_DOCUMENT`.
4. Lower `ANOMALY_HISTORY_LIMIT`.
5. Keep `DOCLING_FOR_PDF=false` unless OCR quality requires it.

Tradeoff note:
- More aggressive speed settings reduce ingest time but may slightly reduce retrieval granularity.

## Observability and Audit
### Logs
- Application log: `data/app.log`
- Token usage: `data/token_usage.jsonl`
- Audit trail: `data/audit_log.jsonl`

### Token report
```bash
python -m monitoring.token_report
```

Audit entries include:
- `event_id`, `timestamp`, `actor`, `action`, `resource`, `outcome`, `request_id`, `payload`

## Evaluation
1. Populate `evaluation/eval_dataset_template.json` with valid `document_id` values.
2. Run:
```bash
python -m evaluation.ragas_eval --backend-url http://localhost:8000
```

## Testing and Quality
### Run tests
```bash
pytest -q
```

### Development tooling
```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
```

### Optional pre-commit hooks
```bash
pre-commit install
pre-commit run --all-files
```

CI is defined in `.github/workflows/ci.yml` and runs lint + tests on push/PR.

## Deployment
### Docker
```bash
docker compose up --build
```

### Free-tier friendly targets
- Render
- Railway
- Hugging Face Spaces (frontend)

See deployment details and environment setup in this file and `.env.example`.

## Troubleshooting
### Upload appears stuck
- Reduce PDF size/pages.
- Lower chunk volume using `MAX_CHUNKS_PER_DOCUMENT`.
- Keep `EXTRACTION_USE_LLM=false`.

### PDF upload fails with no text extracted
- For scan-heavy PDFs, set `DOCLING_FOR_PDF=true` and retry.

### Groq model errors
- Ensure `GROQ_MODEL` is active and supported.

### LangSmith unauthorized warnings
- Keep tracing off or provide valid `LANGCHAIN_API_KEY`.

## Project Guides and Governance
### Project documentation
- Technical deep dive: `TECHNICAL_GUIDE.md`
- Non-technical guide: `NON_TECHNICAL_GUIDE.md`

### Professional standards
- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- AI governance: `docs/AI_GOVERNANCE.md`

---

If you want to publish this as a polished public repository, pair this README with release notes and issue/PR templates for best onboarding.
