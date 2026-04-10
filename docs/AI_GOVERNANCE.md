# AI Governance and Responsible Use

## Purpose
This document defines governance controls for AI behavior in this repository.

## Model Usage
- Primary LLM provider: Groq (configurable via `.env`).
- Embeddings: local sentence-transformers.
- Retrieval: local Chroma vector store.

## Governance Controls Implemented
1. Audit logs for key operations (`upload`, `ask`, `document_read`, optional `question_blocked`).
2. Explainability payloads in answer responses (retrieval diagnostics, cache diagnostics, anomaly details).
3. Deterministic anomaly checks for suspicious, duplicate, and recurring charge signals.
4. Configurable guardrails (`FINANCE_GUARDRAILS_ENABLED`).

## Known Limitations
1. Answers are only as good as parsed text quality.
2. OCR on low-quality scans can be imperfect.
3. Anomaly flags are heuristic indicators, not legal/accounting conclusions.

## Operational Safeguards
1. Keep `MAX_UPLOAD_SIZE_MB` and chunk caps tuned for stability.
2. Keep tracing disabled unless LangSmith key is configured.
3. Validate model availability when changing `GROQ_MODEL`.
4. Review audit logs and token logs regularly.

## Human Oversight Policy
1. Users must verify evidence snippets before acting on outputs.
2. High-risk financial decisions require human approval.
3. Flagged anomalies should be reviewed by finance/ops staff.
