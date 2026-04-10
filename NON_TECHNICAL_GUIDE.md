# Non-Technical Guide: How This Project Works

## What this project does
This app helps users upload financial documents (like invoices, receipts, statements) and ask questions about them.

It then:
- reads the document
- finds key details (vendor, date, total, tax, line items)
- explains charges in plain language
- highlights possible anomalies (duplicate, suspicious, recurring)
- keeps a trace of important events in audit logs

## Who this is for
- finance teams
- operations teams
- founders and managers
- analysts reviewing expenses and charges

## What you can do in the app
1. Upload a document (PDF/image).
2. See extracted data summary.
3. Ask questions about the document.
4. Review confidence and evidence.
5. Check anomaly risk indicators.
6. Open the audit log viewer.

## Main screens
### Upload section
You select a file and process it. The system parses and indexes the document.

### Q&A section
You type a question and get:
- answer
- confidence score
- evidence snippets
- category and flags
- explainability details

### Audit section
You can load recent audit events to see what happened and when.

## What "confidence" means
Confidence is a score between 0 and 1 showing how reliable the answer is based on:
- retrieval quality
- how complete the extracted data is
- how much important info is missing
- anomaly penalties when risk is high

Higher is better, but always read the evidence too.

## What anomaly flags mean
- duplicate: likely repeated charge pattern
- suspicious: unusual values or mismatches found
- recurring: repeated periodic payment pattern

These are warnings, not final financial decisions.

## Why audit logs exist
Audit logs are the project memory of operations.
They help with:
- troubleshooting
- accountability
- compliance review
- understanding what the system did and why

## Data safety in plain terms
- files are stored locally in the `data` folder
- vector index is local (Chroma)
- external LLM calls only happen if API key/config allows
- secrets are stored in `.env`

## Current behavior about finance-only blocking
The system can run in two modes:
- unrestricted questions (default now)
- finance-only blocking (optional)

This is controlled by one setting in `.env`:
- `FINANCE_GUARDRAILS_ENABLED`

## Why uploads can feel slow sometimes
Large PDFs can take longer because the system must:
- parse many pages
- split lots of text chunks
- create embeddings
- index them

Speed controls are already available (chunk cap, parser settings, optional LLM extraction).

## Recommended daily workflow
1. Start backend and frontend.
2. Upload one document at a time.
3. Ask focused questions.
4. Check evidence before acting.
5. Review anomaly flags and audit logs for risky cases.

## Limitations to keep in mind
- scanned/low-quality PDFs may extract poorly
- OCR-heavy files can be slower
- model answers depend on what text was extracted
- anomaly flags are heuristics, not legal/accounting conclusions

## If something goes wrong
- if upload fails: try a smaller or cleaner PDF
- if answer quality is low: ask a more specific question
- if system is slow: reduce chunk-related settings
- if server seems down: check `/health` and restart backend

## What success looks like
A successful run means:
- document uploads and indexes
- extracted fields are visible
- questions return answers with evidence
- audit logs show the request history
- anomalies are flagged where needed

## Quick glossary
- Parsing: reading text from files
- OCR: reading text from images/scans
- Embedding: converting text into searchable vectors
- Retrieval: finding relevant snippets
- Reasoning: generating final answer
- Audit log: system event history
