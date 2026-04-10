from __future__ import annotations

from typing import Any, Dict, List

import requests
import streamlit as st

from frontend.client import BackendClient

st.set_page_config(
    page_title="AuditRAG-Finance",
    page_icon="📄",
    layout="wide",
)


def _init_state() -> None:
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = "http://localhost:8000"
    if "document" not in st.session_state:
        st.session_state.document = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "audit_logs" not in st.session_state:
        st.session_state.audit_logs = []


_init_state()

st.title("AuditRAG-Finance")
st.caption(
    "AI assistant for financial document understanding, "
    "charge explanation, and anomaly detection."
)

with st.sidebar:
    st.header("Connection")
    backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
    st.session_state.backend_url = backend_url

    client = BackendClient(base_url=backend_url)
    if st.button("Check Backend Health"):
        try:
            status = client.health()
            st.success(f"Backend is healthy: {status}")
        except Exception as exc:
            st.error(f"Health check failed: {exc}")

    st.header("Audit")
    audit_limit = st.slider("Rows", min_value=10, max_value=200, value=40, step=10)
    if st.button("Refresh Audit Logs"):
        try:
            audit_payload = client.get_audit_logs(limit=audit_limit)
            st.session_state.audit_logs = audit_payload.get("logs", [])
            st.success(f"Loaded {len(st.session_state.audit_logs)} audit events.")
        except Exception as exc:
            st.error(f"Failed to load audit logs: {exc}")


left, right = st.columns([1, 1])

with left:
    st.subheader("1) Upload Document")
    upload = st.file_uploader(
        "Supported files: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    )

    if st.button("Process Document", type="primary", disabled=upload is None):
        if upload is None:
            st.warning("Please select a document file first.")
        else:
            try:
                with st.spinner("Parsing, extracting, and indexing document..."):
                    result = client.upload_document(
                        filename=upload.name,
                        content=upload.getvalue(),
                        mime_type=upload.type or "application/octet-stream",
                    )
                st.session_state.document = result
                st.success("Document processed successfully.")
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Upload failed: {detail}")
            except Exception as exc:
                st.error(f"Unexpected upload error: {exc}")

    if st.session_state.document:
        doc: Dict[str, Any] = st.session_state.document
        st.markdown("### Extraction Summary")
        st.write(doc.get("summary", "No summary available."))
        st.write(f"Document ID: {doc.get('document_id')}")
        st.write(f"Parser used: {doc.get('parser_used')}")

        anomaly = doc.get("anomaly") or {}
        if anomaly:
            risk = str(anomaly.get("risk_level", "unknown")).upper()
            score = float(anomaly.get("score", 0.0))
            st.markdown("#### Anomaly Detection")
            if anomaly.get("is_suspicious"):
                st.error(f"Suspicious charge pattern detected: {risk} risk ({score:.2f})")
            else:
                st.success(f"No strong suspicious signal: {risk} risk ({score:.2f})")
            if anomaly.get("signals"):
                with st.expander("Anomaly Signals"):
                    st.json(anomaly)

        extracted = doc.get("extracted_data", {})
        st.markdown("#### Structured Fields")
        fields = {
            "vendor": extracted.get("vendor"),
            "date": extracted.get("date"),
            "total": extracted.get("total"),
            "tax": extracted.get("tax"),
            "currency": extracted.get("currency"),
        }
        st.json(fields)

        items: List[Dict[str, Any]] = extracted.get("line_items", [])
        st.markdown("#### Line Items")
        if items:
            st.dataframe(items, use_container_width=True)
        else:
            st.info("No line items were confidently extracted.")


with right:
    st.subheader("2) Ask Questions")
    question = st.text_area(
        "Example: Why was this charge deducted?",
        height=120,
        placeholder="Ask about charges, vendor, total, tax, or anomalies...",
    )

    ask_disabled = st.session_state.document is None or not question.strip()
    if st.button("Get Grounded Answer", disabled=ask_disabled):
        if st.session_state.document is None:
            st.warning("Upload and process a document first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            document_id = st.session_state.document["document_id"]
            try:
                with st.spinner("Retrieving evidence and generating grounded answer..."):
                    answer = client.ask_question(document_id=document_id, question=question.strip())
                if "error" in answer:
                    st.error(str(answer.get("error")))
                else:
                    st.session_state.qa_history.insert(0, {"question": question.strip(), "answer": answer})
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Question failed: {detail}")
            except Exception as exc:
                st.error(f"Unexpected question error: {exc}")

    if st.session_state.qa_history:
        latest = st.session_state.qa_history[0]
        answer = latest["answer"]

        st.markdown("### Answer")
        st.write(answer.get("answer", "No answer."))

        category = str(answer.get("category", "summary"))
        st.markdown("### Category")
        st.info(category)

        flags = answer.get("flags", [])
        st.markdown("### Flags")
        if flags:
            st.warning(", ".join(flags))
        else:
            st.success("No anomaly flags raised.")

        confidence = float(answer.get("confidence", 0.0))
        st.markdown("### Confidence")
        st.progress(int(max(0.0, min(1.0, confidence)) * 100))
        st.write(f"Confidence score: {confidence:.2f}")

        anomaly = answer.get("anomaly") or {}
        if anomaly:
            st.markdown("### Suspicious Charge Risk")
            risk_level = str(anomaly.get("risk_level", "unknown")).upper()
            risk_score = float(anomaly.get("score", 0.0))
            if anomaly.get("is_suspicious"):
                st.error(f"Potential anomaly detected: {risk_level} ({risk_score:.2f})")
            else:
                st.success(f"No strong anomaly detected: {risk_level} ({risk_score:.2f})")
            with st.expander("Anomaly Details"):
                st.json(anomaly)

        st.markdown("### Evidence")
        evidence = answer.get("evidence", [])
        if evidence:
            for idx, snippet in enumerate(evidence, start=1):
                with st.expander(f"Evidence #{idx}"):
                    st.write(snippet)
        else:
            st.info("No evidence snippets returned.")

        missing_info = answer.get("missing_info", [])
        st.markdown("### Missing Information")
        if missing_info:
            st.warning(", ".join(missing_info))
        else:
            st.success("No critical missing fields detected for this query.")

        st.markdown("### Reasoning Trace")
        st.write(answer.get("reasoning", "No reasoning trace available."))

        explainability = answer.get("explainability") or {}
        if explainability:
            st.markdown("### Explainability Panel")
            decision_path: List[str] = explainability.get("decision_path", [])
            retrieval_info: Dict[str, Any] = explainability.get("retrieval", {})
            cache_info: Dict[str, Any] = explainability.get("cache", {})

            exp_left, exp_right = st.columns([1, 1])
            with exp_left:
                st.markdown("#### Decision Path")
                if decision_path:
                    for step in decision_path:
                        st.write(f"- {step}")
                else:
                    st.info("No decision path available.")

            with exp_right:
                st.markdown("#### Retrieval Diagnostics")
                if retrieval_info:
                    st.json(retrieval_info)
                else:
                    st.info("No retrieval diagnostics available.")

            st.markdown("#### Cache Diagnostics")
            if cache_info:
                st.json(cache_info)
            else:
                st.info("No cache diagnostics available.")

            notes = explainability.get("notes", [])
            if notes:
                st.markdown("#### Notes")
                for note in notes:
                    st.write(f"- {note}")

if st.session_state.qa_history:
    st.markdown("---")
    st.subheader("Q&A History")
    for idx, entry in enumerate(st.session_state.qa_history, start=1):
        st.markdown(f"**{idx}. Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer'].get('answer', '')}")

st.markdown("---")
st.subheader("Audit Log Viewer")
if st.session_state.audit_logs:
    for idx, event in enumerate(reversed(st.session_state.audit_logs), start=1):
        title = f"{idx}. {event.get('action', 'unknown')} ({event.get('outcome', 'unknown')})"
        with st.expander(title):
            st.json(event)
else:
    st.info("No audit logs loaded yet. Use 'Refresh Audit Logs' in the sidebar.")
