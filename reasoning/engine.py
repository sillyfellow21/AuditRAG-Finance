from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq

from core.cache import TTLCache
from core.config import Settings
from core.models import (
    AnswerResponse,
    AnomalyAssessment,
    DocumentRecord,
    ExplainabilityDetails,
    ExtractedFields,
    RetrievedContext,
)
from ingestion.service import FileDocumentRepository
from monitoring.observability import Observability, TokenUsage
from reasoning.anomaly import ChargeAnomalyDetector
from retrieval.pipeline import RetrievalPipeline


class ReasoningEngine:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalPipeline,
        repository: FileDocumentRepository,
        observability: Observability,
    ) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.repository = repository
        self.observability = observability
        self.anomaly_detector = ChargeAnomalyDetector(
            high_total_threshold=settings.anomaly_high_total_threshold,
            high_tax_ratio=settings.anomaly_high_tax_ratio,
            mismatch_ratio_threshold=settings.anomaly_line_total_mismatch_ratio,
            robust_outlier_z_threshold=settings.anomaly_robust_z_threshold,
        )
        self.llm: Optional[ChatGroq] = None
        if settings.has_groq and settings.groq_api_key is not None:
            self.llm = ChatGroq(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.groq_model,
                temperature=0,
            )

        self._answer_cache: Optional[TTLCache[tuple[str, str], AnswerResponse]] = None
        if settings.response_cache_enabled:
            self._answer_cache = TTLCache(
                max_entries=settings.response_cache_max_entries,
                ttl_seconds=settings.response_cache_ttl_seconds,
            )

    def answer_question(self, document_id: str, question: str) -> AnswerResponse:
        normalized_question = question.strip()
        cache_key = (document_id, normalized_question.lower())

        if self._answer_cache is not None:
            cached = self._answer_cache.get(cache_key)
            if cached is not None:
                cached_response = cached.model_copy(deep=True)
                cached_response.explainability.cache["answer_cache_hit"] = True
                cached_response.explainability.cache["answer_cache_stats"] = self._answer_cache.stats()
                return cached_response

        record = self.repository.get_document(document_id)
        if record is None:
            raise ValueError(f"Document not found for id={document_id}")

        category = self._classify_query(normalized_question)
        contexts, retrieval_diagnostics = self.retrieval.retrieve_with_diagnostics(
            document_id=document_id,
            question=normalized_question,
            fallback_chunks=record.chunks,
            top_k=self.settings.rag_top_k,
        )
        evidence = self.retrieval.build_evidence(contexts)

        missing_info = self._compute_missing_info(normalized_question, record.extracted_data, contexts)
        anomaly = record.anomaly
        if anomaly is None:
            anomaly = self.anomaly_detector.assess(
                record.extracted_data,
                self.repository.list_recent_totals(),
                self.repository.list_recent_charge_signatures(),
            )

        if self.llm is not None:
            answer_payload = self._answer_with_llm(
                record=record,
                question=normalized_question,
                category=category,
                contexts=contexts,
                missing_info=missing_info,
                anomaly=anomaly,
            )
        else:
            answer_payload = self._answer_rule_based(
                record=record,
                question=normalized_question,
                category=category,
                contexts=contexts,
                missing_info=missing_info,
                anomaly=anomaly,
            )

        merged_evidence = answer_payload.get("evidence", []) or evidence
        if not merged_evidence:
            merged_evidence = evidence

        flags = self._build_flags(anomaly)

        confidence = self._compute_confidence(
            contexts=contexts,
            extracted=record.extracted_data,
            missing_info=answer_payload.get("missing_info", missing_info),
            anomaly=anomaly,
        )

        explainability = ExplainabilityDetails(
            decision_path=[
                f"category={category}",
                f"retrieval_source={retrieval_diagnostics.get('source', 'unknown')}",
                f"context_count={retrieval_diagnostics.get('count', 0)}",
                f"missing_info={','.join(answer_payload.get('missing_info', missing_info)) or 'none'}",
                f"anomaly_risk={anomaly.risk_level}",
            ],
            retrieval=retrieval_diagnostics,
            cache={
                "answer_cache_hit": False,
                "answer_cache_enabled": self._answer_cache is not None,
                "retrieval_cache_hit": bool(retrieval_diagnostics.get("cache_hit", False)),
            },
            anomaly=anomaly,
            notes=[
                "Answer is grounded on extracted fields and retrieval snippets.",
                "Anomaly assessment combines structural checks and historical outlier scoring.",
            ],
        )
        if self._answer_cache is not None:
            explainability.cache["answer_cache_stats"] = self._answer_cache.stats()

        response = AnswerResponse(
            answer=answer_payload.get("answer", "Insufficient information to answer."),
            confidence=confidence,
            category=category,
            evidence=merged_evidence,
            flags=flags,
            missing_info=answer_payload.get("missing_info", missing_info),
            structured_data=record.extracted_data,
            anomaly=anomaly,
            explainability=explainability,
            reasoning=answer_payload.get("reasoning", "Grounded from extracted and retrieved evidence."),
        )

        if self._answer_cache is not None:
            self._answer_cache.set(cache_key, response)

        return response

    def _answer_with_llm(
        self,
        record: DocumentRecord,
        question: str,
        category: str,
        contexts: List[RetrievedContext],
        missing_info: List[str],
        anomaly: AnomalyAssessment,
    ) -> Dict[str, Any]:
        assert self.llm is not None

        context_lines = [ctx.text for ctx in contexts]
        extracted_json = json.dumps(record.extracted_data.model_dump(mode="json"), ensure_ascii=True)
        context_json = json.dumps(context_lines, ensure_ascii=True)

        prompt = (
            "You are a financial document assistant. "
            "Answer using ONLY the extracted fields and retrieved context below. "
            "Do not invent values. "
            "If information is missing, state that clearly and include the missing field names. "
            "Return strict JSON with keys: answer, reasoning, evidence, missing_info. "
            "evidence must be exact snippets from retrieved context or extracted fields.\n\n"
            f"QUESTION: {question}\n"
            f"CATEGORY: {category}\n"
            f"EXTRACTED_FIELDS_JSON: {extracted_json}\n"
            f"RETRIEVED_CONTEXTS_JSON: {context_json}\n"
            f"ANOMALY_ASSESSMENT_JSON: {json.dumps(anomaly.model_dump(mode='json'), ensure_ascii=True)}\n"
            f"KNOWN_MISSING_INFO: {json.dumps(missing_info, ensure_ascii=True)}"
        )

        request_id = self.observability.new_request_id()
        start = time.perf_counter()
        response = self.llm.invoke(prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0

        usage = self.observability.extract_token_usage(response)
        self.observability.record_token_usage(
            TokenUsage(
                request_id=request_id,
                model=self.settings.groq_model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=latency_ms,
                metadata={"document_id": record.document_id, "category": category},
            )
        )

        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = self._safe_json_loads(content)

        answer = str(parsed.get("answer", "Insufficient information to answer.")).strip()
        reasoning = str(parsed.get("reasoning", "Grounded from retrieved context.")).strip()

        evidence_payload = parsed.get("evidence", [])
        if not isinstance(evidence_payload, list):
            evidence_payload = []

        # Keep evidence bounded and textual.
        evidence = [str(item).strip() for item in evidence_payload if str(item).strip()][:8]

        missing_payload = parsed.get("missing_info", missing_info)
        if not isinstance(missing_payload, list):
            missing_payload = missing_info
        missing = [str(item).strip() for item in missing_payload if str(item).strip()]

        return {
            "answer": answer,
            "reasoning": reasoning,
            "evidence": evidence,
            "missing_info": missing,
        }

    def _answer_rule_based(
        self,
        record: DocumentRecord,
        question: str,
        category: str,
        contexts: List[RetrievedContext],
        missing_info: List[str],
        anomaly: AnomalyAssessment,
    ) -> Dict[str, Any]:
        data = record.extracted_data
        evidence = self.retrieval.build_evidence(contexts)

        if category == "anomaly":
            if anomaly.is_suspicious:
                signal_text = ", ".join(
                    [f"{signal.code}: {signal.message}" for signal in anomaly.signals[:4]]
                )
                answer = (
                    "This charge is potentially suspicious "
                    f"(risk={anomaly.risk_level}, score={anomaly.score:.2f}). "
                    f"Signals: {signal_text or 'insufficient normalized evidence'}"
                )
            else:
                answer = (
                    "No strong suspicious-charge signal was detected "
                    f"(risk={anomaly.risk_level}, score={anomaly.score:.2f})."
                )
            reasoning = "Used deterministic anomaly checks on totals, tax ratio, line-item consistency, and history outliers."
            return {
                "answer": answer,
                "reasoning": reasoning,
                "evidence": evidence,
                "missing_info": missing_info,
            }

        if category == "charge_explanation":
            parts: List[str] = []
            if data.total is not None:
                parts.append(f"The document shows a total of {data.total:.2f}.")
            if data.tax is not None:
                parts.append(f"Tax appears as {data.tax:.2f}.")
            if data.line_items:
                top_items = ", ".join(
                    [
                        f"{item.description} ({item.amount:.2f})"
                        for item in data.line_items[:4]
                        if item.amount is not None
                    ]
                )
                if top_items:
                    parts.append(f"The charge is supported by line items: {top_items}.")
            if not parts:
                parts.append("The document does not contain enough explicit charge breakdown to explain the deduction.")
            if anomaly.is_suspicious:
                parts.append(
                    f"Anomaly detector flagged this as {anomaly.risk_level} risk (score {anomaly.score:.2f})."
                )

            answer = " ".join(parts)
            reasoning = "Used extracted totals, taxes, and line items to justify why a deduction appears."
            return {
                "answer": answer,
                "reasoning": reasoning,
                "evidence": evidence,
                "missing_info": missing_info,
            }

        field_lookup_answer = self._field_lookup_answer(question, data)
        if field_lookup_answer is not None:
            return {
                "answer": field_lookup_answer,
                "reasoning": "Mapped the question to extracted fields and returned only available values.",
                "evidence": evidence,
                "missing_info": missing_info,
            }

        # Summary fallback
        context_hint = evidence[0] if evidence else "No supporting context was retrieved."
        answer = (
            "Based on the available document evidence, "
            f"{context_hint}"
        )
        reasoning = "Provided an evidence-first summary because no direct field mapping was detected."
        return {
            "answer": answer,
            "reasoning": reasoning,
            "evidence": evidence,
            "missing_info": missing_info,
        }

    def _field_lookup_answer(self, question: str, data: ExtractedFields) -> Optional[str]:
        q = question.lower()
        if "vendor" in q or "merchant" in q or "seller" in q:
            return f"Vendor: {data.vendor}" if data.vendor else "Vendor is not clearly present in the document."
        if "date" in q:
            return f"Date: {data.date}" if data.date else "Date is not clearly present in the document."
        if "tax" in q:
            return f"Tax: {data.tax:.2f}" if data.tax is not None else "Tax is not clearly present in the document."
        if "total" in q or "amount" in q or "charged" in q:
            return f"Total: {data.total:.2f}" if data.total is not None else "Total amount is not clearly present in the document."
        return None

    def _classify_query(self, question: str) -> str:
        q = question.lower()
        if any(term in q for term in ["suspicious", "fraud", "anomaly", "unusual", "outlier", "risk"]):
            return "anomaly"
        if any(term in q for term in ["why", "deduct", "deduction", "charge", "charged", "billing"]):
            return "charge_explanation"
        return "summary"

    def _build_flags(self, anomaly: AnomalyAssessment) -> List[str]:
        flags: List[str] = []
        signal_codes = {signal.code for signal in anomaly.signals}

        if (
            "duplicate_charge" in signal_codes
            or "document_duplicate_charge" in signal_codes
            or "repeated_line_items" in signal_codes
        ):
            flags.append("duplicate")
        if anomaly.is_suspicious:
            flags.append("suspicious")
        if "recurring_payment" in signal_codes:
            flags.append("recurring")

        deduped: List[str] = []
        seen = set()
        for item in flags:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _compute_missing_info(
        self,
        question: str,
        extracted: ExtractedFields,
        contexts: List[RetrievedContext],
    ) -> List[str]:
        missing: List[str] = []
        q = question.lower()

        if "vendor" in q and not extracted.vendor:
            missing.append("vendor")
        if "date" in q and not extracted.date:
            missing.append("date")
        if any(term in q for term in ["total", "amount", "charged"]) and extracted.total is None:
            missing.append("total")
        if "tax" in q and extracted.tax is None:
            missing.append("tax")
        if not contexts:
            missing.append("supporting_context")

        # Keep order deterministic and unique.
        deduped: List[str] = []
        seen = set()
        for item in missing:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _compute_confidence(
        self,
        contexts: List[RetrievedContext],
        extracted: ExtractedFields,
        missing_info: List[str],
        anomaly: Optional[AnomalyAssessment] = None,
    ) -> float:
        avg_retrieval = sum(ctx.score for ctx in contexts) / len(contexts) if contexts else 0.0

        present_count = sum(
            [
                1 if extracted.vendor else 0,
                1 if extracted.date else 0,
                1 if extracted.total is not None else 0,
                1 if extracted.tax is not None else 0,
                1 if extracted.line_items else 0,
            ]
        )
        completeness = present_count / 5.0

        anomaly_penalty = 0.08 if anomaly is not None and anomaly.is_suspicious else 0.0
        score = 0.35 + (0.35 * avg_retrieval) + (0.25 * completeness) - (0.08 * len(missing_info)) - anomaly_penalty
        score = max(0.05, min(0.98, score))
        return round(score, 2)

    def _safe_json_loads(self, content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}
