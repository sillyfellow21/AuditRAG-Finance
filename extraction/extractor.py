from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq

from core.config import Settings
from core.models import ExtractedFields, LineItem
from monitoring.observability import Observability


class FinancialExtractor:
    def __init__(self, settings: Settings, observability: Observability) -> None:
        self.settings = settings
        self.observability = observability
        self.llm: Optional[ChatGroq] = None
        if settings.extraction_use_llm and settings.has_groq and settings.groq_api_key is not None:
            self.llm = ChatGroq(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.groq_model,
                temperature=0,
            )

    def extract_fields(self, text: str) -> ExtractedFields:
        heuristic = self._heuristic_extract(text)
        llm_result: Optional[ExtractedFields] = None

        if self.llm is not None and len(text) > 80:
            try:
                llm_result = self._extract_with_llm(text)
            except Exception as exc:
                self.observability.log_warning(
                    "extraction.llm_failed",
                    {"error": str(exc)},
                )

        merged = self._merge_results(heuristic, llm_result)
        return merged

    def _heuristic_extract(self, text: str) -> ExtractedFields:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        vendor = self._extract_vendor(lines)
        date = self._extract_date(text)
        total = self._extract_amount_by_keywords(
            text,
            ["grand total", "total due", "amount due", "total", "balance due"],
        )
        tax = self._extract_amount_by_keywords(text, ["tax", "vat", "gst", "sales tax"])
        currency = self._extract_currency(text)
        line_items = self._extract_line_items(lines)

        return ExtractedFields(
            vendor=vendor,
            date=date,
            total=total,
            tax=tax,
            currency=currency,
            line_items=line_items,
            raw_fields={"method": "heuristic"},
        )

    def _extract_with_llm(self, text: str) -> ExtractedFields:
        assert self.llm is not None
        prompt = (
            "You are an extraction engine for financial receipts and invoices. "
            "Extract only fields explicitly present in the text. "
            "If a field is missing, return null. "
            "Return strict JSON with keys: vendor, date, total, tax, currency, line_items. "
            "line_items should be an array of objects with keys: description, quantity, unit_price, amount.\n\n"
            f"DOCUMENT_TEXT:\n{text[:12000]}"
        )

        response = self.llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = self._safe_json_loads(content)

        line_items_payload = parsed.get("line_items") if isinstance(parsed, dict) else []
        line_items: List[LineItem] = []
        if isinstance(line_items_payload, list):
            for item in line_items_payload:
                if isinstance(item, dict) and item.get("description"):
                    line_items.append(
                        LineItem(
                            description=str(item.get("description", "")).strip(),
                            quantity=self._coerce_float(item.get("quantity")),
                            unit_price=self._coerce_float(item.get("unit_price")),
                            amount=self._coerce_float(item.get("amount")),
                        )
                    )

        return ExtractedFields(
            vendor=self._coerce_optional_str(parsed.get("vendor")) if isinstance(parsed, dict) else None,
            date=self._coerce_optional_str(parsed.get("date")) if isinstance(parsed, dict) else None,
            total=self._coerce_float(parsed.get("total")) if isinstance(parsed, dict) else None,
            tax=self._coerce_float(parsed.get("tax")) if isinstance(parsed, dict) else None,
            currency=self._coerce_optional_str(parsed.get("currency")) if isinstance(parsed, dict) else None,
            line_items=line_items,
            raw_fields={"method": "llm"},
        )

    def _merge_results(
        self,
        heuristic: ExtractedFields,
        llm_result: Optional[ExtractedFields],
    ) -> ExtractedFields:
        if llm_result is None:
            return heuristic

        merged = ExtractedFields(
            vendor=llm_result.vendor or heuristic.vendor,
            date=llm_result.date or heuristic.date,
            total=llm_result.total if llm_result.total is not None else heuristic.total,
            tax=llm_result.tax if llm_result.tax is not None else heuristic.tax,
            currency=llm_result.currency or heuristic.currency,
            line_items=llm_result.line_items if llm_result.line_items else heuristic.line_items,
            raw_fields={
                "method": "hybrid",
                "heuristic": heuristic.raw_fields,
                "llm": llm_result.raw_fields,
            },
        )
        return merged

    def _extract_vendor(self, lines: List[str]) -> Optional[str]:
        if not lines:
            return None

        blacklist = {
            "invoice",
            "receipt",
            "tax invoice",
            "statement",
            "bill to",
            "date",
        }
        for line in lines[:12]:
            low = line.lower()
            if len(line) < 3:
                continue
            if any(term in low for term in blacklist):
                continue
            if re.search(r"\d{2}[/\-]\d{2}[/\-]\d{2,4}", line):
                continue
            if re.search(r"\$\s*\d", line):
                continue
            return line[:120]
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        patterns = [
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(\d{2}/\d{2}/\d{4})\b",
            r"\b(\d{2}-\d{2}-\d{4})\b",
            r"\b([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_amount_by_keywords(self, text: str, keywords: List[str]) -> Optional[float]:
        rows = [row.strip() for row in text.splitlines() if row.strip()]
        for row in rows:
            low = row.lower()
            if not any(keyword in low for keyword in keywords):
                continue
            amount = self._extract_first_amount(row)
            if amount is not None:
                return amount

        for keyword in keywords:
            pattern = rf"{re.escape(keyword)}[^\n\d]*([\$€£]?\s*\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?)"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = self._coerce_float(match.group(1))
                if value is not None:
                    return value
        return None

    def _extract_currency(self, text: str) -> Optional[str]:
        mapping = {
            "$": "USD",
            "€": "EUR",
            "£": "GBP",
            "usd": "USD",
            "eur": "EUR",
            "gbp": "GBP",
            "inr": "INR",
            "aud": "AUD",
            "cad": "CAD",
        }

        lower = text.lower()
        for token, code in mapping.items():
            if token in lower or token in text:
                return code
        return None

    def _extract_line_items(self, lines: List[str]) -> List[LineItem]:
        items: List[LineItem] = []
        for line in lines:
            if len(items) >= 12:
                break

            if any(keyword in line.lower() for keyword in ["total", "tax", "subtotal", "balance due"]):
                continue

            # Matches lines like: Service Fee 49.99 or Product A 2 x 19.50 39.00
            match_simple = re.match(r"^(.+?)\s+([\$€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2}))$", line)
            match_qty = re.match(
                r"^(.+?)\s+(\d+(?:\.\d+)?)\s*[xX]\s*([\$€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2}))\s+([\$€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2}))$",
                line,
            )

            if match_qty:
                description = match_qty.group(1).strip()
                qty = self._coerce_float(match_qty.group(2))
                unit_price = self._coerce_float(match_qty.group(3))
                amount = self._coerce_float(match_qty.group(4))
                if description and amount is not None:
                    items.append(
                        LineItem(
                            description=description,
                            quantity=qty,
                            unit_price=unit_price,
                            amount=amount,
                            evidence=line,
                        )
                    )
                continue

            if match_simple:
                description = match_simple.group(1).strip()
                amount = self._coerce_float(match_simple.group(2))
                if description and amount is not None and len(description) > 2:
                    items.append(
                        LineItem(
                            description=description,
                            amount=amount,
                            evidence=line,
                        )
                    )

        return items

    def _extract_first_amount(self, text: str) -> Optional[float]:
        match = re.search(r"([\$€£]?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2}))", text)
        if not match:
            return None
        return self._coerce_float(match.group(1))

    def _safe_json_loads(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content, flags=re.IGNORECASE).strip()
            content = re.sub(r"```$", "", content).strip()

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            cleaned = cleaned.replace("$", "").replace("€", "").replace("£", "")
            cleaned = cleaned.replace(",", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return str(value)
