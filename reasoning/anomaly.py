from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Any, List, Optional, Sequence, Tuple

from core.models import AnomalyAssessment, AnomalySignal, ExtractedFields


class ChargeAnomalyDetector:
    def __init__(
        self,
        high_total_threshold: float = 5000.0,
        high_tax_ratio: float = 0.35,
        mismatch_ratio_threshold: float = 0.2,
        robust_outlier_z_threshold: float = 3.5,
    ) -> None:
        self.high_total_threshold = high_total_threshold
        self.high_tax_ratio = high_tax_ratio
        self.mismatch_ratio_threshold = mismatch_ratio_threshold
        self.robust_outlier_z_threshold = robust_outlier_z_threshold

    def assess(
        self,
        extracted: ExtractedFields,
        historical_totals: Optional[Sequence[float]] = None,
        historical_records: Optional[Sequence[dict[str, Any]]] = None,
    ) -> AnomalyAssessment:
        signals: List[Tuple[AnomalySignal, float]] = []

        if extracted.total is None:
            signals.append(
                (
                    AnomalySignal(
                        code="missing_total",
                        severity="medium",
                        message="Total amount is missing, charge validation is incomplete.",
                    ),
                    0.25,
                )
            )
        else:
            if extracted.total <= 0:
                signals.append(
                    (
                        AnomalySignal(
                            code="non_positive_total",
                            severity="critical",
                            message="Total amount is non-positive.",
                            metric=extracted.total,
                        ),
                        0.9,
                    )
                )

            if extracted.total >= self.high_total_threshold:
                signals.append(
                    (
                        AnomalySignal(
                            code="high_absolute_total",
                            severity="medium",
                            message="Total is unusually high in absolute terms.",
                            metric=extracted.total,
                        ),
                        0.25,
                    )
                )

            if extracted.tax is not None and extracted.total > 0:
                tax_ratio = extracted.tax / extracted.total
                if tax_ratio > self.high_tax_ratio:
                    signals.append(
                        (
                            AnomalySignal(
                                code="high_tax_ratio",
                                severity="high",
                                message="Tax-to-total ratio is unexpectedly high.",
                                metric=round(tax_ratio, 4),
                            ),
                            0.35,
                        )
                    )

            line_item_total = sum(item.amount for item in extracted.line_items if item.amount is not None)
            if extracted.line_items and extracted.total > 0:
                mismatch_ratio = abs(line_item_total - extracted.total) / extracted.total
                if mismatch_ratio > self.mismatch_ratio_threshold:
                    signals.append(
                        (
                            AnomalySignal(
                                code="line_item_total_mismatch",
                                severity="high",
                                message="Sum of line items does not match the total.",
                                metric=round(mismatch_ratio, 4),
                            ),
                            0.4,
                        )
                    )

            outlier_score = self._robust_outlier_score(extracted.total, historical_totals)
            if outlier_score is not None and outlier_score > self.robust_outlier_z_threshold:
                signals.append(
                    (
                        AnomalySignal(
                            code="historical_outlier",
                            severity="high",
                            message="Total is a statistical outlier versus historical charges.",
                            metric=round(outlier_score, 4),
                        ),
                        0.4,
                    )
                )

        repeated_item_penalty = self._repeated_line_item_penalty(extracted)
        if repeated_item_penalty > 0:
            signals.append(
                (
                    AnomalySignal(
                        code="repeated_line_items",
                        severity="low",
                        message="Repeated line item descriptions detected.",
                        metric=repeated_item_penalty,
                    ),
                    0.1,
                )
            )

        history_signals = self._history_signals(
            extracted=extracted,
            historical_records=historical_records,
        )
        signals.extend(history_signals)

        score = 0.0
        for _, weight in signals:
            score += weight
        score = round(min(1.0, score), 2)

        signal_payload = [signal for signal, _ in signals]
        risk_level = self._risk_level(score, signal_payload)
        is_suspicious = score >= 0.4 or any(signal.severity in {"high", "critical"} for signal in signal_payload)

        return AnomalyAssessment(
            is_suspicious=is_suspicious,
            risk_level=risk_level,
            score=score,
            signals=signal_payload,
        )

    def _repeated_line_item_penalty(self, extracted: ExtractedFields) -> float:
        seen = {}
        for item in extracted.line_items:
            key = item.description.strip().lower()
            if not key:
                continue
            seen[key] = seen.get(key, 0) + 1

        repeats = sum(count - 1 for count in seen.values() if count > 1)
        return round(float(repeats), 2) if repeats > 0 else 0.0

    def _robust_outlier_score(self, total: float, historical_totals: Optional[Sequence[float]]) -> Optional[float]:
        if not historical_totals:
            return None

        clean = [value for value in historical_totals if value > 0]
        if len(clean) < 5:
            return None

        center = median(clean)
        deviations = [abs(value - center) for value in clean]
        mad = median(deviations)
        if mad == 0:
            return None

        # 0.6745 scales MAD into a z-like score under normal assumptions.
        robust_z = 0.6745 * abs(total - center) / mad
        return float(robust_z)

    def _risk_level(self, score: float, signals: Sequence[AnomalySignal]) -> str:
        if any(signal.severity == "critical" for signal in signals) or score >= 0.75:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _history_signals(
        self,
        extracted: ExtractedFields,
        historical_records: Optional[Sequence[dict[str, Any]]],
    ) -> List[Tuple[AnomalySignal, float]]:
        if not historical_records:
            return []
        if extracted.total is None or extracted.total <= 0 or not extracted.vendor:
            return []

        vendor = extracted.vendor.strip().lower()
        total = float(extracted.total)
        tolerance = max(0.5, total * 0.01)
        current_date = self._parse_date(extracted.date)

        vendor_amount_matches: List[dict[str, Any]] = []
        same_day_matches = 0
        for record in historical_records:
            record_vendor = str(record.get("vendor") or "").strip().lower()
            record_total = record.get("total")
            record_date = self._parse_date(record.get("date"))
            if record_vendor != vendor or not isinstance(record_total, (int, float)):
                continue
            if abs(float(record_total) - total) > tolerance:
                continue

            vendor_amount_matches.append(record)
            if current_date is not None and record_date is not None and record_date.date() == current_date.date():
                same_day_matches += 1

        signals: List[Tuple[AnomalySignal, float]] = []
        if same_day_matches > 0:
            signals.append(
                (
                    AnomalySignal(
                        code="duplicate_charge",
                        severity="high",
                        message="Possible duplicate charge with same vendor, amount, and date.",
                        metric=float(same_day_matches),
                    ),
                    0.35,
                )
            )

        if self._is_recurring(extracted, vendor_amount_matches):
            signals.append(
                (
                    AnomalySignal(
                        code="recurring_payment",
                        severity="low",
                        message="Repeated vendor and amount pattern suggests recurring payment.",
                        metric=float(len(vendor_amount_matches)),
                    ),
                    0.15,
                )
            )

        return signals

    def _is_recurring(
        self,
        extracted: ExtractedFields,
        vendor_amount_matches: Sequence[dict[str, Any]],
    ) -> bool:
        if len(vendor_amount_matches) < 2:
            return False

        dates: List[datetime] = []
        for record in vendor_amount_matches:
            parsed = self._parse_date(record.get("date"))
            if parsed is not None:
                dates.append(parsed)

        current = self._parse_date(extracted.date)
        if current is not None:
            dates.append(current)

        if len(dates) < 3:
            return len(vendor_amount_matches) >= 3

        dates.sort()
        intervals = [abs((dates[idx] - dates[idx - 1]).days) for idx in range(1, len(dates))]
        near_monthly = [days for days in intervals if 25 <= days <= 35]
        return len(near_monthly) >= 1

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None

        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y/%m/%d",
            "%b %d, %Y",
            "%B %d, %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None
