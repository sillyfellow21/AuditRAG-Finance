from __future__ import annotations

from core.models import ExtractedFields, LineItem
from reasoning.anomaly import ChargeAnomalyDetector


def test_detects_suspicious_patterns_from_tax_and_mismatch() -> None:
    detector = ChargeAnomalyDetector()
    extracted = ExtractedFields(
        vendor="Example Store",
        total=100.0,
        tax=42.0,
        line_items=[
            LineItem(description="Service A", amount=20.0),
            LineItem(description="Service A", amount=20.0),
        ],
    )

    assessment = detector.assess(
        extracted,
        historical_totals=[95.0, 97.0, 99.0, 102.0, 101.0, 98.0, 96.0],
    )

    codes = {signal.code for signal in assessment.signals}
    assert assessment.is_suspicious is True
    assert "high_tax_ratio" in codes
    assert "line_item_total_mismatch" in codes
    assert assessment.risk_level in {"medium", "high"}


def test_returns_low_risk_for_consistent_charge() -> None:
    detector = ChargeAnomalyDetector()
    extracted = ExtractedFields(
        vendor="Cafe",
        total=25.0,
        tax=2.0,
        line_items=[LineItem(description="Lunch", amount=23.0)],
    )

    assessment = detector.assess(
        extracted,
        historical_totals=[22.0, 24.0, 25.0, 26.0, 23.0, 25.0, 24.0],
    )

    assert assessment.is_suspicious is False
    assert assessment.risk_level == "low"


def test_detects_duplicate_and_recurring_signals_from_history() -> None:
    detector = ChargeAnomalyDetector()
    extracted = ExtractedFields(
        vendor="Acme Subscriptions",
        total=19.99,
        tax=0.0,
        date="2026-04-10",
        line_items=[LineItem(description="Plan", amount=19.99)],
    )

    assessment = detector.assess(
        extracted,
        historical_totals=[19.99, 19.99, 19.99, 20.0, 19.95, 20.1],
        historical_records=[
            {"vendor": "Acme Subscriptions", "total": 19.99, "date": "2026-04-10"},
            {"vendor": "Acme Subscriptions", "total": 19.99, "date": "2026-03-10"},
            {"vendor": "Acme Subscriptions", "total": 19.99, "date": "2026-02-10"},
        ],
    )

    codes = {signal.code for signal in assessment.signals}
    assert "duplicate_charge" in codes
    assert "recurring_payment" in codes
