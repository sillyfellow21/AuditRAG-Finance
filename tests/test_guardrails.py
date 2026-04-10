from __future__ import annotations

from guardrails.finance import block_non_finance, finance_classifier


def test_finance_classifier_accepts_finance_query() -> None:
    assert finance_classifier("Why was this invoice charge deducted?") == "finance_related"


def test_finance_classifier_blocks_non_finance_query() -> None:
    assert finance_classifier("What is photosynthesis?") == "non_finance"


def test_block_non_finance_returns_required_error_payload() -> None:
    payload = block_non_finance("Tell me about the Roman Empire")
    assert payload == {"error": "This assistant only handles finance-related queries."}
