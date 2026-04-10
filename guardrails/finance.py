from __future__ import annotations

import re

FINANCE_TOPICS = {
    "invoice",
    "invoices",
    "tax",
    "taxes",
    "transaction",
    "transactions",
    "receipt",
    "receipts",
    "subscription",
    "subscriptions",
    "charge",
    "charges",
    "billing",
    "payment",
    "payments",
    "refund",
    "refunds",
    "fee",
    "fees",
    "merchant",
    "vendor",
    "bank",
    "statement",
    "expense",
    "expenses",
    "deducted",
    "deduction",
    "credit",
    "debit",
    "balance",
}

MONEY_PATTERN = re.compile(r"(?:\$|€|£|usd|eur|gbp|inr)\s*\d", flags=re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d{2})\b")


def finance_classifier(query: str) -> str:
    text = query.strip().lower()
    if not text:
        return "non_finance"

    tokens = set(re.findall(r"[a-z0-9_]+", text))
    if tokens.intersection(FINANCE_TOPICS):
        return "finance_related"

    if MONEY_PATTERN.search(text) and any(token in text for token in ["charge", "payment", "invoice", "receipt"]):
        return "finance_related"

    if "statement" in text and AMOUNT_PATTERN.search(text):
        return "finance_related"

    return "non_finance"


def block_non_finance(query: str) -> dict[str, str] | None:
    if finance_classifier(query) == "non_finance":
        return {"error": "This assistant only handles finance-related queries."}
    return None
