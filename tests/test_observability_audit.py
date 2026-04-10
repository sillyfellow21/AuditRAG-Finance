from __future__ import annotations


def test_audit_event_roundtrip(observability) -> None:
    event_id = observability.audit_event(
        actor="test-suite",
        action="unit_test",
        resource="audit",
        outcome="success",
        request_id="req-1",
        payload={"k": "v"},
    )

    rows = observability.read_audit_logs(limit=10)
    assert rows

    last = rows[-1]
    assert last["event_id"] == event_id
    assert last["action"] == "unit_test"
    assert last["outcome"] == "success"
