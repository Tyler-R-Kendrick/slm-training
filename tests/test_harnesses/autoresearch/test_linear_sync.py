"""Tests for the AP-038 / SLM-339 idempotent Linear evidence sync."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from slm_training.harnesses.autoresearch import linear_sync as ls
from slm_training.harnesses.autoresearch.planning_result import (
    AbstractPlanningResultV1,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "planning_result"
    / "ap037_fixture.json"
)
CANARY_KEY = "lin_api_CANARYSECRET0123456789abcdef"


def _result() -> AbstractPlanningResultV1:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return AbstractPlanningResultV1.from_dict(payload)


class FakeTransport:
    """Recording transport with scriptable failures. Zero network."""

    def __init__(self, comments: list[dict[str, str]] | None = None) -> None:
        self.comments = list(comments or [])
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.failures: dict[str, list[Exception]] = {}

    def _maybe_fail(self, op: str) -> None:
        if self.failures.get(op):
            raise self.failures[op].pop(0)

    def list_comments(self, issue_key: str) -> list[dict[str, str]]:
        self.calls.append(("list_comments", (issue_key,)))
        self._maybe_fail("list_comments")
        return [dict(c) for c in self.comments]

    def create_comment(self, issue_key: str, body: str) -> dict[str, str]:
        self.calls.append(("create_comment", (issue_key, body)))
        self._maybe_fail("create_comment")
        comment = {"id": f"c-{len(self.comments) + 1}", "body": body}
        self.comments.append(comment)
        return {"id": comment["id"]}

    def update_comment(self, comment_id: str, body: str) -> dict[str, str]:
        self.calls.append(("update_comment", (comment_id, body)))
        self._maybe_fail("update_comment")
        for comment in self.comments:
            if comment["id"] == comment_id:
                comment["body"] = body
        return {"id": comment_id}


def _changed(result: AbstractPlanningResultV1) -> AbstractPlanningResultV1:
    payload = result.to_dict()
    payload["latency"]["total_seconds"] = (payload["latency"].get("total_seconds") or 1.0) + 5.0
    return AbstractPlanningResultV1.from_dict(payload)


# --- key resolution / comment body / marker ---


def test_resolve_issue_key_from_campaign_id() -> None:
    assert ls.resolve_issue_key(_result()) == "AP-037"


def test_resolve_issue_key_errors_without_ap_key() -> None:
    payload = _result().to_dict()
    payload["campaign_id"] = "random-experiment-name"
    with pytest.raises(ls.IssueKeyError):
        ls.resolve_issue_key(AbstractPlanningResultV1.from_dict(payload))


def test_comment_is_deterministic_and_carries_marker() -> None:
    result = _result()
    first = ls.build_evidence_comment(result)
    assert first == ls.build_evidence_comment(result)
    assert ls.comment_marker(result) in first
    assert result.content_sha256() in ls.comment_marker(result)


def test_comment_gate_summary_textual_per_claim_class() -> None:
    for claim, needle in (
        ("diagnostic", "cannot alter ship/promotion gates"),
        ("fixture", "NOT ship evidence"),
        ("promotion_candidate", "never promotes"),
        ("ship_gate", "never ships"),
    ):
        payload = _result().to_dict()
        payload["claim_class"] = claim
        body = ls.build_evidence_comment(AbstractPlanningResultV1.from_dict(payload))
        assert needle in body


# --- dry run / idempotency / update ---


def test_dry_run_makes_zero_transport_calls() -> None:
    transport = FakeTransport()
    report = ls.sync_result(_result(), transport, dry_run=True)
    assert report.action == "dry-run"
    assert transport.calls == []
    assert "planned" in report.resume


def test_idempotent_replay_creates_no_duplicate() -> None:
    result = _result()
    transport = FakeTransport()
    first = ls.sync_result(result, transport, dry_run=False)
    assert first.action == "create"
    second = ls.sync_result(result, transport, dry_run=False)
    assert second.action == "noop"
    assert len(transport.comments) == 1
    ops = [name for name, _ in transport.calls]
    assert ops.count("create_comment") == 1


def test_human_comments_are_never_touched() -> None:
    human = {"id": "human-1", "body": "looks like evidence but has no marker"}
    transport = FakeTransport([human])
    report = ls.sync_result(_result(), transport, dry_run=False)
    assert report.action == "create"
    assert transport.comments[0] == human


def test_changed_result_updates_machine_comment_only() -> None:
    result = _result()
    transport = FakeTransport()
    ls.sync_result(result, transport, dry_run=False)
    updated = ls.sync_result(_changed(result), transport, dry_run=False)
    assert updated.action == "update"
    assert len(transport.comments) == 1
    assert ls.comment_marker(_changed(result)) in transport.comments[0]["body"]


# --- retry / backoff / rate limit / partial failure ---


def test_retry_on_429_then_success_honors_retry_after() -> None:
    result = _result()
    transport = FakeTransport()
    transport.failures["list_comments"] = [ls.RateLimitError("slow down", retry_after=7.5)]
    sleeps: list[float] = []
    report = ls.sync_result(
        result, transport, dry_run=False, base_delay=0.01, sleeper=sleeps.append
    )
    assert report.action == "create"
    assert sleeps == [7.5]
    assert report.attempts == 1


def test_retry_on_5xx_uses_bounded_backoff() -> None:
    result = _result()
    transport = FakeTransport()
    transport.failures["list_comments"] = [
        ls.TransportError("boom", status=500),
        ls.TransportError("boom", status=503),
    ]
    sleeps: list[float] = []
    report = ls.sync_result(
        result, transport, dry_run=False, base_delay=0.5, sleeper=sleeps.append
    )
    assert report.action == "create"
    assert sleeps == [0.5, 1.0]


def test_partial_failure_leaves_resumable_report() -> None:
    result = _result()
    transport = FakeTransport()
    transport.failures["create_comment"] = [ls.TransportError("down", status=500)] * 4
    report = ls.sync_result(
        result, transport, dry_run=False, base_delay=0.0, sleeper=lambda _: None
    )
    assert report.action == "error"
    assert report.error
    assert report.resume["issue_key"] == "AP-037"
    assert "body" in report.resume
    # Resume: same call succeeds and is idempotent.
    recovered = ls.sync_result(
        result, transport, dry_run=False, base_delay=0.0, sleeper=lambda _: None
    )
    assert recovered.action == "create"


# --- comment-only claim classes ---


class EvilTransport(FakeTransport):
    def update_issue_status(self, *_: Any) -> None:  # pragma: no cover
        raise AssertionError("unreachable")


def test_diagnostic_claim_cannot_mutate_status() -> None:
    result = _result()  # fixture claim_class is "diagnostic"
    assert result.claim_class in ls.COMMENT_ONLY_CLASSES
    guard = ls._CommentOnlyGuard(EvilTransport())
    with pytest.raises(AssertionError):
        guard.update_issue_status("done")
    transport = EvilTransport()
    ls.sync_result(result, transport, dry_run=False)
    assert {name for name, _ in transport.calls} <= {
        "list_comments",
        "create_comment",
        "update_comment",
    }


# --- offline transport parity ---


def test_offline_payload_matches_live_graphql() -> None:
    result = _result()
    offline = ls.OfflineTransport()
    ls.sync_result(result, offline, dry_run=False)
    queries = [p["query"] for p in offline.payloads]
    assert ls.COMMENTS_QUERY in queries
    assert ls.ISSUE_LOOKUP_QUERY in queries
    assert ls.COMMENT_CREATE_MUTATION in queries
    body = ls.build_evidence_comment(result)
    create = offline.payloads[-1]
    assert create["variables"]["body"] == body


# --- secret handling ---


def test_missing_api_key_is_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    with pytest.raises(ls.MissingApiKeyError) as excinfo:
        ls.LinearGraphQLTransport()
    assert "LINEAR_API_KEY" in str(excinfo.value)


def test_http_error_redacts_api_key() -> None:
    transport = ls.LinearGraphQLTransport(api_key=CANARY_KEY)

    def boom(request: Any, timeout: float = 0.0) -> Any:
        body = f"bad key {CANARY_KEY} rejected".encode("utf-8")
        error = urllib.error.HTTPError(
            "https://api.linear.app/graphql", 401, "unauthorized", hdrs=None, fp=BytesIO(body)
        )
        raise error

    transport._opener = boom
    with pytest.raises(ls.TransportError) as excinfo:
        transport.list_comments("AP-037")
    assert CANARY_KEY not in str(excinfo.value)
    assert "<redacted-linear-api-key>" in str(excinfo.value)


def test_no_secret_in_comment_report_or_offline_payloads() -> None:
    result = _result()
    transport = FakeTransport()
    report = ls.sync_result(result, transport, dry_run=False)
    blob = json.dumps(report.to_dict()) + ls.build_evidence_comment(result)
    offline = ls.OfflineTransport()
    ls.sync_result(result, offline, dry_run=False)
    blob += json.dumps(offline.payloads)
    assert CANARY_KEY not in blob
    assert "api_key" not in blob.lower()
