"""Idempotent Linear evidence sync for abstract-planning results (AP-038 / SLM-339).

Posts or updates exactly ONE machine-owned evidence comment on the Linear issue
matching an ``AbstractPlanningResultV1`` campaign. The sync is dry-run-first,
idempotent (content-sha marker), and read-only with respect to everything but
its own comment: it never touches human comments, labels, status, assignee, or
priority, and diagnostic/fixture claim classes are hard-asserted to be
comment-only (no gate or promotion state mutation, ever).

Transports:

- ``LinearGraphQLTransport`` — live GraphQL over urllib; reads the API key from
  the environment ONLY and redacts it from every error message.
- ``OfflineTransport`` — records the exact GraphQL payloads the live path would
  send and writes them to a file / stdout for CI review. Zero network.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from slm_training.harnesses.autoresearch.planning_result import (
    AbstractPlanningResultV1,
    CLAIM_CLASSES,
    PROMOTION_CLASSES,
)

COMPONENT_ID = "harness.experiments.slm339_linear_sync"
SCHEMA_ID = "LinearSyncReportV1"
LINEAR_API_URL = "https://api.linear.app/graphql"
DEFAULT_API_KEY_ENV = "LINEAR_API_KEY"
MARKER_RE = re.compile(r"<!--\s*ap-sync:sha256=([0-9a-f]{64})\s*-->")
_AP_KEY_RE = re.compile(r"ap[-_ ]?0*(\d+)", re.IGNORECASE)
# Claim classes that must never alter gate/promotion state anywhere. The sync
# surface has no such mutation methods at all; the guard below makes that an
# executable assertion instead of a comment.
COMMENT_ONLY_CLASSES: frozenset[str] = CLAIM_CLASSES - PROMOTION_CLASSES

GATE_SUMMARY: Mapping[str, str] = {
    "wiring": "Wiring evidence only — no gate signal; nothing is promotable.",
    "fixture": "Fixture evidence only — demo wiring, NOT ship evidence; no gate effect.",
    "diagnostic": "Diagnostic evidence — informs research; cannot alter ship/promotion gates.",
    "screening": "Screening evidence — pre-confirmatory; no promotion or ship-gate effect.",
    "promotion_candidate": "Promotion-candidate evidence — feeds the human promotion review; this comment never promotes.",
    "ship_gate": "Ship-gate evidence — feeds the human ship review; this comment never ships or closes gates.",
}


class LinearSyncError(RuntimeError):
    """Base error for the Linear sync."""


class IssueKeyError(LinearSyncError):
    """The campaign_id carries no resolvable AP key."""


class MissingApiKeyError(LinearSyncError):
    """Live mode requested without the configured API-key env var."""


class TransportError(LinearSyncError):
    """Non-retryable or retry-exhausted transport failure (key-redacted)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RateLimitError(TransportError):
    """HTTP 429 — honors the server-provided retry delay."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, status=429)
        self.retry_after = retry_after


def resolve_issue_key(result: AbstractPlanningResultV1) -> str:
    """Stable ``AP-###`` key extracted from the campaign_id (e.g. ``AP-037``)."""
    match = _AP_KEY_RE.search(result.campaign_id or "")
    if not match:
        raise IssueKeyError(
            f"campaign_id {result.campaign_id!r} has no AP-### key; "
            "cannot resolve the Linear issue"
        )
    return f"AP-{int(match.group(1)):03d}"


def comment_marker(result: AbstractPlanningResultV1) -> str:
    """Idempotency marker embedded in (and only in) the machine comment."""
    return f"<!-- ap-sync:sha256={result.content_sha256()} -->"


def _marker_sha(body: str) -> str | None:
    match = MARKER_RE.search(body or "")
    return match.group(1) if match else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_evidence_comment(result: AbstractPlanningResultV1) -> str:
    """Deterministic machine-owned comment body built from the result object only.

    Contains no credentials, no environment data, and no mutable issue state.
    Ends with the idempotency marker.
    """
    key = resolve_issue_key(result)
    gate_summary = GATE_SUMMARY.get(result.claim_class, GATE_SUMMARY["diagnostic"])
    lines: list[str] = [
        f"### Autoresearch evidence — `{key}` (machine-owned, do not edit)",
        "",
        f"- Campaign: `{result.campaign_id}`",
        f"- Campaign manifest sha256: `{result.campaign_manifest_sha256}`",
        f"- Claim class / disposition: **{result.claim_class}**",
        f"- Source: `{result.source_commit}` (dirty: {result.source_dirty})",
        f"- Data snapshot sha256: `{result.data_snapshot_sha256}`",
        f"- Gate state: {gate_summary}",
        "",
        "| Path | n | meaning-v2 | binder F1 | parse |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in result.metrics:
        lines.append(
            f"| {item.path} | {item.n} | {_fmt(item.meaning_v2)} "
            f"| {_fmt(item.binder_reference_f1)} | {_fmt(item.parse_rate)} |"
        )
    if result.latency.total_seconds is not None:
        lines += [
            "",
            f"Latency total {_fmt(result.latency.total_seconds)}s "
            f"(p95 {_fmt(result.latency.p95_seconds)}s).",
        ]
    gates = [f"{g.gate_id}={'pass' if g.passed else 'FAIL'}" for g in result.verifier_gates]
    if gates:
        lines += ["", f"Verifier gates: {', '.join(gates)}."]
    for label, ref in (("AgentV", result.agentv), ("Human audit", result.human_audit)):
        if ref is not None:
            lines.append(f"- {label}: `{ref.path}` (n={ref.n})")
    lines += [
        "",
        "This comment is written and maintained by the AP-038 linear-sync harness. "
        "It never changes status, labels, assignee, priority, gates, or promotion state.",
        "",
        comment_marker(result),
    ]
    return "\n".join(lines) + "\n"


# --- GraphQL payloads (shared verbatim by the live and offline transports) ---

ISSUE_LOOKUP_QUERY = (
    "query ApSyncIssueLookup($key: String!) { issue(id: $key) { id identifier } }"
)
COMMENTS_QUERY = (
    "query ApSyncComments($key: String!) { issue(id: $key) { id identifier "
    "comments(first: 250) { nodes { id body } } } }"
)
COMMENT_CREATE_MUTATION = (
    "mutation ApSyncCommentCreate($issueId: String!, $body: String!) { "
    "commentCreate(input: {issueId: $issueId, body: $body}) { comment { id } } }"
)
COMMENT_UPDATE_MUTATION = (
    "mutation ApSyncCommentUpdate($id: String!, $body: String!) { "
    "commentUpdate(id: $id, input: {body: $body}) { comment { id } } }"
)


def _payload(query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
    return {"query": query, "variables": dict(variables)}


def issue_lookup_payload(issue_key: str) -> dict[str, Any]:
    return _payload(ISSUE_LOOKUP_QUERY, {"key": issue_key})


def comments_payload(issue_key: str) -> dict[str, Any]:
    return _payload(COMMENTS_QUERY, {"key": issue_key})


def comment_create_payload(issue_id: str, body: str) -> dict[str, Any]:
    return _payload(COMMENT_CREATE_MUTATION, {"issueId": issue_id, "body": body})


def comment_update_payload(comment_id: str, body: str) -> dict[str, Any]:
    return _payload(COMMENT_UPDATE_MUTATION, {"id": comment_id, "body": body})


@runtime_checkable
class SyncTransport(Protocol):
    """Comment-only transport surface. There are intentionally no issue-field
    mutations here — the sync cannot touch status/labels/assignee/priority."""

    def list_comments(self, issue_key: str) -> list[dict[str, str]]:
        """Return ``[{"id": ..., "body": ...}]`` for the issue."""
        ...

    def create_comment(self, issue_key: str, body: str) -> dict[str, str]:
        ...

    def update_comment(self, comment_id: str, body: str) -> dict[str, str]:
        ...


def _redact(text: str, api_key: str | None) -> str:
    if api_key:
        text = text.replace(api_key, "<redacted-linear-api-key>")
    return text


class LinearGraphQLTransport:
    """Live Linear GraphQL transport (urllib only; stdlib).

    The API key is read from ``api_key_env`` (default ``LINEAR_API_KEY``) and
    never logged; it is redacted from every raised error message.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        url: str = LINEAR_API_URL,
        timeout: float = 30.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not self._api_key:
            raise MissingApiKeyError(
                f"live Linear sync requires the {api_key_env} environment variable; "
                "it is not set"
            )
        self._url = url
        self._timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def _execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": self._api_key or "",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - body is best-effort only
                body = ""
            detail = _redact(f"Linear GraphQL HTTP {exc.code}: {body[:500]}", self._api_key)
            if exc.code == 429:
                retry_after: float | None = None
                raw = exc.headers.get("Retry-After") if exc.headers else None
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        retry_after = None
                raise RateLimitError(detail, retry_after=retry_after) from None
            raise TransportError(detail, status=exc.code) from None
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise TransportError(
                _redact(f"Linear GraphQL request failed: {exc}", self._api_key)
            ) from None
        errors = data.get("errors")
        if errors:
            raise TransportError(
                _redact(f"Linear GraphQL errors: {json.dumps(errors)[:500]}", self._api_key)
            )
        return data.get("data", {})

    def _issue_id(self, issue_key: str) -> str:
        data = self._execute(issue_lookup_payload(issue_key))
        issue = data.get("issue") or {}
        issue_id = issue.get("id")
        if not issue_id:
            raise TransportError(f"Linear issue {issue_key!r} not found")
        return str(issue_id)

    def list_comments(self, issue_key: str) -> list[dict[str, str]]:
        data = self._execute(comments_payload(issue_key))
        issue = data.get("issue") or {}
        if not issue.get("id"):
            raise TransportError(f"Linear issue {issue_key!r} not found")
        nodes = (issue.get("comments") or {}).get("nodes") or []
        return [{"id": str(n["id"]), "body": str(n.get("body") or "")} for n in nodes]

    def create_comment(self, issue_key: str, body: str) -> dict[str, str]:
        data = self._execute(comment_create_payload(self._issue_id(issue_key), body))
        comment = (data.get("commentCreate") or {}).get("comment") or {}
        return {"id": str(comment.get("id", ""))}

    def update_comment(self, comment_id: str, body: str) -> dict[str, str]:
        data = self._execute(comment_update_payload(comment_id, body))
        comment = (data.get("commentUpdate") or {}).get("comment") or {}
        return {"id": str(comment.get("id", comment_id))}


class OfflineTransport:
    """Zero-network transport: records the exact GraphQL payloads the live path
    would send, for CI review. ``list_comments`` replays injected comments.
    """

    def __init__(
        self,
        *,
        existing_comments: Sequence[Mapping[str, str]] = (),
        sink: Callable[[str], None] | None = None,
    ) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._comments = [dict(c) for c in existing_comments]
        self._sink = sink

    def _record(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)
        if self._sink is not None:
            self._sink(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def list_comments(self, issue_key: str) -> list[dict[str, str]]:
        self._record(comments_payload(issue_key))
        return [dict(c) for c in self._comments]

    def create_comment(self, issue_key: str, body: str) -> dict[str, str]:
        self._record(issue_lookup_payload(issue_key))
        self._record(comment_create_payload(f"offline-id:{issue_key}", body))
        return {"id": f"offline-comment:{issue_key}"}

    def update_comment(self, comment_id: str, body: str) -> dict[str, str]:
        self._record(comment_update_payload(comment_id, body))
        return {"id": comment_id}


@dataclass(frozen=True)
class SyncReport:
    """Outcome of one sync. ``resume`` carries everything needed to retry a
    partial failure (no secrets, no transport state)."""

    schema: str
    issue_key: str
    action: str  # "dry-run" | "noop" | "create" | "update" | "error"
    content_sha256: str
    marker: str
    dry_run: bool = True
    comment_id: str | None = None
    attempts: int = 0
    error: str | None = None
    resume: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "issue_key": self.issue_key,
            "action": self.action,
            "content_sha256": self.content_sha256,
            "marker": self.marker,
            "dry_run": self.dry_run,
            "comment_id": self.comment_id,
            "attempts": self.attempts,
            "error": self.error,
            "resume": dict(self.resume),
        }


class _CommentOnlyGuard:
    """Assertion wrapper: for comment-only claim classes (wiring / fixture /
    diagnostic / screening) ANY transport call outside the three comment
    methods is a hard failure. The surface has none; this keeps it that way."""

    _ALLOWED = frozenset({"list_comments", "create_comment", "update_comment"})

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def __getattr__(self, name: str) -> Any:
        assert name in self._ALLOWED, (
            f"comment-only claim class attempted transport mutation {name!r}; "
            "linear sync may only create/update its own evidence comment"
        )
        return getattr(self._transport, name)


def _call_with_retry(
    fn: Callable[[], Any],
    *,
    max_retries: int,
    base_delay: float,
    sleeper: Callable[[float], None],
    state: dict[str, int],
) -> Any:
    attempt = 0
    while True:
        try:
            return fn()
        except RateLimitError as exc:
            attempt += 1
            state["attempts"] += 1
            if attempt > max_retries:
                raise
            # Rate-limit-aware delay: honor Retry-After, else bounded backoff.
            sleeper(exc.retry_after if exc.retry_after is not None else base_delay * (2 ** (attempt - 1)))
        except TransportError as exc:
            if exc.status is None or not (500 <= exc.status < 600):
                raise
            attempt += 1
            state["attempts"] += 1
            if attempt > max_retries:
                raise
            sleeper(min(base_delay * (2 ** (attempt - 1)), 30.0))


def sync_result(
    result: AbstractPlanningResultV1,
    transport: SyncTransport | None = None,
    *,
    dry_run: bool = True,
    max_retries: int = 3,
    base_delay: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> SyncReport:
    """Synchronize one result to its Linear issue as a single machine comment.

    ``dry_run=True`` (default) makes ZERO transport calls and reports the
    planned action plus the idempotency marker. Live mode: same content sha
    → NO-OP; different sha → update the machine comment; none → create.
    Human comments (no marker) are never touched.
    """
    issue_key = resolve_issue_key(result)
    body = build_evidence_comment(result)
    marker = comment_marker(result)
    sha = result.content_sha256()
    if dry_run:
        return SyncReport(
            schema=SCHEMA_ID,
            issue_key=issue_key,
            action="dry-run",
            content_sha256=sha,
            marker=marker,
            dry_run=True,
            resume={
                "planned": "create-or-update single machine-owned comment",
                "body": body,
            },
        )
    guarded: SyncTransport
    if transport is None:
        raise LinearSyncError("live sync requires a transport")
    guarded = transport
    if result.claim_class in COMMENT_ONLY_CLASSES:
        guarded = _CommentOnlyGuard(transport)
    state = {"attempts": 0}
    retry = {
        "max_retries": max_retries,
        "base_delay": base_delay,
        "sleeper": sleeper,
        "state": state,
    }
    try:
        comments = _call_with_retry(lambda: guarded.list_comments(issue_key), **retry)  # type: ignore[arg-type]
        machine = next((c for c in comments if _marker_sha(c.get("body", ""))), None)
        if machine is not None and _marker_sha(machine["body"]) == sha:
            return SyncReport(
                schema=SCHEMA_ID,
                issue_key=issue_key,
                action="noop",
                content_sha256=sha,
                marker=marker,
                dry_run=False,
                comment_id=machine["id"],
                attempts=state["attempts"],
            )
        if machine is not None:
            updated = _call_with_retry(
                lambda: guarded.update_comment(machine["id"], body), **retry  # type: ignore[arg-type]
            )
            return SyncReport(
                schema=SCHEMA_ID,
                issue_key=issue_key,
                action="update",
                content_sha256=sha,
                marker=marker,
                dry_run=False,
                comment_id=updated.get("id", machine["id"]),
                attempts=state["attempts"],
            )
        created = _call_with_retry(lambda: guarded.create_comment(issue_key, body), **retry)  # type: ignore[arg-type]
        return SyncReport(
            schema=SCHEMA_ID,
            issue_key=issue_key,
            action="create",
            content_sha256=sha,
            marker=marker,
            dry_run=False,
            comment_id=created.get("id"),
            attempts=state["attempts"],
        )
    except TransportError as exc:
        # Partial failure: resumable report — replay sync_result with the same
        # result/transport; the marker keeps the retry idempotent.
        return SyncReport(
            schema=SCHEMA_ID,
            issue_key=issue_key,
            action="error",
            content_sha256=sha,
            marker=marker,
            dry_run=False,
            attempts=state["attempts"],
            error=str(exc),
            resume={"issue_key": issue_key, "body": body},
        )


def format_dry_run(report: SyncReport) -> str:
    """Human-readable dry-run plan (no network data)."""
    lines = [
        f"Issue key: {report.issue_key}",
        f"Planned action: {report.resume.get('planned', 'n/a')}",
        f"Idempotency marker: {report.marker}",
        "",
        "Planned comment:",
        str(report.resume.get("body", "")),
    ]
    return "\n".join(lines)
