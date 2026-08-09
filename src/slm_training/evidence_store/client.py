"""Fail-soft client for the durable evidence store.

Remote backend is Supabase Postgres reached through PostgREST
(``{SUPABASE_URL}/rest/v1/evidence_records``). The remote is strictly
optional: when ``SUPABASE_URL`` / ``SUPABASE_SERVICE_ROLE_KEY`` are unset —
or any network call fails — every entry point degrades to the committed local
mirror with a single logged notice and never raises. ``httpx`` is not a core
dependency of this repo (dev-extra only), so HTTP goes through
``urllib.request`` from the stdlib behind an injectable transport callable
(tests inject a fake transport; no real network in tests).

``find_prior_attempts`` is the stable consumer surface (guarded-imported by
the continuous loop's closeout); its signature is frozen.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from slm_training.evidence_store.local_index import (
    DEFAULT_LOCAL_INDEX_PATH,
    load_local_index,
)
from slm_training.evidence_store.records import EvidenceRecordV1

_LOGGER = logging.getLogger(__name__)

SUPABASE_URL_ENV = "SUPABASE_URL"
SUPABASE_KEY_ENV = "SUPABASE_SERVICE_ROLE_KEY"
EVIDENCE_TABLE = "evidence_records"
_DEFAULT_TIMEOUT_S = 20.0
_REMOTE_QUERY_LIMIT = 100

#: process-wide flag so the "remote disabled" notice logs exactly once.
_NOTICE_EMITTED = False


class TransportResponse(NamedTuple):
    status: int
    body: bytes


#: (method, url, headers, body, timeout_s) -> TransportResponse
Transport = Callable[
    [str, str, Mapping[str, str], bytes | None, float], TransportResponse
]


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_s: float,
) -> TransportResponse:
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return TransportResponse(int(response.status), response.read())
    except urllib.error.HTTPError as exc:  # non-2xx still carries a body
        return TransportResponse(int(exc.code), exc.read())


@dataclass(frozen=True)
class SupabaseConfig:
    """Resolved Supabase connection settings (both env vars required)."""

    url: str
    service_role_key: str

    @classmethod
    def from_env(cls) -> "SupabaseConfig | None":
        url = os.environ.get(SUPABASE_URL_ENV, "").strip()
        key = os.environ.get(SUPABASE_KEY_ENV, "").strip()
        if not url or not key:
            return None
        return cls(url=url.rstrip("/"), service_role_key=key)


def _notice_remote_disabled() -> None:
    global _NOTICE_EMITTED
    if _NOTICE_EMITTED:
        return
    _NOTICE_EMITTED = True
    _LOGGER.info(
        "evidence_store: %s/%s unset — remote sync disabled; "
        "using the committed local index only",
        SUPABASE_URL_ENV,
        SUPABASE_KEY_ENV,
    )


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _local_match_score(
    record: EvidenceRecordV1,
    hypothesis_text: str | None,
    lever_keys: Sequence[str] | None,
) -> float:
    """Substring + token-overlap relevance score against one local record."""
    score = 0.0
    if hypothesis_text:
        query_tokens = _tokens(hypothesis_text)
        record_tokens = _tokens(record.hypothesis_text) | _tokens(
            " ".join(record.lever_keys)
        )
        if query_tokens:
            score += len(query_tokens & record_tokens) / len(query_tokens)
        needle = hypothesis_text.strip().lower()
        if needle and needle in record.hypothesis_text.lower():
            score += 1.0
    if lever_keys:
        record_levers = {key.lower() for key in record.lever_keys}
        wanted = {key.lower() for key in lever_keys if key}
        if wanted:
            score += len(wanted & record_levers) / len(wanted)
    return score


class EvidenceStoreClient:
    """Query/upsert client with automatic local-index degradation."""

    def __init__(
        self,
        config: SupabaseConfig | None = None,
        *,
        local_index_path: Path | None = None,
        transport: Transport | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._config = config if config is not None else SupabaseConfig.from_env()
        self._local_index_path = local_index_path or DEFAULT_LOCAL_INDEX_PATH
        self._transport = transport or _urllib_transport
        self._timeout_s = timeout_s

    @property
    def remote_enabled(self) -> bool:
        return self._config is not None

    # -- remote plumbing ---------------------------------------------------

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        assert self._config is not None
        headers = {
            "apikey": self._config.service_role_key,
            "Authorization": f"Bearer {self._config.service_role_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _rest_url(self, params: Mapping[str, str] | None = None) -> str:
        assert self._config is not None
        url = f"{self._config.url}/rest/v1/{EVIDENCE_TABLE}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return url

    def upsert_records(
        self, records: Sequence[EvidenceRecordV1], *, batch_size: int = 200
    ) -> int:
        """Batched PostgREST upsert; returns rows pushed (0 when degraded).

        Never raises: missing credentials or any transport/HTTP failure logs
        one notice/warning and returns the count pushed so far.
        """
        if not records:
            return 0
        if self._config is None:
            _notice_remote_disabled()
            return 0
        pushed = 0
        url = self._rest_url(
            {"on_conflict": "config_fingerprint,endpoint_metric"}
        )
        headers = self._headers(
            prefer="resolution=merge-duplicates,return=minimal"
        )
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            body = json.dumps(
                [_row_from_record(record) for record in batch], default=str
            ).encode("utf-8")
            try:
                response = self._transport(
                    "POST", url, headers, body, self._timeout_s
                )
            except Exception as exc:  # noqa: BLE001 — fail-soft by contract
                _LOGGER.warning(
                    "evidence_store: upsert transport failure (%s); "
                    "pushed %d/%d records",
                    exc,
                    pushed,
                    len(records),
                )
                return pushed
            if response.status not in (200, 201, 204):
                _LOGGER.warning(
                    "evidence_store: upsert rejected (HTTP %d): %s",
                    response.status,
                    response.body[:200].decode("utf-8", "replace"),
                )
                return pushed
            pushed += len(batch)
        return pushed

    def _remote_query(self, params: dict[str, str]) -> list[EvidenceRecordV1] | None:
        """One PostgREST GET; ``None`` signals degradation to the local index."""
        if self._config is None:
            return None
        params.setdefault("limit", str(_REMOTE_QUERY_LIMIT))
        try:
            response = self._transport(
                "GET", self._rest_url(params), self._headers(), None, self._timeout_s
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft by contract
            _LOGGER.warning(
                "evidence_store: remote query failed (%s); using local index", exc
            )
            return None
        if response.status != 200:
            _LOGGER.warning(
                "evidence_store: remote query HTTP %d; using local index",
                response.status,
            )
            return None
        try:
            rows = json.loads(response.body.decode("utf-8"))
        except ValueError:
            _LOGGER.warning(
                "evidence_store: unparseable remote payload; using local index"
            )
            return None
        records = []
        for row in rows if isinstance(rows, list) else []:
            try:
                records.append(_record_from_row(row))
            except ValueError:
                continue
        return records

    # -- lookup ------------------------------------------------------------

    def find_prior_attempts(
        self,
        hypothesis_text: str | None = None,
        lever_keys: Sequence[str] | None = None,
        config_fingerprint: str | None = None,
    ) -> list[EvidenceRecordV1]:
        """Prior attempts matching the query, best match first. Never raises.

        Exact ``config_fingerprint`` matches always rank first. With
        credentials present the hypothesis search uses PostgREST full-text
        search (``wfts`` over the generated tsvector); otherwise — or on any
        remote failure — the committed local index is scored by substring +
        token overlap.
        """
        results: list[EvidenceRecordV1] = []
        seen: set[tuple[str, str]] = set()

        def _extend(records: Sequence[EvidenceRecordV1]) -> None:
            for record in records:
                key = record.dedup_key()
                if key not in seen:
                    seen.add(key)
                    results.append(record)

        if self._config is None:
            _notice_remote_disabled()

        if config_fingerprint:
            exact = self._remote_query(
                {"config_fingerprint": f"eq.{config_fingerprint}"}
            )
            if exact is None:
                exact = [
                    record
                    for record in self._load_local()
                    if record.config_fingerprint == config_fingerprint
                ]
            _extend(
                sorted(exact, key=lambda record: record.dedup_key())
            )

        if hypothesis_text or lever_keys:
            remote: list[EvidenceRecordV1] | None = None
            if hypothesis_text:
                remote = self._remote_query(
                    {"search_tsv": f"wfts.{hypothesis_text}"}
                )
            if remote is not None and lever_keys:
                remote = [
                    record
                    for record in remote
                    if _local_match_score(record, None, lever_keys) > 0
                ] or remote
            if remote is not None and hypothesis_text:
                _extend(remote)
            else:
                scored = [
                    (
                        _local_match_score(record, hypothesis_text, lever_keys),
                        record,
                    )
                    for record in self._load_local()
                ]
                ranked = sorted(
                    (item for item in scored if item[0] > 0.0),
                    key=lambda item: (-item[0], item[1].dedup_key()),
                )
                _extend([record for _, record in ranked])

        return results

    def _load_local(self) -> list[EvidenceRecordV1]:
        return load_local_index(self._local_index_path)


def _row_from_record(record: EvidenceRecordV1) -> dict[str, Any]:
    """Map a record onto the ``evidence_records`` table columns."""
    row = record.model_dump(mode="json")
    row.pop("schema_version", None)
    row["lever_keys_text"] = " ".join(record.lever_keys)
    return row


def _record_from_row(row: Mapping[str, Any]) -> EvidenceRecordV1:
    payload = {
        key: value
        for key, value in row.items()
        if key in EvidenceRecordV1.model_fields
    }
    return EvidenceRecordV1.model_validate(payload)


def find_prior_attempts(
    hypothesis_text: str | None = None,
    lever_keys: Sequence[str] | None = None,
    config_fingerprint: str | None = None,
) -> list[EvidenceRecordV1]:
    """Module-level stable entry point (frozen signature; never raises)."""
    try:
        client = EvidenceStoreClient()
        return client.find_prior_attempts(
            hypothesis_text=hypothesis_text,
            lever_keys=lever_keys,
            config_fingerprint=config_fingerprint,
        )
    except Exception as exc:  # noqa: BLE001 — consumer surface must not raise
        _LOGGER.warning("evidence_store: find_prior_attempts degraded (%s)", exc)
        return []
