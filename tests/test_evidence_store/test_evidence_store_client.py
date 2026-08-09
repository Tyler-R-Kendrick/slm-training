"""find_prior_attempts (local + mocked-transport remote) — no real network."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

import pytest

import slm_training.evidence_store.client as client_mod
from slm_training.evidence_store.client import (
    EvidenceStoreClient,
    SupabaseConfig,
    TransportResponse,
    find_prior_attempts,
)
from slm_training.evidence_store.local_index import write_local_index
from slm_training.evidence_store.records import EvidenceRecordV1


def _record(**overrides: object) -> EvidenceRecordV1:
    payload: dict[str, object] = {
        "experiment_id": "exp",
        "config_fingerprint": "a" * 64,
        "endpoint_metric": "smoke.structural_similarity",
        "outcome": "screen_negative",
    }
    payload.update(overrides)
    return EvidenceRecordV1.model_validate(payload)


@pytest.fixture()
def local_index(tmp_path: Path) -> Path:
    path = tmp_path / "local_index.jsonl"
    write_local_index(
        [
            _record(
                experiment_id="arm-bounds",
                config_fingerprint="b" * 64,
                hypothesis_text="Screening arm 'bounds' improves structural similarity",
                lever_keys=["bounds"],
            ),
            _record(
                experiment_id="arm-macro",
                config_fingerprint="c" * 64,
                hypothesis_text="Macro token vocabulary compresses container runs",
                lever_keys=["macro-tokens"],
                outcome="screen_positive",
            ),
            _record(
                experiment_id="unrelated",
                config_fingerprint="d" * 64,
                hypothesis_text="Perf arm lowers decode latency",
                lever_keys=["compiler-prefill"],
            ),
        ],
        path,
    )
    return path


class FakeTransport:
    """Records requests; replies from a queue of TransportResponse objects."""

    def __init__(self, responses: list[TransportResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> TransportResponse:
        self.calls.append((method, url, dict(headers), body))
        if not self.responses:
            raise AssertionError("unexpected extra request")
        return self.responses.pop(0)


def _no_creds_client(local_index: Path) -> EvidenceStoreClient:
    return EvidenceStoreClient(config=None, local_index_path=local_index)


# -- local mode --------------------------------------------------------------


def test_local_exact_fingerprint_match_first(local_index: Path) -> None:
    client = _no_creds_client(local_index)
    results = client.find_prior_attempts(
        hypothesis_text="bounds structural similarity",
        config_fingerprint="c" * 64,
    )
    assert results
    assert results[0].config_fingerprint == "c" * 64
    # hypothesis match still contributes further results
    assert any(record.config_fingerprint == "b" * 64 for record in results)


def test_local_hypothesis_token_overlap(local_index: Path) -> None:
    client = _no_creds_client(local_index)
    results = client.find_prior_attempts(
        hypothesis_text="macro token vocabulary containers"
    )
    assert results
    assert results[0].experiment_id == "arm-macro"


def test_local_lever_key_match(local_index: Path) -> None:
    client = _no_creds_client(local_index)
    results = client.find_prior_attempts(lever_keys=["bounds"])
    assert [record.experiment_id for record in results] == ["arm-bounds"]


def test_local_no_match_returns_empty(local_index: Path) -> None:
    client = _no_creds_client(local_index)
    assert client.find_prior_attempts(hypothesis_text="zzzz qqqq") == []


def test_missing_creds_logs_single_notice(
    local_index: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client_mod, "_NOTICE_EMITTED", False)
    client = _no_creds_client(local_index)
    with caplog.at_level(logging.INFO, logger=client_mod.__name__):
        client.find_prior_attempts(hypothesis_text="bounds")
        client.find_prior_attempts(hypothesis_text="bounds")
        client.upsert_records([_record()])
    notices = [
        record
        for record in caplog.records
        if "remote sync disabled" in record.getMessage()
    ]
    assert len(notices) == 1


def test_module_level_find_prior_attempts_never_raises(
    monkeypatch: pytest.MonkeyPatch, local_index: Path
) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr(
        client_mod, "DEFAULT_LOCAL_INDEX_PATH", local_index, raising=True
    )
    results = find_prior_attempts(lever_keys=["macro-tokens"])
    assert [record.experiment_id for record in results] == ["arm-macro"]


# -- remote mode (mocked transport) ------------------------------------------


def _creds_client(
    local_index: Path, transport: FakeTransport
) -> EvidenceStoreClient:
    return EvidenceStoreClient(
        config=SupabaseConfig(url="https://example.supabase.co", service_role_key="sk"),
        local_index_path=local_index,
        transport=transport,
    )


def test_upsert_batches_with_merge_duplicates(local_index: Path) -> None:
    transport = FakeTransport(
        [TransportResponse(201, b""), TransportResponse(201, b"")]
    )
    client = _creds_client(local_index, transport)
    records = [
        _record(config_fingerprint=f"{i:064x}", lever_keys=["bounds"])
        for i in range(3)
    ]
    assert client.upsert_records(records, batch_size=2) == 3
    assert len(transport.calls) == 2
    method, url, headers, body = transport.calls[0]
    assert method == "POST"
    assert url.startswith("https://example.supabase.co/rest/v1/evidence_records")
    assert "on_conflict=config_fingerprint%2Cendpoint_metric" in url
    assert headers["apikey"] == "sk"
    assert headers["Authorization"] == "Bearer sk"
    assert "resolution=merge-duplicates" in headers["Prefer"]
    rows = json.loads(body.decode("utf-8"))
    assert len(rows) == 2
    assert rows[0]["lever_keys_text"] == "bounds"
    assert "schema_version" not in rows[0]


def test_upsert_http_error_degrades_without_raising(local_index: Path) -> None:
    transport = FakeTransport([TransportResponse(500, b"boom")])
    client = _creds_client(local_index, transport)
    assert client.upsert_records([_record()]) == 0


def test_remote_fts_query_used_when_creds_present(local_index: Path) -> None:
    remote_row = _record(
        experiment_id="remote-hit",
        config_fingerprint="e" * 64,
        hypothesis_text="Macro tokens hypothesis",
    ).model_dump(mode="json")
    remote_row.pop("schema_version")
    transport = FakeTransport(
        [TransportResponse(200, json.dumps([remote_row]).encode("utf-8"))]
    )
    client = _creds_client(local_index, transport)
    results = client.find_prior_attempts(hypothesis_text="macro tokens")
    assert [record.experiment_id for record in results] == ["remote-hit"]
    method, url, _, body = transport.calls[0]
    assert method == "GET"
    assert body is None
    assert "search_tsv=wfts.macro+tokens" in url


def test_remote_failure_degrades_to_local(local_index: Path) -> None:
    def broken_transport(*args: object, **kwargs: object) -> TransportResponse:
        raise OSError("network down")

    client = EvidenceStoreClient(
        config=SupabaseConfig(url="https://example.supabase.co", service_role_key="sk"),
        local_index_path=local_index,
        transport=broken_transport,
    )
    results = client.find_prior_attempts(
        hypothesis_text="macro token vocabulary containers"
    )
    assert results
    assert results[0].experiment_id == "arm-macro"
