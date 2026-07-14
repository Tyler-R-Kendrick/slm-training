"""Tests for the observability + control-plane web layer.

Covers the read-only observability API, cold-start fallback + provenance, the
pure-compute gate endpoint, the execution capability gate, the job allowlist
(the security boundary), and an end-to-end job run.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates
from slm_training.web import jobs as jobs_mod
from slm_training.web.app import create_app
from slm_training.web.observability import Readers

SMOKE_SUITE = {
    "smoke": {
        "parse_rate": 0.9,
        "structural_similarity": 0.5,
        "placeholder_fidelity": 0.4,
        "reward_score": 0.5,
    }
}


@pytest.fixture
def ro_client() -> TestClient:
    with TestClient(create_app(execution=False)) as client:
        yield client


# --- observability reads ---------------------------------------------------
def test_overview_aggregates_committed_evidence(ro_client: TestClient) -> None:
    overview = ro_client.get("/api/overview").json()
    assert {"scoreboards", "experiment_totals", "checkpoints", "system"} <= set(overview)
    assert overview["experiment_totals"]["count"] >= 1


def test_scoreboard_unknown_kind_is_404(ro_client: TestClient) -> None:
    assert ro_client.get("/api/scoreboards/bogus").status_code == 404
    assert ro_client.get("/api/scoreboards/quality").json()["kind"] == "quality"


def test_checkpoints_roster_includes_fixture(ro_client: TestClient) -> None:
    roster = ro_client.get("/api/checkpoints").json()["checkpoints"]
    assert any("playground_demo" in (c.get("run_id") or "") for c in roster)


def test_spa_at_root_classic_playground_at_playground(ro_client: TestClient) -> None:
    """The SPA shell serves "/"; the classic annotate playground moves to /playground."""
    root = ro_client.get("/")
    assert root.status_code == 200
    assert 'id="root"' in root.text  # SPA mount point (built bundle committed)
    classic = ro_client.get("/playground")
    assert classic.status_code == 200
    assert "TwoTower" in classic.text
    # Client-side deep routes fall through to the SPA shell.
    assert 'id="root"' in ro_client.get("/checkpoints").text


def test_readers_cold_start_fallback(tmp_path) -> None:
    """An empty repo root must never raise; it reports committed provenance."""
    readers = Readers(tmp_path)
    assert readers.scoreboard("quality")["results"] == []
    assert readers.train_data()["provenance"] == "committed"
    assert readers.test_data()["provenance"] == "committed"
    assert readers.runs()["provenance"] == "committed"


# --- capability gate -------------------------------------------------------
def test_read_only_reports_capabilities(ro_client: TestClient) -> None:
    caps = ro_client.get("/api/capabilities").json()
    assert caps["execution"] is False
    assert caps["read_only"] is True
    assert len(caps["jobs"]) >= 5


def test_read_only_blocks_execution(ro_client: TestClient) -> None:
    resp = ro_client.post(
        "/api/jobs",
        json={"job": "build_train_data", "params": {"source": "fixture", "version": "v0"}},
    )
    assert resp.status_code == 403


# --- pure-compute gate endpoint (works even read-only) ---------------------
def test_gates_evaluate_matches_pure_function(ro_client: TestClient) -> None:
    thresholds = {"smoke": {"parse_rate": 0.66}}
    resp = ro_client.post(
        "/api/gates/evaluate", json={"suites": SMOKE_SUITE, "thresholds": thresholds}
    ).json()
    assert resp == evaluate_ship_gates(SMOKE_SUITE, thresholds=thresholds)
    assert resp["pass"] is True


# --- execution mode + allowlist (the security boundary) --------------------
def test_allowlist_rejects_unknown_and_malicious(tmp_path) -> None:
    with TestClient(create_app(execution=True, root=tmp_path)) as client:
        assert client.get("/api/capabilities").json()["execution"] is True
        assert client.post("/api/jobs", json={"job": "nope", "params": {}}).status_code == 400
        # shell-injection / path-escape attempts are rejected at validation.
        assert (
            client.post(
                "/api/jobs",
                json={"job": "build_train_data", "params": {"source": "x;rm -rf /", "version": "v0"}},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/jobs",
                json={"job": "build_test_data", "params": {"source": "both", "version": "../etc"}},
            ).status_code
            == 422
        )


def test_job_runs_to_completion(tmp_path, monkeypatch) -> None:
    # Inject a trivial stdlib-module job so the runner is exercised without the
    # full training toolchain. `python -m this` prints and exits 0.
    monkeypatch.setitem(jobs_mod.JOB_SPECS, "_zen", jobs_mod.JobSpec("this"))
    with TestClient(create_app(execution=True, root=tmp_path)) as client:
        job = client.post("/api/jobs", json={"job": "_zen", "params": {}}).json()
        job_id = job["id"]
        assert job["status"] in {"queued", "running"}
        status = _await_terminal(client, job_id)
        assert status == "succeeded"
        detail = client.get(f"/api/jobs/{job_id}").json()
        assert detail["returncode"] == 0
        assert len(detail["tail"]) > 0


def test_job_cancel(tmp_path, monkeypatch) -> None:
    # A long-lived stdlib server we can cancel (bound to an ephemeral port).
    monkeypatch.setitem(
        jobs_mod.JOB_SPECS,
        "_serve",
        jobs_mod.JobSpec(
            "http.server", positional=("port",), params={"port": jobs_mod.IntRange(20000, 65000)}
        ),
    )
    with TestClient(create_app(execution=True, root=tmp_path)) as client:
        job = client.post("/api/jobs", json={"job": "_serve", "params": {"port": 48231}}).json()
        job_id = job["id"]
        # wait until it is actually running
        for _ in range(50):
            if client.get(f"/api/jobs/{job_id}").json()["status"] == "running":
                break
            time.sleep(0.1)
        client.post(f"/api/jobs/{job_id}/cancel")
        assert _await_terminal(client, job_id) == "cancelled"


def _await_terminal(client: TestClient, job_id: str, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        if status in {"succeeded", "failed", "cancelled"}:
            return status
        time.sleep(0.1)
    return "timeout"
