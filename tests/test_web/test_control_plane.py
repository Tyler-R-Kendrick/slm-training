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


def test_spa_routes_and_classic_playground_fallback(ro_client: TestClient) -> None:
    """The SPA serves "/" and /playground; the classic page is at /playground/classic."""
    root = ro_client.get("/")
    assert root.status_code == 200 and 'id="root"' in root.text
    # /playground is now a SPA route (the React playground).
    assert 'id="root"' in ro_client.get("/playground").text
    classic = ro_client.get("/playground/classic")
    assert classic.status_code == 200 and "TwoTower" in classic.text
    # Client-side deep routes (incl. /runs/<id>) fall through to the SPA shell.
    assert 'id="root"' in ro_client.get("/checkpoints").text
    assert 'id="root"' in ro_client.get("/runs/qx_e70_stability").text


def test_readers_cold_start_fallback(tmp_path) -> None:
    """An empty repo root must never raise; it reports committed provenance."""
    readers = Readers(tmp_path)
    assert readers.scoreboard("quality")["results"] == []
    assert readers.train_data()["provenance"] == "committed"
    assert readers.test_data()["provenance"] == "committed"
    assert readers.runs()["provenance"] == "committed"


def test_train_records_supports_browsing_filters_and_pagination(tmp_path) -> None:
    from slm_training.dsl.schema import ExampleRecord, write_jsonl

    path = tmp_path / "outputs" / "train_data" / "v1" / "records.jsonl"
    write_jsonl(
        path,
        [
            ExampleRecord(
                id=f"row-{i}",
                prompt=f"Prompt {i}",
                openui='root = TextContent(":copy.value")',
                source="template" if i % 2 else "layout",
            )
            for i in range(6)
        ],
    )
    readers = Readers(tmp_path)
    page = readers.train_records("v1", offset=2, limit=2)
    assert page["count"] == 6
    assert [row["id"] for row in page["records"]] == ["row-2", "row-3"]
    assert page["sources"] == ["layout", "template"]
    filtered = readers.train_records("v1", source="template", query="Prompt 3")
    assert filtered["count"] == 1
    assert filtered["records"][0]["id"] == "row-3"


def test_run_detail_merges_scoreboard(ro_client: TestClient) -> None:
    board = ro_client.get("/api/scoreboards/quality").json()["results"]
    run_id = board[0].get("run_id") or board[0].get("id")
    detail = ro_client.get(f"/api/runs/{run_id}").json()
    assert detail["scoreboard"] is not None
    assert detail["scoreboard"]["matrix"] == "quality"
    # gates are derived from the scoreboard suites even with an empty outputs/.
    assert detail["gates"] is not None and "pass" in detail["gates"]


def test_run_detail_missing_is_graceful(ro_client: TestClient) -> None:
    detail = ro_client.get("/api/runs/nope_xyz").json()
    assert detail["provenance"] == "committed"
    assert detail["scoreboard"] is None
    assert detail["gates"] is None


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


# --- remote dispatch monitoring --------------------------------------------
def test_dispatches_endpoint_shape(ro_client: TestClient) -> None:
    payload = ro_client.get("/api/dispatches").json()
    assert set(payload) >= {"jobs", "remotes", "bucket_url"}
    assert payload["bucket_url"].startswith("https://huggingface.co/")


def test_remote_url_extraction() -> None:
    from slm_training.web.observability import _first_remote_url

    assert _first_remote_url("submitted https://huggingface.co/jobs/x9 ok") == (
        "https://huggingface.co/jobs/x9"
    )
    assert _first_remote_url("trackio https://tk-openui.hf.space/.") == (
        "https://tk-openui.hf.space/"
    )
    assert _first_remote_url("no url here") is None


def test_remote_train_allowlisted_and_safe(tmp_path) -> None:
    assert jobs_mod.JOB_SPECS["remote_train"].kind == "dispatch"
    with TestClient(create_app(execution=True, root=tmp_path)) as client:
        assert (
            client.post(
                "/api/jobs",
                json={"job": "remote_train", "params": {"host": "a;rm -rf /", "run_id": "r"}},
            ).status_code
            == 422
        )


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


def test_train_data_job_renders_existing_derivative_controls() -> None:
    argv = jobs_mod.JOB_SPECS["build_train_data"].render_argv(
        {
            "source": "existing",
            "base_version": "v1",
            "version": "v1-derived",
            "synthesizer": "layout",
            "namespace_augment": True,
            "edit_derivatives": False,
            "repairs_per_program": 0,
        }
    )
    assert ["--derive-from", "outputs/train_data/v1/records.jsonl"] == argv[
        argv.index("--derive-from") : argv.index("--derive-from") + 2
    ]
    assert "--namespace-augment" in argv
    assert "--no-edit-derivatives" in argv


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
