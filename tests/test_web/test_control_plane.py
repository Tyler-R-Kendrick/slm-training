"""Tests for the observability + control-plane web layer.

Covers the read-only observability API, cold-start fallback + provenance, the
pure-compute gate endpoint, the execution capability gate, the job allowlist
(the security boundary), and an end-to-end job run.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates
from slm_training.web import jobs as jobs_mod
from slm_training.web.app import create_app
from slm_training.web.observability import Readers

SMOKE_SUITE = {
    "smoke": {
        # Above the DEFAULT_MIN_SUITE_N evidence floor.
        "n": 32,
        "parse_rate": 0.9,
        "structural_similarity": 0.5,
        "placeholder_fidelity": 0.4,
        "reward_score": 0.5,
        # Measured (zero) fallback telemetry — certified_fallback fails closed
        # when unmeasured.
        "fallback_count": 0,
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


def test_spa_routes_and_retired_classic_redirect(ro_client: TestClient) -> None:
    """The SPA owns /playground and old classic bookmarks redirect to it."""
    root = ro_client.get("/")
    assert root.status_code == 200 and 'id="root"' in root.text
    # /playground is now a SPA route (the React playground).
    assert 'id="root"' in ro_client.get("/playground").text
    classic = ro_client.get("/playground/classic", follow_redirects=False)
    assert classic.status_code == 308
    assert classic.headers["location"] == "/playground"
    # Client-side deep routes (incl. /runs/<id>) fall through to the SPA shell.
    assert 'id="root"' in ro_client.get("/checkpoints").text
    assert 'id="root"' in ro_client.get("/runs/qx_e70_stability").text


def test_readers_cold_start_fallback(tmp_path) -> None:
    """An empty repo root must never raise; it reports committed provenance."""
    from slm_training.dsl.schema import ExampleRecord, write_jsonl

    write_jsonl(
        tmp_path / "src" / "slm_training" / "resources" / "train_seeds.jsonl",
        [
            ExampleRecord(
                id="example-1",
                prompt="Show the training data",
                openui='root = TextContent(":copy.value")',
                source="fixture",
            )
        ],
    )
    readers = Readers(tmp_path)
    assert readers.scoreboard("quality")["results"] == []
    train = readers.train_data()
    assert train["provenance"] == "committed"
    assert train["version"] == "examples"
    assert train["versions"] == ["examples"]
    assert train["record_count"] == 1
    assert readers.train_records("examples")["records"][0]["id"] == "example-1"
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
    assert readers.train_data()["versions"] == ["examples", "v1"]
    page = readers.train_records("v1", offset=2, limit=2)
    assert page["count"] == 6
    assert [row["id"] for row in page["records"]] == ["row-2", "row-3"]
    assert page["sources"] == ["layout", "template"]
    filtered = readers.train_records("v1", source="template", query="Prompt 3")
    assert filtered["count"] == 1
    assert filtered["records"][0]["id"] == "row-3"


def test_preference_data_lists_committed_event_corpora(tmp_path) -> None:
    directory = (
        tmp_path
        / "src/slm_training/resources/data/preference/events-v1"
    )
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "decision_event_corpus",
                "dataset_id": "events-v1",
                "record_count": 12,
                "splits": {"train": 9, "held_out": 3},
                "evidence_kinds": {"constraint_shadow": 12},
                "set_valued_events": 0,
                "content_fingerprint": "abcdef1234567890",
            }
        )
    )
    data = Readers(tmp_path).preference_data()
    assert data["provenance"] == "committed"
    assert data["rows"] == [
        {
            "dataset_id": "events-v1",
            "kind": "exact-state decisions",
            "records": 12,
            "train": 9,
            "held_out": 3,
            "evidence": "constraint_shadow:12",
            "usage": "decoder evidence only",
            "fingerprint": "abcdef123456",
        }
    ]


def test_preference_data_describes_counterfactual_corpora_by_capability(
    tmp_path,
) -> None:
    directory = tmp_path / "src/slm_training/resources/data/preference/events-v1"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "decision_event_corpus",
                "dataset_id": "events-v1",
                "record_count": 12,
                "splits": {"train": 9, "held_out": 3},
                "evidence_kinds": {"counterfactual": 12},
                "content_fingerprint": "abcdef1234567890",
            }
        )
    )

    assert Readers(tmp_path).preference_data()["rows"][0]["usage"] == (
        "semantic preference training"
    )


def test_committed_train_version_is_default_and_browsable(tmp_path) -> None:
    from slm_training.dsl.schema import ExampleRecord, write_jsonl

    vdir = (
        tmp_path
        / "src"
        / "slm_training"
        / "resources"
        / "data"
        / "train"
        / "remediated_roots_judged"
    )
    write_jsonl(
        vdir / "records.jsonl",
        [
            ExampleRecord(
                id="judged-1",
                prompt="A judged prompt/output pair",
                openui='root = TextContent(":copy.value")',
                source="judged",
            )
        ],
    )
    (vdir / "stats.json").write_text('{"record_count": 1}\n', encoding="utf-8")
    (vdir / "manifest.json").write_text("{}\n", encoding="utf-8")

    readers = Readers(tmp_path)
    data = readers.train_data()
    assert data["provenance"] == "committed"
    assert data["version"] == "remediated_roots_judged"
    assert data["record_count"] == 1
    assert readers.train_records(data["version"])["records"][0]["id"] == "judged-1"


def test_run_detail_merges_scoreboard(ro_client: TestClient) -> None:
    board = ro_client.get("/api/scoreboards/quality").json()["results"]
    # The committed matrix can contain metadata-only rows before the actual
    # suite scoreboards; choose a row whose suites can produce gate output.
    row = next(row for row in board if "suites" in row)
    run_id = row.get("run_id") or row.get("id")
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


def test_research_evidence_and_autoresearch_run_are_current(tmp_path) -> None:
    design = tmp_path / "docs" / "design"
    design.mkdir(parents=True)
    run_dir = (
        tmp_path / "outputs" / "autoresearch" / "e9-current" / "runs" / "e9-run"
    )
    run_dir.mkdir(parents=True)
    suites = {
        "smoke": {
            "n": 3,
            "parse_rate": 1.0,
            "meaningful_program_rate": 0.25,
            "structural_similarity": 0.4,
            "placeholder_fidelity": 0.5,
            "reward_score": 0.6,
        }
    }
    (design / "iter-e9-current-20260716.json").write_text(
        json.dumps(
            {
                "campaign": "E9 current experiment",
                "date_utc": "2026-07-16",
                "run_id": "e9-run",
                "train_result": {"trace_id": "a" * 32},
                "suites": suites,
                "ship_gates": {"pass": False},
                "agentv": {"total": 5, "passed": 1},
                "scoreboard": "outputs/autoresearch/e9-current/runs/e9-run/scoreboard.json",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train_summary.json").write_text(
        json.dumps({"run_id": "e9-run", "steps": 12, "last_loss": 1.25}),
        encoding="utf-8",
    )
    (run_dir / "train_telemetry.json").write_text(
        json.dumps({"spans": {"forward": {"pct": 75.0}}}), encoding="utf-8"
    )
    (run_dir / "trace.json").write_text(
        json.dumps({"trace_id": "a" * 32}), encoding="utf-8"
    )
    (run_dir / "gates.json").write_text(
        json.dumps({"pass": False, "failures": ["smoke:meaningful_program_rate"]}),
        encoding="utf-8",
    )

    readers = Readers(tmp_path)
    research = readers.scoreboard("research")
    assert research["results"][0]["run_id"] == "e9-run"
    assert research["results"][0]["agentv"] == {"total": 5, "passed": 1}
    assert any(row["run_id"] == "e9-run" for row in readers.runs()["runs"])
    detail = readers.run("e9-run")
    assert detail["provenance"] == "live"
    assert detail["train_summary"]["steps"] == 12
    assert detail["telemetry"]["spans"]["forward"]["pct"] == 75.0
    assert detail["trace"]["trace_id"] == "a" * 32
    assert detail["scoreboard"]["suites"] == suites
    # Headline comparisons use meaningful output, not syntax-only parse success,
    # keyed by the ship-gate policy's lever names.
    comparisons, _ = readers._performance_rows([])
    current = next(row for row in comparisons if row["run_id"] == "e9-run")
    assert current["metrics"]["meaningful_program_rate"] == 0.25


def test_research_evidence_accepts_nested_train_and_evaluation(tmp_path) -> None:
    design = tmp_path / "docs" / "design"
    design.mkdir(parents=True)
    suites = {"smoke": {"n": 3, "meaningful_program_rate": 1 / 3}}
    (design / "iter-e230-diverse-roots-20260716.json").write_text(
        json.dumps(
            {
                "campaign": "E230 diverse judged generation roots",
                "date": "2026-07-16",
                "train": {
                    "run_id": "e230-diverse-roots-32step",
                    "path": "outputs/autoresearch/e230/runs/e230-diverse-roots-32step",
                    "trace_id": "b" * 32,
                },
                "evaluation": {
                    "suites": suites,
                    "failed_gates": 4,
                    "agentv": {"total": 5, "passed": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    result = Readers(tmp_path).scoreboard("research")["results"][0]
    assert result["run_id"] == "e230-diverse-roots-32step"
    assert result["pass"] is False
    assert result["suites"] == suites
    assert result["agentv"] == {"total": 5, "passed": 1}
    assert result["trace_id"] == "b" * 32
    assert result["run_dir"].endswith("e230-diverse-roots-32step")


def test_rl_traces_are_paginated_and_malformed_rows_are_skipped(tmp_path) -> None:
    path = tmp_path / "outputs" / "runs" / "molt-smoke" / "rl_traces.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"run_id": "molt-smoke", "engine": "molt", "rollout_id": f"r-{i}"}
        for i in range(3)
    ]
    path.write_text(
        json.dumps(rows[0])
        + "\nnot-json\n"
        + json.dumps({"run_id": "other"})
        + "\n"
        + json.dumps(rows[1])
        + "\n"
        + json.dumps(rows[2])
        + "\n"
    )
    with TestClient(create_app(execution=False, root=tmp_path)) as client:
        response = client.get("/api/runs/molt-smoke/rl-traces?offset=1&limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert payload["count"] == 1
    assert payload["invalid_rows"] == 2
    assert payload["traces"][0]["rollout_id"] == "r-1"

    missing = Readers(tmp_path).rl_traces("../escape")
    assert missing["provenance"] == "missing"
    assert missing["traces"] == []


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
    assert ["--derive-from", "outputs/data/train/v1/records.jsonl"] == argv[
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
