"""Operator commands cannot invent readiness, activate on import, or stop strangers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from slm_training.autoresearch.runtime import operations_control as control
from slm_training.autoresearch.runtime import operations_doctor as readiness
from slm_training.autoresearch.runtime.activity_process import process_identity
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.harness_core.execution_release import (
    prepare_release,
    runtime_source_identity,
)
from slm_training.autoresearch.runtime.operations_status import loop_status, runtime_store


@pytest.mark.parametrize(
    "name", ["doctor", "status", "start", "resume", "stop", "verify-autonomy"]
)
def test_actual_cli_parser_help(name):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.autoresearch", name, "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0 and "usage:" in result.stdout


@pytest.mark.parametrize("loop_id", ["../outside", ".", "..", "/tmp/other", "a/b", ""])
def test_status_and_stop_reject_path_escape(tmp_path, loop_id):
    for operation in (loop_status, control.stop_supervisor):
        with pytest.raises(ValueError, match="invalid_loop_id"):
            operation(tmp_path, loop_id)
    assert not list(tmp_path.iterdir())


def test_stop_rejects_linked_runtime_root(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "root"
    (root / "loops").mkdir(parents=True)
    (root / "loops" / "owned").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="loop_state_symlink"):
        control.stop_supervisor(root, "owned")
    assert not list(external.iterdir())


def test_prepare_release_has_private_metadata_and_strict_source_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("source documentation\n")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("private metadata")
    (source / ".claude").mkdir()
    (source / ".claude" / "settings.local.json").write_text("private grant")
    release, execution, output = (
        tmp_path / name for name in ("release", "execution", "output")
    )
    result = prepare_release(source, release, execution, output)
    assert runtime_source_identity(execution) == result["source_digest"]
    assert not (execution / ".git").exists()
    assert not (execution / ".claude/settings.local.json").exists()
    (output / "report.json").write_text("{}")
    assert runtime_source_identity(execution) == result["source_digest"]
    (execution / "README.md").write_text("not an exempt source mutation")
    with pytest.raises(ValueError, match="execution_source_drift"):
        runtime_source_identity(execution)


def test_release_rejects_alias_into_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="must_be_disjoint"):
        prepare_release(
            source, alias / "release", tmp_path / "execution", tmp_path / "output"
        )


def test_doctor_stderr_redacted_and_directory_is_not_import_evidence(tmp_path, monkeypatch):
    for variable in ("AGENTV_RUNNER", "OPENUI_BRIDGE_CLI", "DESIGN_MD_BRIDGE_CLI"):
        monkeypatch.delenv(variable, raising=False)
    (tmp_path / "node_modules").mkdir()
    result = readiness._probe(
        (
            sys.executable,
            "-c",
            "import sys;sys.stderr.write('secret-canary');sys.exit(1)",
        ),
        tmp_path,
    )
    assert not result["ready"] and "secret-canary" not in json.dumps(result)
    js = readiness._javascript_probes(tmp_path)
    assert not js["agentv"]["ready"]
    assert not js["openui_bridge"]["ready"]


@pytest.mark.parametrize("minimum", [-1, True, 1.5])
def test_storage_refuses_invalid_budget(tmp_path, minimum):
    with pytest.raises(ValueError):
        readiness.storage_health(tmp_path, minimum_free_bytes=minimum)


def test_storage_pressure_requests_action_without_deletion(tmp_path):
    evidence = tmp_path / "checkpoint"
    evidence.write_bytes(b"preserve")
    result = readiness.storage_health(tmp_path, minimum_free_bytes=10**30)
    assert not result["ready"] and result["action"] == "storage_maintenance"
    assert not result["deletion_performed"] and evidence.read_bytes() == b"preserve"


def test_memory_pressure_backpressures_without_erasing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "_available_memory_bytes", lambda: 1000)
    result = readiness.storage_health(tmp_path, minimum_free_bytes=0)
    assert not result["ready"] and result["action"] == "memory_capacity"
    assert result["memory_pressure"] and not result["deletion_performed"]


def test_provider_absence_and_bad_configuration_are_scoped(tmp_path):
    assert readiness._agent_probe(None)["reason"] == "agent_grant_missing"
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert readiness._agent_probe(bad)["reason"] == "invalid_recovery_configuration"


def test_doctor_executes_bridge_protocol_and_rechecks_transitive_import(
    tmp_path, monkeypatch
):
    for variable in ("AGENTV_RUNNER", "OPENUI_BRIDGE_CLI", "DESIGN_MD_BRIDGE_CLI"):
        monkeypatch.delenv(variable, raising=False)
    sdk = tmp_path / "node_modules/@agentv/core/dist/index.js"
    sdk.parent.mkdir(parents=True)
    (sdk.parent / "package.json").write_text('{"type":"module"}')
    sdk.write_text(
        "import 'missing-transitive-dependency';export const evaluate=()=>{};"
    )
    for bridge in ("openui_bridge", "design_md_bridge"):
        path = tmp_path / "src/apps" / bridge / "cli.mjs"
        path.parent.mkdir(parents=True)
        path.write_text(
            "let s='';process.stdin.on('data',d=>s+=d);"
            "process.stdin.on('end',()=>{if(JSON.parse(s).op!=='ping')process.exit(2);"
            "console.log(JSON.stringify({ok:true,pong:true}));});"
        )
    before = readiness._javascript_probes(tmp_path)
    assert not before["agentv"]["ready"]
    assert before["openui_bridge"]["ready"] and before["design_md_bridge"]["ready"]
    sdk.write_text("export const evaluate=()=>{};")
    after = readiness._javascript_probes(tmp_path)
    assert after["agentv"]["ready"]


def test_no_controller_is_not_running_and_service_recipe_does_not_start(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        control.subprocess, "Popen", lambda *a, **k: pytest.fail("service activated")
    )
    recipe = json.loads(control.service_template(tmp_path / "config", tmp_path))
    assert recipe["installed"] is False and recipe["activated"] is False
    status = loop_status(tmp_path, "not-started")
    assert not status["active"] and not status["scientific_progress"]
    assert status["verified_paired_comparisons"] is None
    assert (
        control.stop_supervisor(tmp_path, "not-started")["reason"]
        == "no_owned_controller"
    )


def test_status_current_lease_not_remembered_pid(tmp_path):
    store = runtime_store(tmp_path, "fixture")
    with ActivityRuntime(store):
        state = loop_status(tmp_path, "fixture")
        assert state["active"] and not state["scientific_progress"]
    assert not loop_status(tmp_path, "fixture")["active"]


def test_stop_real_owned_session_leaves_unrelated_session_alive(tmp_path):
    code = (
        "import signal,sys,time;from pathlib import Path;"
        "from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime;"
        "from slm_training.autoresearch.storage import CampaignStore;"
        "from slm_training.harness_core.activity_contract import ActivitySpec,ResourceGrant;"
        "r=ActivityRuntime(CampaignStore('runtime',Path(sys.argv[1])/'loops'/'fixture'));"
        "r.__enter__();signal.signal(signal.SIGINT,lambda *a:r.cancel_event.set());"
        "s=ActivitySpec(activity_id='owned',family='fixture',kind='control',"
        "source_digest='a'*40,environment_digest='b'*64,input_digest='c'*64,"
        "output_namespace='runs/owned',grant=ResourceGrant(interrupt_seconds=15,"
        "kill_grace_seconds=.1,total_seconds=20,finalization_reserve_seconds=1));"
        "r.register(s);lease=r.claim_next(capabilities={'local_process'});"
        "work='import os,sys,time;from pathlib import Path;"
        "Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(25)';"
        "r.run(lease,[sys.executable,'-c',work,str(Path(sys.argv[1])/'child.pid')],"
        "cwd=Path(sys.argv[1]));"
        "r.cancel_all(reason='user stop');r.__exit__(None,None,None)"
    )
    owned = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)], start_new_session=True
    )
    stranger = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(25)"], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (tmp_path / "child.pid").exists():
            time.sleep(0.02)
        assert loop_status(tmp_path, "fixture")["active"]
        worker_pid = int((tmp_path / "child.pid").read_text())
        stranger_identity = process_identity(stranger.pid)
        result = control.stop_supervisor(tmp_path, "fixture")
        assert result["stopped"]
        assert owned.wait(timeout=2) == 0
        with pytest.raises(FileNotFoundError):
            process_identity(worker_pid)
        assert loop_status(tmp_path, "fixture")["activity_counts"] == {"cancelled": 1}
        assert (
            stranger.poll() is None
            and process_identity(stranger.pid) == stranger_identity
        )
    finally:
        for child in (owned, stranger):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)


def test_start_grant_required_before_launch(tmp_path):
    config = control.StartConfig(
        schema_version="local_supervisor_start/v1",
        execution=str(tmp_path),
        loop_id="fixture",
        train_version="fixture",
        steps=1,
        max_cycles=1,
        local_execution_authorized=False,
    )
    with pytest.raises(ValueError, match="local_execution_grant_missing"):
        control.start_supervisor(tmp_path, config)


def test_start_and_stop_use_real_process_with_disposable_controller_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "inherited-cache"))
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    runtime_source = Path(__file__).resolve().parents[2] / "src"
    # Only the controller fixture is substituted; launch, identity, locks and
    # stop execute the real imported owners. This is not a model-loop proof.
    (source / "scripts/run_autotrain_supervisor.py").write_text(
        "import argparse,os,signal,sys\nfrom pathlib import Path\n"
        "assert os.environ['GIT_CEILING_DIRECTORIES'] == str(Path.cwd().parent)\n"
        "assert 'PYTHONPYCACHEPREFIX' not in os.environ\n"
        "assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n"
        f"sys.path.insert(0, {str(runtime_source)!r})\n"
        "from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime\n"
        "from slm_training.autoresearch.storage import CampaignStore\n"
        "p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--loop-id')\n"
        "a,_=p.parse_known_args()\n"
        "with ActivityRuntime(CampaignStore('runtime',Path(a.root)/'loops'/a.loop_id)) as r:\n"
        " signal.signal(signal.SIGINT,lambda *a:r.cancel_event.set())\n"
        " r.cancel_event.wait(25)\n"
    )
    execution, root = tmp_path / "execution", tmp_path / "output"
    prepare_release(source, tmp_path / "release", execution, root)
    config = control.StartConfig(
        schema_version="local_supervisor_start/v1",
        execution=str(execution),
        loop_id="start-fixture",
        train_version="fixture",
        steps=1,
        max_cycles=1,
        local_execution_authorized=True,
    )
    try:
        first = control.start_supervisor(root, config)
        assert first["started"] and first["controller"]["alive"], first
        second = control.start_supervisor(root, config)
        assert second["already_owned"] and not second["started"]
        assert (
            second["controller"]["process_identity"]
            == first["controller"]["process_identity"]
        )
    finally:
        stopped = control.stop_supervisor(root, "start-fixture")
    assert stopped["stopped"] and not loop_status(root, "start-fixture")["active"]
    assert not (root / "operation-cache").exists()
    assert not (tmp_path / "inherited-cache").exists()


def test_completed_analysis_survives_delivery_failure_and_retries_idempotently(
    tmp_path, monkeypatch
):
    from slm_training.autoresearch.runtime import operations_analysis as analysis
    from slm_training.autoresearch.storage import CampaignStore

    request = analysis.SignComparisonRequest(
        schema_version="offline_sign_request/v1",
        campaign_id="signs",
        locked_design_digest="a" * 64,
        endpoint="candidate_better_probability_conditional_on_non_tie",
        unit_ids=tuple(f"unit-{n}" for n in range(6)),
        deltas=(1.0,) * 6,
        iid_units_declared=True,
        design_law="Artificial independent sign fixture only",
        ancestor_claim="no model ancestor; artificial arithmetic fixture",
    )
    path = tmp_path / "request.json"
    path.write_text(request.model_dump_json())
    with monkeypatch.context() as scoped:
        scoped.setattr(
            analysis,
            "_publish_docs",
            lambda *a: (_ for _ in ()).throw(OSError("disk full")),
        )
        with pytest.raises(OSError, match="disk full"):
            analysis.compare_sign_request(path, tmp_path / "runs", tmp_path / "docs")
    first = analysis.compare_sign_request(path, tmp_path / "runs", tmp_path / "docs")
    second = analysis.compare_sign_request(path, tmp_path / "runs", tmp_path / "docs")
    assert first == second and first["promotion_authority"] is False
    events = CampaignStore("signs", tmp_path / "runs").verify_event_chain()
    assert (
        sum(e["event_type"] == "offline_sign_analysis_completed" for e in events) == 1
    )
    artifacts = list(
        (tmp_path / "runs/signs/artifacts/offline_sign_result").glob("*.json")
    )
    assert len(artifacts) == 1
    artifacts[0].write_text("{}")
    with pytest.raises(ValueError, match="integrity_failure"):
        analysis.compare_sign_request(path, tmp_path / "runs", tmp_path / "docs")
