"""Real launch/unit boundaries; connector fixtures are not live delivery proof."""

import hashlib
import json
import shutil
import subprocess
import sys
import time

import pytest

from slm_training.autoresearch.runtime import operations_control as control
from slm_training.autoresearch.runtime import operations_service as service
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.runtime.operations_status import (
    loop_status,
    runtime_store,
)
from slm_training.harness_core.activity_contract import ActivitySpec, contract_digest
from slm_training.harness_core.execution_release import prepare_release


RUNTIME_ENVIRONMENT = {
    "PATH": "/opt/runtime/bin:/usr/bin:/bin",
    "OPENUI_BRIDGE_CLI": "/opt/runtime/openui/cli.mjs",
    "DESIGN_MD_BRIDGE_CLI": "/opt/runtime/design/cli.mjs",
    "GRAPHQL_BRIDGE_CLI": "/opt/runtime/graphql/cli.mjs",
    "AGENTV_RUNNER": "/opt/runtime/agentv/cli.mjs",
    "AGENTV_NODE_MODULES": "/opt/runtime/agentv/node_modules",
    "SLM_DATA_ROOT": "/opt/data space % $",
}


def start_config(execution):
    return control.StartConfig(
        schema_version="local_supervisor_start/v1",
        execution=str(execution),
        loop_id="managed",
        train_version="fixture",
        steps=1,
        max_cycles=1,
        local_execution_authorized=True,
        runtime_environment=RUNTIME_ENVIRONMENT,
    )


def test_generated_units_verify_and_install_idempotently(tmp_path):
    if not shutil.which("systemd-analyze"):
        pytest.skip("systemd-analyze unavailable")
    execution = tmp_path / "execution space % $"
    execution.mkdir()
    settings = start_config(execution)
    config = tmp_path / "config space.json"
    config.write_text(settings.model_dump_json())
    result = service.install_service(config, tmp_path, settings, tmp_path / "units")
    assert result["installed"] and not result["activated"]
    assert (
        service.install_service(config, tmp_path, settings, tmp_path / "units")
        == result
    )
    unit = result["units"][result["service"]]
    for key, value in RUNTIME_ENVIRONMENT.items():
        assert "Environment=" + service._quote(key + "=" + value) in unit
    assert "--foreground" in unit and "KillMode=mixed" in unit
    assert "Restart=on-failure" in unit and "StartLimitBurst=5" in unit
    timer = result["units"][result["timer"]]
    assert "OnCalendar=*-*-* 00/3:00:00 UTC" in timer and "Persistent=true" in timer
    (tmp_path / "units" / result["service"]).write_text("unrelated unit")
    with pytest.raises(ValueError, match="existing_service_unit_conflict"):
        service.install_service(config, tmp_path, settings, tmp_path / "units")


def test_unit_command_injection_rejected(tmp_path):
    bad = tmp_path / "execution\nExecStart=bad"
    bad.mkdir()
    with pytest.raises(ValueError, match="control_character"):
        service.service_units(tmp_path / "config", tmp_path, start_config(bad))


def test_foreground_exec_preserves_real_pid_and_exit_code(tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    result = tmp_path / "pid.json"
    (source / "scripts/run_autotrain_supervisor.py").write_text(
        "import os,sys,json\nfrom pathlib import Path\n"
        f"Path({str(result)!r}).write_text(json.dumps({{'pid':os.getpid(),'argv':sys.argv,'env':dict(os.environ)}}))\n"
        "raise SystemExit(7)\n"
    )
    execution, root = tmp_path / "execution", tmp_path / "output"
    prepare_release(source, tmp_path / "release", execution, root)
    config = tmp_path / "config.json"
    config.write_text(start_config(execution).model_dump_json())
    command = [
        sys.executable,
        "-m",
        "scripts.autoresearch",
        "--root",
        str(root),
        "start",
        "--config",
        str(config),
        "--foreground",
    ]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as child:
        stdout, stderr = child.communicate(timeout=20)
        assert child.returncode == 7, (stdout, stderr)
        payload = json.loads(result.read_text())
        assert payload["pid"] == child.pid
        assert {
            key: payload["env"][key] for key in RUNTIME_ENVIRONMENT
        } == RUNTIME_ENVIRONMENT


def test_monitor_records_actual_checks_without_starting_work(tmp_path, monkeypatch):
    monkeypatch.setattr(
        service,
        "timer_status",
        lambda _: {"ready": False, "reason": "user_systemd_unavailable"},
    )
    assert loop_status(tmp_path, "managed")["last_monitor_check"] is None
    first = service.monitor_check(tmp_path, "managed")
    assert first["assessment"] == "stopped"
    assert loop_status(tmp_path, "managed")["last_monitor_check"] == first
    with ActivityRuntime(runtime_store(tmp_path, "managed")):
        second = service.monitor_check(tmp_path, "managed")
        assert second["assessment"] == "stalled"
        assert second["checked_at"] >= first["checked_at"]
    assert not loop_status(tmp_path, "managed")["scientific_progress"]


def test_timer_failure_does_not_invent_next_firing(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(service.subprocess, "run", missing)
    assert service.timer_status("managed") == {
        "ready": False,
        "reason": "user_systemd_unavailable",
    }


def connector_fixture(tmp_path):
    executable = tmp_path / "connector-fixture"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """import argparse,json
from pathlib import Path
import hashlib
def contract_digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
r=json.loads(Path(a.request).read_text())
out={'schema_version':'connector_delivery_receipt/v1','request_digest':contract_digest(r),
     'provider':'github_connector','repository':r['repository'],'source_digest':r['source_digest'],
     'merged':True,'checks_verified':True,'reviews_resolved':True,'content_verified':True,
     'merge_sha':'c'*40,'verified_head_sha':'d'*40}
Path(a.output).write_text(json.dumps(out))
"""
    )
    executable.chmod(0o700)
    return control.DeliveryHost(
        command=(str(executable),),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        repository="fixture/repo",
        base_ref="f" * 40,
        source_digest="a" * 40,
        expires_at=time.time() + 100,
        authorized=True,
    )


def test_writer_without_independent_reader_stays_waiting(tmp_path):
    host = connector_fixture(tmp_path)
    wait = {"kind": "document", "artifact_sha256": "e" * 64}
    with ActivityRuntime(runtime_store(tmp_path, "managed")) as runtime:
        runtime.register(
            ActivitySpec(
                activity_id="delivery",
                family="managed",
                kind="delivery",
                source_digest="a" * 40,
                environment_digest="b" * 64,
                input_digest=contract_digest(wait),
                output_namespace="delivery",
                capabilities=("authorized_github_connector_delivery",),
            )
        )
        result = control.consume_delivery(runtime, "delivery", wait, host)
        assert result["state"] == "waiting_capability"
        assert (
            result["reason"]
            == "configured_writer_reader_and_authenticated_local_gate_required"
        )
        assert runtime.snapshot()["delivery"].attempts == 0


def test_connector_cannot_self_assert_unbound_success(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"merged": True}))
    with pytest.raises(ValueError, match="not_verified"):
        control._delivery_receipt(receipt, {})


def test_workspace_inventory_never_invokes_writer(tmp_path, monkeypatch):
    from slm_training.autoresearch.runtime import operations_delivery

    def forbidden(*args):
        pytest.fail("workspace inventory reached delivery host")

    monkeypatch.setattr(operations_delivery, "_delivery_capable", forbidden)
    wait = {"kind": "workspace", "paths": ["unrelated-work.py"]}
    with ActivityRuntime(runtime_store(tmp_path, "managed")) as runtime:
        runtime.register(ActivitySpec(
            activity_id="delivery", family="managed", kind="delivery",
            source_digest="a" * 40, environment_digest="b" * 64,
            input_digest=contract_digest(wait), output_namespace="delivery",
            capabilities=("authorized_github_connector_delivery",),
        ))
        result = control.consume_delivery(runtime, "delivery", wait, object())
        assert result["state"] == "waiting_capability"
        assert result["reason"] == "immutable_workspace_successor_and_reconciliation_required"
        assert runtime.snapshot()["delivery"].attempts == 0
        assert runtime.store.verify_event_chain()[-1]["detail"] == {
            "predicate": result["reason"], "input_digest": contract_digest(wait),
            "writer_started": False,
        }


def test_delivery_requires_gate_for_exact_materialized_documents(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.test_autoresearch.test_delivery_reconciliation import document_wait
    from slm_training.autoresearch.runtime import operations_delivery

    _, _, wait, files = document_wait(tmp_path)
    host = SimpleNamespace(verification_plan={})
    monkeypatch.setattr(operations_delivery, "_delivery_capable", lambda *_: True)
    with ActivityRuntime(runtime_store(tmp_path, "managed")) as runtime:
        runtime.register(ActivitySpec(
            activity_id="delivery", family="managed", kind="delivery",
            source_digest="a" * 40, environment_digest="b" * 64,
            input_digest=contract_digest(wait), output_namespace="delivery",
            capabilities=("authorized_github_connector_delivery",),
        ))
        with pytest.raises(ValueError, match="delivery_document_successor_gate_required"):
            control.consume_delivery(runtime, "delivery", wait, host)
        assert runtime.snapshot()["delivery"].attempts == 0
        host.verification_plan["delivery_documents_sha256"] = {
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()
        }
        assert operations_delivery._document_capable(runtime, wait, host, runtime.snapshot()["delivery"])


def test_comparison_requires_complete_untampered_reference_files(tmp_path):
    from slm_training.autoresearch.runtime.operations_status import _validate_comparison
    from slm_training.autoresearch.storage import CampaignStore
    from scripts.autotrain_metrics import read_paired_nll

    store = CampaignStore("fixture", tmp_path)
    selected = [f"case-{n}" for n in range(6)]
    result = {
        "schema": "supervised_measurement_result/v1",
        "diagnostic_complete": True,
        "confirmation_complete": False,
        "promotion_allowed": False,
        "ship_eligible": False,
        "operation": {"campaign_id": "fixture"},
        "selection": {"selected_record_ids": selected},
        "decision": {
            "control_metrics": {"parse_rate": 1.0},
            "candidate_metrics": {"parse_rate": 1.0},
        },
        "arms": {},
    }
    for name in ("control", "candidate"):
        directory = store.root / "runs" / name
        directory.mkdir(parents=True)
        nll = {
            "schema": "eval_nll_records/v1",
            "definition_hash": "a" * 64,
            "records": {case: 1.0 for case in selected},
            "selection": {"selected_record_ids": selected},
        }
        (directory / "eval_nll_records.json").write_text(json.dumps(nll))
        hashes = {}
        for filename in ("scoreboard.json", "loss_suites.json", "agentv.jsonl"):
            path = directory / filename
            path.write_text("{}")
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["arms"][name] = {
            "scoreboard_sha256": hashes["scoreboard.json"],
            "loss_report_sha256": hashes["loss_suites.json"],
            "agentv_artifacts": {
                str(directory / "agentv.jsonl"): hashes["agentv.jsonl"]
            },
        }
    _, result["paired_counts"], _ = read_paired_nll(
        store.root / "runs/control", store.root / "runs/candidate"
    )
    _validate_comparison(store, result)
    (store.root / "runs/control/scoreboard.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="referenced_artifact_changed"):
        _validate_comparison(store, result)


def test_monitor_records_invalid_scientific_evidence(tmp_path, monkeypatch):
    def invalid(*args):
        raise ValueError("changed artifact")

    monkeypatch.setattr(service, "loop_status", invalid)
    monkeypatch.setattr(service, "timer_status", lambda _: {"ready": False})
    result = service.monitor_check(tmp_path, "managed")
    assert result["assessment"] == "invalid_evidence"
    assert (
        runtime_store(tmp_path, "managed").verify_event_chain()[-1]["detail"] == result
    )


def test_launcher_exact_arguments_accepted_by_current_supervisor(tmp_path, monkeypatch):
    from pathlib import Path

    execution = tmp_path / "execution"
    execution.mkdir()
    recovery, delivery = tmp_path / "recovery.json", tmp_path / "delivery.json"
    recovery.write_text("{}")
    delivery.write_text("{}")
    config = start_config(execution).model_copy(
        update={"recovery_config": str(recovery), "delivery_config": str(delivery)}
    )
    captured = {}

    def exec_capture(executable, argv, env):
        captured["argv"] = argv
        raise RuntimeError("captured_exec")

    monkeypatch.setattr(control.os, "execve", exec_capture)
    original = Path.cwd()
    try:
        with pytest.raises(RuntimeError, match="captured_exec"):
            control._launch(
                tmp_path, config, execution, "a" * 40, None, foreground=True
            )
    finally:
        control.os.chdir(original)
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;from scripts.run_autotrain_supervisor import _build_parser;"
            "a=_build_parser().parse_args(sys.argv[1:]);"
            "assert a.delivery_config and a.repair_config and a.max_cycles==1",
            *captured["argv"][3:],
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert check.returncode == 0, check.stderr


def test_runtime_environment_rejects_non_allowlisted_credentials(tmp_path):
    payload = start_config(tmp_path).model_dump()
    payload["runtime_environment"] = {"HF_TOKEN": "never-load-secret"}
    with pytest.raises(ValueError):
        control.StartConfig.model_validate(payload)
