"""Real Node imports/execution with a separately installed SDK fixture; no downloads."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from slm_training.autoresearch.runtime import operations_doctor as doctor
from slm_training.evals import agentv


@pytest.fixture
def installed(tmp_path, monkeypatch):
    source, modules = tmp_path / "source", tmp_path / "installed-modules"
    runner = source / "scripts/run_agentv_eval.mjs"
    runner.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[2] / "scripts/run_agentv_eval.mjs", runner)
    sdk = modules / "@agentv/core/dist/index.js"
    sdk.parent.mkdir(parents=True)
    (sdk.parent.parent / "package.json").write_text('{"type":"module"}')
    sdk.write_text("export async function evaluate(options) {"
                   "if(options.threshold!==1||options.workers!==1||options.cache!==false)throw Error('policy');"
                   "return {summary:{executionErrors:0,fixtureExecuted:true},artifacts:{}};}")
    packages = {"node_modules/@agentv/core": {"version": "fixture"}}
    for path in (source / "package-lock.json", modules / ".package-lock.json"):
        path.write_text(json.dumps({"packages": packages}))
    monkeypatch.delenv("AGENTV_RUNNER", raising=False)
    monkeypatch.setenv("AGENTV_NODE_MODULES", str(modules))
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.setattr(agentv, "checkout_roots", lambda _: (source,))
    monkeypatch.setattr(agentv, "_bootstrap_agentv_sdk", lambda *_: pytest.fail("download attempted"))
    return source, runner, modules, sdk


@pytest.mark.parametrize("override", [False, True])
def test_python_and_js_execute_explicit_modules_with_runner_override(installed, tmp_path, monkeypatch, override):
    source, runner, modules, _ = installed
    if override:
        monkeypatch.setenv("AGENTV_RUNNER", str(runner))
    selected, root = agentv._agentv_runtime(source)
    assert selected == runner
    result = agentv.publish_agentv_evaluation(tmp_path / "run", name="module-wiring",
        claim="fixture_wiring_only", cases=[{"id":"case", "criteria":"fixture", "pass":True}])
    assert result["summary"]["fixtureExecuted"] is True
    assert result["criteria"]["pass"] is False  # Fixture SDK is never scientific evidence.
    assert root == (modules.parent if override else source)


def test_explicit_modules_must_match_install_marker_and_never_bootstrap(installed):
    source, _, modules, _ = installed
    (modules / ".package-lock.json").write_text('{"packages":{}}')
    with pytest.raises(RuntimeError, match="Explicit AgentV modules"):
        agentv._agentv_runtime(source)


def test_doctor_imports_explicit_modules_and_rejects_broken_transitive_dependency(installed, monkeypatch):
    source, runner, _, sdk = installed
    monkeypatch.setenv("AGENTV_RUNNER", str(runner))
    assert doctor._javascript_probes(source)["agentv"]["ready"]
    sdk.write_text("import 'missing-transitive-package';export function evaluate(){}")
    assert not doctor._javascript_probes(source)["agentv"]["ready"]


def test_js_sdk_root_and_bootstrap_symlink_remain_compatible(installed, tmp_path, monkeypatch):
    source, runner, modules, _ = installed
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "node_modules").symlink_to(modules, target_is_directory=True)
    monkeypatch.delenv("AGENTV_NODE_MODULES")
    result = subprocess.run(["node", "--preserve-symlinks", str(runner), "--sdk-root", str(staged),
        "--spec", str(source / "fixture.jsonl"), "--output-dir", str(tmp_path / "out"),
        "--experiment", "fixture"], env={**os.environ, "NODE_OPTIONS":""},
        capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(result.stdout)["summary"]["fixtureExecuted"]
