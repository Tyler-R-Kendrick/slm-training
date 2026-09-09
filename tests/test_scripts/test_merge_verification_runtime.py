"""Zero-copy granted JS dependencies in the actual rootless workload boundary."""

import json
import os
import sys
from pathlib import Path

import pytest

from scripts.merge_verification_runtime import javascript_grants, runtime_argv
from scripts.merge_verification_isolation import run_workload
from slm_training.autoresearch.heal.isolation import (
    IsolationSpec,
    IsolationUnavailable,
    run_isolated,
)


def test_missing_js_roots_is_precise_capability_wait(tmp_path):
    with pytest.raises(
        IsolationUnavailable, match="explicit_js_runtime_grants_missing"
    ):
        javascript_grants(tmp_path, (), required=True)


def collect_fixture(tmp_path, source):
    root = tmp_path / "candidate"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test_case.py").write_text(source)
    return run_workload(
        root, ["tests"], collect_only=True, seconds=15, directory=tmp_path
    )


def test_collection_terminal_is_compact_but_protocol_keeps_all_nodes(tmp_path):
    record = collect_fixture(
        tmp_path,
        "import pytest\n@pytest.mark.parametrize('value', range(1200))\n"
        "def test_case(value): pass\n",
    )
    assert record["status"] == "ok", record
    assert len(record["nodes"]) == len(set(record["nodes"])) == 1200
    assert len(record["output_tail"]) < 200
    assert json.loads(record["output_tail"])["collected"] == 1200


def test_compact_collection_retains_original_import_failure(tmp_path):
    record = collect_fixture(tmp_path, "raise RuntimeError('collection-canary')\n")
    assert record["status"] != "ok"
    assert "collection-canary" in record["output_tail"]


def sdk_fixture(tmp_path, monkeypatch):
    source = tmp_path / "candidate"
    runner = source / "scripts/run_agentv_eval.mjs"
    runner.parent.mkdir(parents=True)
    runner.write_text("export const fixture = true;\n")
    monkeypatch.setenv("AGENTV_RUNNER", str(runner))
    modules = tmp_path / "install/node_modules"
    core = modules / "@agentv/core"
    (core / "dist").mkdir(parents=True)
    (core / "package.json").write_text('{"type":"module"}')
    (core / "dist/index.js").write_text("export { value } from 'fixture-transitive';\n")
    transitive = modules / "fixture-transitive"
    transitive.mkdir()
    (transitive / "package.json").write_text('{"type":"module","exports":"./index.js"}')
    (transitive / "index.js").write_text("export const value = 42;\n")
    return source, modules


def node_root():
    executable = Path(
        os.environ.get(
            "SLM_TEST_NODE", "/home/codex/.nvm/versions/node/v22.23.1/bin/node"
        )
    )
    assert executable.is_file(), (
        "This real-process fixture requires the declared existing Node runtime"
    )
    return executable.parent.parent


def test_complete_transitive_tree_is_mounted_not_copied(tmp_path, monkeypatch):
    source, modules = sdk_fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "work"
    workspace.mkdir()
    runtimes = (node_root(), modules)
    program = """import {dirname} from 'node:path'; import {pathToFileURL} from 'node:url';
import {existsSync,lstatSync} from 'node:fs';
const root=dirname(dirname(process.env.AGENTV_RUNNER));
const m=await import(pathToFileURL(root+'/node_modules/@agentv/core/dist/index.js'));
if(m.value!==42 || !lstatSync(root+'/node_modules').isSymbolicLink()) process.exit(2);
if(existsSync('/home/codex/repos/slm-training/.git')) process.exit(3);
console.log(JSON.stringify({transitive_value:m.value}));"""
    argv = runtime_argv(
        source,
        runtimes,
        ["/runtime/0/bin/node", "--input-type=module", "-e", program],
        workspace=workspace,
    )
    staged = [p for p in workspace.rglob("*") if p.is_file()]
    assert [p.name for p in staged] == ["run_agentv_eval.mjs"]
    result = run_isolated(
        IsolationSpec(workspace, runtime_roots=runtimes, timeout_seconds=20), argv
    )
    assert result.returncode == 0 and not result.timed_out, result.stderr
    assert json.loads(result.stdout) == {"transitive_value": 42}
    assert not (workspace / "node_modules").exists()


def test_missing_transitive_dependency_is_not_ready(tmp_path, monkeypatch):
    source, modules = sdk_fixture(tmp_path, monkeypatch)
    (modules / "fixture-transitive/index.js").unlink()
    workspace = tmp_path / "work"
    workspace.mkdir()
    runtimes = (node_root(), modules)
    program = """import {dirname} from 'node:path'; import {pathToFileURL} from 'node:url';
const root=dirname(dirname(process.env.AGENTV_RUNNER));
await import(pathToFileURL(root+'/node_modules/@agentv/core/dist/index.js'));"""
    argv = runtime_argv(
        source,
        runtimes,
        ["/runtime/0/bin/node", "--input-type=module", "-e", program],
        workspace=workspace,
    )
    result = run_isolated(
        IsolationSpec(workspace, runtime_roots=runtimes, timeout_seconds=20), argv
    )
    assert result.returncode != 0
    assert "ERR_MODULE_NOT_FOUND" in result.stderr


def test_real_installed_sdk_and_both_bridge_protocols(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2]
    primary = Path("/home/codex/repos/slm-training")
    openui = primary / "src/apps/openui_bridge"
    design = primary / "src/apps/design_md_bridge"
    monkeypatch.setenv("OPENUI_BRIDGE_CLI", str(openui / "cli.mjs"))
    monkeypatch.setenv("DESIGN_MD_BRIDGE_CLI", str(design / "cli.mjs"))
    monkeypatch.setenv("AGENTV_RUNNER", str(primary / "scripts/run_agentv_eval.mjs"))
    runtimes = (
        Path(sys.prefix),
        node_root(),
        openui,
        design,
        primary / "node_modules",
        source / "src",
    )
    grants = javascript_grants(source, runtimes, required=True)
    assert grants["sdk_index"] == 4
    workspace = tmp_path / "work"
    workspace.mkdir()
    program = """import json,os,pathlib,subprocess,sys
root=pathlib.Path(os.environ['AGENTV_RUNNER']).parent.parent
code="const m=await import(process.argv[1]);if(typeof m.evaluate!=='function')process.exit(2)"
r=subprocess.run(['node','--input-type=module','-e',code,(root/'node_modules/@agentv/core/dist/index.js').as_uri()],capture_output=True,text=True,timeout=15)
assert r.returncode==0,r.stderr
for key in ('OPENUI_BRIDGE_CLI','DESIGN_MD_BRIDGE_CLI'):
 r=subprocess.run(['node',os.environ[key],'--repl'],input=json.dumps({'op':'ping'})+'\\n',text=True,capture_output=True,timeout=15)
 assert r.returncode==0 and json.loads(r.stdout).get('ok') is True,(key,r.stderr,r.stdout)
assert not pathlib.Path('/home/codex/repos/slm-training').exists()
from slm_training.dsl import design_md
assert pathlib.Path(design_md.__file__).is_relative_to('/runtime/5')
assert design_md._CLI == pathlib.Path('/runtime/3/cli.mjs')
assert design_md._BRIDGE_DIR == design_md._CLI.parent
assert design_md.bridge_available()
document='# Fixture design\\n\\nA fixture used only to test bridge transport.\\n'
repl=design_md.lint(document)
assert design_md._REPL_PROC is not None and design_md._REPL_PROC.poll() is None
design_md._close_repl()
design_md._LINT_CACHE.clear()
os.environ['DESIGN_MD_BRIDGE_NO_REPL']='1'
once=design_md.lint(document)
assert once == repl and 'findings' in once
assert design_md._REPL_PROC is None
negative="from slm_training.dsl import design_md; assert not design_md.bridge_available(); design_md.lint('fixture')"
env=dict(os.environ,DESIGN_MD_BRIDGE_CLI='/runtime/3/nonexistent.mjs')
r=subprocess.run([sys.executable,'-c',negative],env=env,capture_output=True,text=True,timeout=15)
assert r.returncode!=0 and 'DESIGN.md bridge unavailable' in r.stderr,r.stderr
print(json.dumps({'real_sdk_import':True,'openui_ping':True,'design_md_ping':True,
 'python_design_repl':True,'python_design_oneshot':True,'python_missing_override_refused':True}))
"""
    argv = runtime_argv(
        source,
        runtimes,
        ["/runtime/0/bin/python", "-c", program],
        workspace=workspace,
        required=True,
    )
    result = run_isolated(
        IsolationSpec(workspace, runtime_roots=runtimes, timeout_seconds=100), argv
    )
    assert result.returncode == 0 and not result.timed_out, result.stderr
    assert all(json.loads(result.stdout).values())
    assert sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file()) < 10000
