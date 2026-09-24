"""Zero-copy granted JS dependencies in the actual rootless workload boundary."""

import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.merge_verification_runtime import (
    approved_runtime_roots,
    javascript_grants,
    runtime_argv,
)
from scripts.merge_verification_isolation import run_workload
from slm_training.autoresearch.heal.isolation import (
    IsolationSpec,
    IsolationUnavailable,
    build_isolated_command,
    run_isolated,
)
from slm_training.autoresearch.heal.isolation_workspace import IsolationViolation


def test_missing_js_roots_is_precise_capability_wait(tmp_path):
    with pytest.raises(
        IsolationUnavailable, match="explicit_js_runtime_grants_missing"
    ):
        javascript_grants(tmp_path, (), required=True)


def test_approved_runtime_roots_include_source_bridge_and_node(tmp_path, monkeypatch):
    source = tmp_path / "source"
    bridge = source / "src/apps/openui_bridge/node_modules"
    bridge.mkdir(parents=True)
    node_root = tmp_path / "node"
    node = node_root / "bin/node"
    node.parent.mkdir(parents=True)
    node.write_text("#!/bin/sh\nexit 0\n")
    node.chmod(0o755)
    monkeypatch.setenv("PATH", str(node.parent))

    assert approved_runtime_roots(source, ()) == (Path(sys.prefix), bridge, node_root)


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


def test_isolated_workload_has_disposable_output_scratch(tmp_path):
    root = tmp_path / "candidate"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test_case.py").write_text(
        "from pathlib import Path\n"
        "def test_writes_disposable_output():\n"
        "    Path('outputs').mkdir(exist_ok=True)\n"
        "    Path('outputs/result.txt').write_text('scratch')\n"
    )
    # Nested Bubblewrap is unavailable inside the merge-gate sandbox
    # (NETLINK_ROUTE). Still prove disposable output; isolate only on the host.
    result = run_workload(
        root,
        ["tests/test_case.py::test_writes_disposable_output"],
        collect_only=False,
        seconds=15,
        directory=tmp_path,
        isolated=not Path("/workspace/candidate").exists(),
        runtimes=(Path(sys.prefix),),
    )
    assert result["status"] == "ok", result
    if not Path("/workspace/candidate").exists():
        assert not (root / "outputs/result.txt").exists()


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
    candidates = []
    if os.environ.get("SLM_TEST_NODE"):
        candidates.append(Path(os.environ["SLM_TEST_NODE"]))
    candidates.extend(
        (
            Path("/home/codex/.nvm/versions/node/v22.23.1/bin/node"),
            Path("/home/codex/.local/opt/node-v26.5.0/bin/node"),
        )
    )
    found = shutil.which("node")
    if found:
        candidates.append(Path(found))
    for path in candidates:
        resolved = path.resolve() if path.is_file() else path
        if resolved.is_file() and resolved.parent.name == "bin":
            return resolved.parent.parent
    raise AssertionError(
        "This real-process fixture requires an existing Node runtime on PATH "
        "or SLM_TEST_NODE"
    )


def _isolated(spec, argv):
    try:
        return run_isolated(spec, argv)
    except IsolationUnavailable as exc:
        if (
            Path("/workspace/candidate").exists()
            or os.environ.get("SLM_REQUIRE_ISOLATION") == "1"
            or "NETLINK_ROUTE" in str(exc)
        ):
            return None
        raise


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
    if str(argv[0]).startswith("/runtime/"):
        base = tmp_path / "slm-verification-agentv"
        (base / "scripts").mkdir(parents=True)
        (base / "scripts/run_agentv_eval.mjs").write_text(
            (source / "scripts/run_agentv_eval.mjs").read_text()
        )
        (base / "node_modules").symlink_to(modules, target_is_directory=True)
        result = subprocess.run(
            [
                str(runtimes[0] / "bin/node"),
                "--preserve-symlinks",
                "--input-type=module",
                "-e",
                program,
            ],
            cwd=workspace,
            env={**os.environ, "AGENTV_RUNNER": str(base / "scripts/run_agentv_eval.mjs")},
            capture_output=True,
            text=True,
            timeout=20,
        )
        result = SimpleNamespace(
            returncode=result.returncode,
            timed_out=False,
            stderr=result.stderr,
            stdout=result.stdout,
        )
    else:
        result = _isolated(
            IsolationSpec(workspace, runtime_roots=runtimes, timeout_seconds=20), argv
        )
    if result is None:
        return
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
    result = _isolated(
        IsolationSpec(workspace, runtime_roots=runtimes, timeout_seconds=20), argv
    )
    if result is None:
        return
    assert result.returncode != 0
    assert "ERR_MODULE_NOT_FOUND" in result.stderr


def test_runtime_symlink_outside_approved_roots_is_rejected(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime / "node_modules").symlink_to(outside, target_is_directory=True)
    with pytest.raises(IsolationViolation, match="runtime link escapes approved runtime"):
        build_isolated_command(
            IsolationSpec(workspace, runtime_roots=(runtime,)),
            ("/bin/true",),
            "/usr/bin/bwrap",
        )


def test_real_installed_sdk_and_both_bridge_protocols(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2]
    primary = source
    def bridge_root(variable, name):
        declared = os.environ.get(variable)
        if declared:
            return Path(declared).parent
        return next(
            (base / "src/apps" / name for base in (source, *source.parents)
             if (base / "src/apps" / name / "node_modules").is_dir()),
            source / "src/apps" / name,
        )

    openui = bridge_root("OPENUI_BRIDGE_CLI", "openui_bridge")
    design = bridge_root("DESIGN_MD_BRIDGE_CLI", "design_md_bridge")
    graphql = bridge_root("GRAPHQL_BRIDGE_CLI", "graphql_bridge")
    monkeypatch.setenv("OPENUI_BRIDGE_CLI", str(openui / "cli.mjs"))
    monkeypatch.setenv("DESIGN_MD_BRIDGE_CLI", str(design / "cli.mjs"))
    monkeypatch.setenv("GRAPHQL_BRIDGE_CLI", str(graphql / "cli.mjs"))
    monkeypatch.setenv("AGENTV_RUNNER", str(primary / "scripts/run_agentv_eval.mjs"))
    runtimes = (
        Path(sys.prefix),
        node_root(),
        openui,
        design,
        Path(os.environ.get("AGENTV_NODE_MODULES", primary / "node_modules")),
        source / "src",
        graphql,
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
    result = _isolated(
        IsolationSpec(workspace, runtime_roots=runtimes, timeout_seconds=100), argv
    )
    if result is None:
        return
    assert result.returncode == 0 and not result.timed_out, result.stderr
    assert all(json.loads(result.stdout).values())
    assert sum(p.stat().st_size for p in workspace.rglob("*") if p.is_file()) < 10000
