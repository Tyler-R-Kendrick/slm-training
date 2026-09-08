"""Explicit read-only JS grants for the canonical isolated verifier.

Large installed dependency trees are mounted once by the existing isolation
backend, never copied. Only the content-bound AgentV entrypoint is staged; its
SDK-shaped scratch directory refers to the already-approved read-only mount.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.merge_verification_evidence import file_digest
from slm_training.autoresearch.heal.isolation import IsolationUnavailable

BRIDGES = {
    "OPENUI_BRIDGE_CLI": (
        "openui_bridge",
        ("cli.mjs", "library.mjs", "package.json", "package-lock.json"),
    ),
    "DESIGN_MD_BRIDGE_CLI": (
        "design_md_bridge",
        ("cli.mjs", "package.json", "package-lock.json"),
    ),
}

# This runs inside the existing namespace, without network/host home/store.
# The entrypoint remains byte-identical; Node resolves all transitive imports
# through the complete granted node_modules tree, not a fabricated SDK result.
SDK_BOOTSTRAP = """import json,os,pathlib,shlex,shutil,sys
config=json.loads(sys.argv[1])
base=pathlib.Path('/tmp/slm-verification-agentv')
(base/'scripts').mkdir(parents=True,exist_ok=True)
shutil.copyfile(config['runner'],base/'scripts/run_agentv_eval.mjs')
(base/'node_modules').symlink_to(config['modules'],target_is_directory=True)
(base/'bin').mkdir()
node=base/'bin/node'
node.write_text('#!/bin/sh\\nexec '+shlex.quote(config['node'])+' --preserve-symlinks "$@"\\n')
node.chmod(0o755)
os.environ['PATH']=str(base/'bin')+':'+os.environ['PATH']
os.environ['AGENTV_RUNNER']=str(base/'scripts/run_agentv_eval.mjs')
argv=sys.argv[2:]
if argv[0]==config['node']: argv.insert(1,'--preserve-symlinks')
os.execvpe(argv[0],argv,os.environ)
"""


def bridge_grant(source, runtime, index):
    for variable, (name, files) in BRIDGES.items():
        if runtime.name != name:
            continue
        for filename in files:
            expected, installed = (
                source / "src/apps" / name / filename,
                runtime / filename,
            )
            if (
                not expected.is_file()
                or not installed.is_file()
                or file_digest(expected) != file_digest(installed)
            ):
                raise ValueError(
                    "bridge_runtime_source_mismatch:" + name + "/" + filename
                )
        declared = os.environ.get(variable)
        if declared and Path(declared).resolve() != runtime / "cli.mjs":
            raise ValueError("declared_bridge_not_in_approved_runtime:" + variable)
        if not (runtime / "node_modules").is_dir():
            raise IsolationUnavailable("bridge_transitive_dependencies_missing:" + name)
        return {variable: f"/runtime/{index}/cli.mjs"}
    return {}


def preflight_js(args):
    if args.require_js_runtime:
        return javascript_grants(
            args.source.resolve(), tuple(args.runtime_root), required=True
        )
    return None


def javascript_grants(source: Path, runtimes: tuple[Path, ...], *, required=False):
    """Classify explicit roots; no discovery authorizes an additional mount."""
    grants = {"bridges": {}, "sdk_index": None, "node_index": None}
    for index, runtime in enumerate(runtimes):
        runtime = runtime.resolve(strict=True)
        if (runtime / "bin/node").is_file():
            grants["node_index"] = index
        if (
            runtime.name == "node_modules"
            and (runtime / "@agentv/core/dist/index.js").is_file()
        ):
            if grants["sdk_index"] is not None:
                raise ValueError("ambiguous_agentv_runtime_grant")
            grants["sdk_index"] = index
        grants["bridges"].update(bridge_grant(source, runtime, index))
    if required:
        missing = [name for name in BRIDGES if name not in grants["bridges"]]
        missing += [
            name for name in ("sdk_index", "node_index") if grants[name] is None
        ]
        if missing:
            raise IsolationUnavailable(
                "explicit_js_runtime_grants_missing:" + ",".join(missing)
            )
    if grants["sdk_index"] is not None:
        if grants["node_index"] is None:
            raise IsolationUnavailable("agentv_requires_explicit_node_runtime")
        runner = Path(
            os.environ.get("AGENTV_RUNNER", source / "scripts/run_agentv_eval.mjs")
        ).resolve()
        expected = source / "scripts/run_agentv_eval.mjs"
        if not expected.is_file() or file_digest(runner) != file_digest(expected):
            raise ValueError("agentv_runner_source_mismatch")
        grants["runner"] = str(runner)
    return grants


def runtime_argv(source, runtimes, argv, *, workspace=None, required=False):
    """Compose environment and tiny SDK adapter within the existing boundary."""
    grants = javascript_grants(source, runtimes, required=required)
    pythonpath = ["/workspace/candidate/src"] + [
        f"/runtime/{index}" for index in range(1, len(runtimes))
    ]
    bins = [f"/runtime/{index}/bin" for index in range(len(runtimes))]
    settings = [
        "PYTHONPATH=" + ":".join(pythonpath),
        "PATH=" + ":".join([*bins, "/usr/bin", "/bin"]),
    ]
    settings.extend(f"{key}={value}" for key, value in grants["bridges"].items())
    if grants["sdk_index"] is not None:
        if workspace is None:
            raise IsolationUnavailable("agentv_requires_private_control_workspace")
        import shutil

        staged = workspace / "control/js/run_agentv_eval.mjs"
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(grants["runner"], staged)
        config = {
            "runner": "/workspace/control/js/run_agentv_eval.mjs",
            "modules": f"/runtime/{grants['sdk_index']}",
            "node": f"/runtime/{grants['node_index']}/bin/node",
        }
        argv = [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            SDK_BOOTSTRAP,
            json.dumps(config),
            *argv,
        ]
    return ["/usr/bin/env", *settings, *argv]
