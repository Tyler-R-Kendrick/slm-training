"""Explicit read-only JS grants for the canonical isolated verifier.

Large installed dependency trees are mounted once by the existing isolation
backend, never copied. Only the content-bound AgentV entrypoint is staged; its
SDK-shaped scratch directory refers to the already-approved read-only mount.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
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
    "GRAPHQL_BRIDGE_CLI": (
        "graphql_bridge",
        ("cli.mjs", "package.json", "package-lock.json"),
    ),
}


def approved_runtime_roots(source: Path, requested: list[Path]) -> tuple[Path, ...]:
    roots = list(requested) or [Path(sys.prefix)]
    bridge = source.resolve() / "src/apps/openui_bridge/node_modules"
    if bridge.is_dir() and bridge not in roots:
        roots.append(bridge)
    node = shutil.which("node")
    if node:
        node_root = Path(node).resolve().parent.parent
        if node_root.is_dir() and node_root not in roots:
            roots.append(node_root)
    return tuple(roots)

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
        if not (runtime / "cli.mjs").is_file():
            continue
        if not all(
            (source / "src/apps" / name / filename).is_file()
            and (runtime / filename).is_file()
            and file_digest(source / "src/apps" / name / filename)
            == file_digest(runtime / filename)
            for filename in files
        ):
            if runtime.name == name:
                raise ValueError("bridge_runtime_source_mismatch:" + name)
            continue
        declared = os.environ.get(variable)
        if declared and Path(declared).resolve() != runtime / "cli.mjs":
            raise ValueError("declared_bridge_not_in_approved_runtime:" + variable)
        if not (runtime / "node_modules").is_dir():
            raise IsolationUnavailable("bridge_transitive_dependencies_missing:" + name)
        return {variable: f"/runtime/{index}/cli.mjs"}
    return {}


def _locked_dependency(runtime, packages, importer, name):
    parent = importer
    while parent == runtime or runtime in parent.parents:
        package = parent / "node_modules" / name
        key = package.relative_to(runtime).as_posix()
        record = packages.get(key)
        if record is not None:
            manifest = json.loads((package / "package.json").read_text())
            if manifest.get("version") != record.get("version"):
                raise ValueError("playwright_installed_package_lock_mismatch:" + name)
            return package, manifest
        parent = parent.parent
    raise ValueError("playwright_locked_dependency_missing:" + name)


def _playwright_source_lock_matches(runtime, source_files):
    return all(path.is_file() for path in source_files) and all(
        (runtime / path.name).is_file()
        and file_digest(runtime / path.name) == file_digest(path)
        for path in source_files
    )


def _playwright_revision(source_files, runtime):
    test_package = runtime / "node_modules/@playwright/test/package.json"
    if not test_package.exists():
        if _playwright_source_lock_matches(runtime, source_files) and (
            runtime / "node_modules/@playwright"
        ).exists():
            raise ValueError("incomplete_playwright_runtime_grant")
        return None
    if not _playwright_source_lock_matches(runtime, source_files):
        raise ValueError("playwright_runtime_source_lock_mismatch")
    packages = json.loads(source_files[1].read_text()).get("packages", {})
    test_record = packages.get("node_modules/@playwright/test")
    test_doc = json.loads(test_package.read_text())
    if not test_record or test_doc.get("version") != test_record.get("version"):
        raise ValueError("playwright_test_package_lock_mismatch")
    playwright, _ = _locked_dependency(runtime, packages, test_package.parent, "playwright")
    core, _ = _locked_dependency(runtime, packages, playwright.parent, "playwright-core")
    browsers = json.loads((core / "browsers.json").read_text())
    revisions = [
        str(item["revision"])
        for item in browsers["browsers"]
        if item.get("name") == "chromium-headless-shell"
    ]
    if len(revisions) != 1:
        raise ValueError("playwright_chromium_revision_missing_or_ambiguous")
    return revisions[0]


def _playwright_browser_index(runtimes, revision):
    browser_index = None
    expected = f"chromium_headless_shell-{revision}"
    for index, runtime in enumerate(runtimes):
        for browser in runtime.glob("chromium_headless_shell-*"):
            if browser.name != expected or browser.is_symlink():
                raise ValueError("playwright_browser_revision_mismatch")
            executable = browser / "chrome-linux/headless_shell"
            if executable.is_symlink() or not executable.is_file():
                raise ValueError("playwright_browser_executable_missing_or_unsafe")
            if browser_index is not None:
                raise ValueError("ambiguous_playwright_browser_grant")
            browser_index = index
    return browser_index


def _playwright_grants(source, runtimes):
    source_files = (source / "package.json", source / "package-lock.json")
    complete = []
    for index, runtime in enumerate(runtimes):
        revision = _playwright_revision(source_files, runtime.resolve(strict=True))
        if revision is not None:
            complete.append((index, revision))
    if len(complete) > 1:
        raise ValueError("ambiguous_playwright_runtime_grant")
    if not complete:
        return None, None
    playwright_index, revision = complete[0]
    return playwright_index, _playwright_browser_index(runtimes, revision)


def _requires_playwright(targets):
    return any(
        str(target).split("::", 1)[0].rstrip("/")
        in {"tests", "tests/test_data", "tests/test_data/test_verify.py"}
        for target in targets
    )


def preflight_js(args):
    if args.require_js_runtime:
        return javascript_grants(
            args.source.resolve(), tuple(args.runtime_root), required=True
        )
    return None


def _record_js_runtime(source, index, runtime, grants, node_indexes):
    runtime = runtime.resolve(strict=True)
    if (runtime / "bin/node").is_file():
        grants["node_index"] = index
        node_indexes.append(index)
    if (runtime / "@agentv/core/dist/index.js").is_file():
        if grants["sdk_index"] is not None:
            raise ValueError("ambiguous_agentv_runtime_grant")
        grants["sdk_index"] = index
    grants["bridges"].update(bridge_grant(source, runtime, index))


def _require_playwright_grants(grants, node_indexes, targets):
    if not _requires_playwright(targets):
        return
    if grants["playwright_index"] is None or grants["playwright_browser_index"] is None:
        raise IsolationUnavailable("explicit_playwright_runtime_grants_missing")
    if grants["node_index"] is None:
        raise IsolationUnavailable("explicit_playwright_node_runtime_missing")
    if len(node_indexes) != 1:
        raise ValueError("ambiguous_playwright_node_runtime_grant")


def _require_js_bundle(grants, required):
    if not required:
        return
    missing = [name for name in BRIDGES if name not in grants["bridges"]]
    missing += [name for name in ("sdk_index", "node_index") if grants[name] is None]
    if missing:
        raise IsolationUnavailable("explicit_js_runtime_grants_missing:" + ",".join(missing))


def _authorize_agentv(source, grants):
    if grants["sdk_index"] is None:
        return
    if grants["node_index"] is None:
        raise IsolationUnavailable("agentv_requires_explicit_node_runtime")
    runner = Path(
        os.environ.get("AGENTV_RUNNER", source / "scripts/run_agentv_eval.mjs")
    ).resolve()
    expected = source / "scripts/run_agentv_eval.mjs"
    if not expected.is_file() or file_digest(runner) != file_digest(expected):
        raise ValueError("agentv_runner_source_mismatch")
    grants["runner"] = str(runner)


def javascript_grants(
    source: Path, runtimes: tuple[Path, ...], *, required=False, targets=()
):
    """Classify explicit roots; no discovery authorizes an additional mount."""
    grants = {
        "bridges": {},
        "sdk_index": None,
        "node_index": None,
        "playwright_index": None,
        "playwright_browser_index": None,
    }
    node_indexes = []
    for index, runtime in enumerate(runtimes):
        _record_js_runtime(source, index, runtime, grants, node_indexes)
    grants["playwright_index"], grants["playwright_browser_index"] = (
        _playwright_grants(source, runtimes)
    )
    _require_playwright_grants(grants, node_indexes, targets)
    _require_js_bundle(grants, required)
    _authorize_agentv(source, grants)
    return grants


def runtime_argv(source, runtimes, argv, *, workspace=None, required=False):
    """Compose environment and tiny SDK adapter within the existing boundary."""
    grants = javascript_grants(source, runtimes, required=required)
    pythonpath = ["/workspace/candidate/src"] + [
        f"/runtime/{index}" for index in range(1, len(runtimes))
    ]
    bins = (["/workspace/control/bin"] if workspace is not None and (workspace / "control/bin/git").is_file() else []) + [
        f"/runtime/{index}/bin" for index in range(len(runtimes))
    ]
    settings = [
        "PYTHONPATH=" + ":".join(pythonpath),
        "PATH=" + ":".join([*bins, "/usr/bin", "/bin"]),
    ]
    if grants["node_index"] is not None:
        settings.append(f"SLM_TEST_NODE=/runtime/{grants['node_index']}/bin/node")
    library_paths = [
        f"/runtime/{index}/lib"
        for index, root in enumerate(runtimes)
        if (root / "libstdc++.so.6").is_file()
    ]
    if library_paths:
        settings.append("LD_LIBRARY_PATH=" + ":".join(library_paths))
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


def runtime_roots_with_js_modules(runtimes, grants):
    index = grants["playwright_index"]
    if index is None:
        return runtimes
    modules = runtimes[index].resolve(strict=True) / "node_modules"
    if modules.is_symlink():
        raise ValueError("playwright_node_modules_symlink_not_allowed")
    modules = modules.resolve(strict=True)
    if modules in {root.resolve() for root in runtimes}:
        return runtimes
    return (*runtimes, modules)
