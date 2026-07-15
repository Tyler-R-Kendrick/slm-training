"""Run lightweight checks for the suites affected by local changes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_TEST_FILES = {
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "setup.py",
    "tests/conftest.py",
}
SUITES_BY_PREFIX = (
    ("api/", ("tests/test_web",)),
    ("gpu_multi_farm/", ("tests/test_gpu_multi_farm",)),
    ("grammars/", ("tests/test_dsl", "tests/test_harnesses/model_build")),
    (
        "scripts/build_train_data.py",
        ("tests/test_data", "tests/test_harnesses/train_data"),
    ),
    (
        "scripts/build_test_data.py",
        ("tests/test_data", "tests/test_harnesses/test_data", "tests/test_integration"),
    ),
    (
        "scripts/hf_jobs_train.py",
        ("tests/test_runtime/accel", "tests/test_harnesses/model_build"),
    ),
    ("scripts/model_cycle.py", ("tests/test_lineage",)),
    (
        "scripts/remote_train.py",
        ("tests/test_runtime/accel", "tests/test_regressions"),
    ),
    (
        "scripts/train_",
        ("tests/test_harnesses/model_build", "tests/test_harnesses/quality", "tests/test_harnesses/rl"),
    ),
    (
        "scripts/evaluate_",
        ("tests/test_evals", "tests/test_harnesses/model_build"),
    ),
    (
        "scripts/serve_playground.py",
        ("tests/test_web",),
    ),
    ("scripts/", ("tests/test_scripts",)),
    (
        "src/slm_training/data/",
        (
            "tests/test_data",
            "tests/test_harnesses/test_data",
            "tests/test_harnesses/train_data",
            "tests/test_rico",
        ),
    ),
    ("src/slm_training/dsl/", ("tests/test_dsl", "tests/test_harnesses/model_build")),
    ("src/slm_training/evals/", ("tests/test_evals", "tests/test_harnesses/model_build")),
    ("src/slm_training/harnesses/distill/", ("tests/test_harnesses/distill",)),
    (
        "src/slm_training/harnesses/experiments/",
        ("tests/test_harnesses/experiments",),
    ),
    (
        "src/slm_training/harnesses/model_build/",
        ("tests/test_harnesses/model_build",),
    ),
    ("src/slm_training/harnesses/quality/", ("tests/test_harnesses/quality",)),
    ("src/slm_training/harnesses/rl/", ("tests/test_harnesses/rl",)),
    (
        "src/slm_training/harnesses/test_data/",
        ("tests/test_harnesses/test_data",),
    ),
    (
        "src/slm_training/harnesses/train_data/",
        ("tests/test_harnesses/train_data",),
    ),
    (
        "src/slm_training/lineage/",
        ("tests/test_lineage", "tests/test_web/test_lineage_deployments.py"),
    ),
    (
        "src/slm_training/models/",
        ("tests/test_models", "tests/test_harnesses/model_build"),
    ),
    ("src/slm_training/runtime/", ("tests/test_runtime",)),
    ("src/slm_training/web/", ("tests/test_web",)),
    (
        "tools/openui_bridge/",
        ("tests/test_dsl", "tests/test_awwwards", "tests/test_regressions"),
    ),
    ("tools/design_md_bridge/", ("tests/test_dsl/design_md",)),
    ("tools/dashboard/", ("tests/test_web",)),
    ("tools/openui_preview/", ("tests/test_web",)),
)
CODE_SUFFIXES = {".c", ".css", ".html", ".js", ".json", ".mjs", ".py", ".ts", ".tsx", ".yaml", ".yml"}


def changed_files(*, staged: bool) -> list[str]:
    diff = ["git", "diff"]
    diff += ["--cached"] if staged else ["HEAD"]
    diff += ["--name-only", "--diff-filter=ACMRD", "--"]
    paths = set(_git(diff).splitlines())
    if not staged:
        paths.update(
            _git(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
        )
    return sorted(path for path in paths if path)


def select_tests(paths: list[str]) -> list[str]:
    """Return conservative pytest targets for repo-relative changed paths."""
    targets: set[str] = set()
    for path in paths:
        if path in GLOBAL_TEST_FILES:
            return ["tests"]
        if path.startswith("tests/") and path.endswith(".py"):
            targets.add(path)
            continue
        matches = [suites for prefix, suites in SUITES_BY_PREFIX if path.startswith(prefix)]
        if matches:
            targets.update(suite for suites in matches for suite in suites)
            continue
        if path.startswith(("docs/", "openwiki/")) or Path(path).suffix in {
            ".md",
            ".rst",
            ".txt",
        }:
            continue
        if Path(path).suffix in CODE_SUFFIXES:
            return ["tests"]
    return _remove_nested_targets(targets)


def check(paths: list[str]) -> int:
    tests = select_tests(paths)
    python_paths = [path for path in paths if path.endswith(".py") and (ROOT / path).is_file()]
    print(f"changed-check: {len(paths)} file(s), pytest targets: {', '.join(tests) or 'none'}")
    if python_paths and _run([sys.executable, "-m", "ruff", "check", *python_paths]):
        return 1
    if python_paths and _run([sys.executable, "-m", "py_compile", *python_paths]):
        return 1
    if tests and _run([sys.executable, "-m", "pytest", "-q", *tests]):
        return 1
    return 0


def _git(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _remove_nested_targets(targets: set[str]) -> list[str]:
    return sorted(
        target
        for target in targets
        if not any(target.startswith(parent.rstrip("/") + "/") for parent in targets - {target})
    )


def _run(command: list[str]) -> int:
    print("+", " ".join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="check only staged paths")
    parser.add_argument("--list", action="store_true", help="print selected tests without running")
    parser.add_argument(
        "--hook",
        action="store_true",
        help="emit an agent Stop-hook decision instead of regular output",
    )
    args = parser.parse_args(argv)
    paths = changed_files(staged=args.staged)
    if args.list:
        print("\n".join(select_tests(paths)))
        return 0
    if args.hook:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_changed"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            detail = (result.stdout + result.stderr)[-8000:]
            print(json.dumps({"decision": "block", "reason": f"Changed tests failed:\n{detail}"}))
        else:
            print(json.dumps({"decision": "allow"}))
        return 0
    return check(paths)


if __name__ == "__main__":
    raise SystemExit(main())
