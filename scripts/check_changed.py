"""Run lightweight checks for the suites affected by local changes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.repo_policy import validate_repository


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_TEST_FILES = {
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "setup.py",
    "tests/conftest.py",
}
SUITES_BY_PREFIX = (
    ("gpu_multi_farm/", ("tests/test_gpu_multi_farm",)),
    ("src/slm_training/dsl/grammars/", ("tests/test_dsl", "tests/test_harnesses/model_build")),
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
    ("scripts/autoresearch.py", ("tests/test_autoresearch",)),
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
    # Checkpoint-reference audit: committed frontier/ship claims must stay
    # resolvable, so model-card and README changes run it. (Design result JSON
    # is covered by the always-on CI audit step; docs stay otherwise
    # conservative for the local hook.)
    (
        "docs/MODEL_CARD.md",
        ("tests/test_scripts/test_verify_checkpoint_references.py",),
    ),
    (
        "README.md",
        ("tests/test_scripts/test_verify_checkpoint_references.py",),
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
    ("src/slm_training/autoresearch/", ("tests/test_autoresearch",)),
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
        "src/apps/openui_bridge/",
        ("tests/test_dsl", "tests/test_awwwards", "tests/test_regressions"),
    ),
    ("src/apps/design_md_bridge/", ("tests/test_dsl/design_md",)),
    ("src/apps/dashboard/", ("tests/test_web",)),
    ("src/apps/openui_preview/", ("tests/test_web",)),
)
CODE_SUFFIXES = {".c", ".css", ".html", ".js", ".json", ".mjs", ".py", ".ts", ".tsx", ".yaml", ".yml"}
HOOK_TEST_FILE_LIMIT = 100


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
    unknown_code = False
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
        if path.startswith("docs/") or Path(path).suffix in {
            ".md",
            ".rst",
            ".txt",
        }:
            continue
        if Path(path).suffix in CODE_SUFFIXES:
            unknown_code = True
    if targets:
        return _remove_nested_targets(targets)
    if unknown_code:
        return ["tests"]
    return _remove_nested_targets(targets)


def select_changed_tests(paths: list[str]) -> list[str]:
    """Prefer explicit regression files for latency-bounded local hooks."""
    changed = {
        path for path in paths if path.startswith("tests/") and path.endswith(".py")
    }
    return sorted(changed) if changed else select_tests(paths)


def hook_test_targets(paths: list[str]) -> list[str]:
    """Keep local hooks bounded; CI owns broad validation for large diffs."""
    if len(paths) > HOOK_TEST_FILE_LIMIT:
        return []
    return select_changed_tests(paths)


def check(paths: list[str], *, changed_tests_only: bool = False) -> int:
    policy_errors = validate_repository()
    if policy_errors:
        print("repo-policy: failed")
        for error in policy_errors:
            print(f"- {error}")
        return 1
    tests = hook_test_targets(paths) if changed_tests_only else select_tests(paths)
    python_paths = [path for path in paths if path.endswith(".py") and (ROOT / path).is_file()]
    print(f"changed-check: {len(paths)} file(s), pytest targets: {', '.join(tests) or 'none'}")
    if changed_tests_only and len(paths) > HOOK_TEST_FILE_LIMIT:
        print(
            f"changed-check: pytest deferred for large diff ({len(paths)} > "
            f"{HOOK_TEST_FILE_LIMIT} files); run the listed suites manually or in CI"
        )
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
    parser.add_argument(
        "--changed-tests-only",
        action="store_true",
        help="prefer explicitly changed test files; intended for latency-bounded hooks",
    )
    parser.add_argument("--list", action="store_true", help="print selected tests without running")
    parser.add_argument(
        "--hook",
        action="store_true",
        help="emit an agent Stop-hook decision instead of regular output",
    )
    args = parser.parse_args(argv)
    paths = changed_files(staged=args.staged)
    if args.list:
        selected = (
            hook_test_targets(paths) if args.changed_tests_only else select_tests(paths)
        )
        print("\n".join(selected))
        return 0
    if args.hook:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_changed", "--changed-tests-only"],
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
    return check(paths, changed_tests_only=args.changed_tests_only)


if __name__ == "__main__":
    raise SystemExit(main())
