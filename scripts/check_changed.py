"""Run lightweight checks for the suites affected by local changes."""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.repo_policy import validate_repository
from scripts.verify_agent_surfaces import OBLIGATIONS


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHANGED_TEST_WORKERS: int | None = None
# Read as a plain repo-relative path rather than importlib.resources so the
# table resolves in worktrees and uninstalled checkouts too.
TEST_DURATIONS_PATH = (
    ROOT / "src" / "slm_training" / "resources" / "ci" / "test_durations_v1.json"
)


# Editing any instruction surface re-certifies the whole parity matrix; the
# laws only stay identical across harnesses if a one-sided edit is caught.
AGENT_SURFACE_FILES = frozenset(
    relative for obligation in OBLIGATIONS for relative in obligation.surfaces()
) | {"scripts/verify_agent_surfaces.py"}
GLOBAL_TEST_FILES = {
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "setup.py",
    "tests/conftest.py",
    "tests/casefiles.py",
}
CASE_RESOURCE_PREFIX = "src/slm_training/resources/test_cases/"
EVAL_RESOURCE_PREFIX = "src/slm_training/resources/evals/"
NON_PYTHON_LOCKFILES = {
    "package-lock.json",
    "src/apps/dashboard/package-lock.json",
    "src/apps/design_md_bridge/package-lock.json",
    "src/apps/openui_bridge/package-lock.json",
}
SUITES_BY_PREFIX = (
    (".agents/skills/autotrain/", ("tests/test_scripts/test_slm_cli.py",)),
    (
        ".agents/skills/autoresearch/",
        ("tests/test_scripts/test_autoresearch_skill.py",),
    ),
    ("docs/brains/", ("tests/test_scripts/test_autoresearch_skill.py",)),
    ("gpu_multi_farm/", ("tests/test_gpu_multi_farm",)),
    (
        "src/slm_training/dsl/grammars/",
        ("tests/test_dsl", "tests/test_harnesses/model_build"),
    ),
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
        (
            "tests/test_harnesses/model_build",
            "tests/test_harnesses/quality",
            "tests/test_harnesses/rl",
        ),
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
    (
        "scripts/verify_agent_surfaces.py",
        ("tests/test_scripts/test_verify_agent_surfaces.py",),
    ),
    (
        "scripts/check_changed.py",
        ("tests/test_scripts/test_check_changed.py",),
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
    (
        "src/slm_training/evals/",
        ("tests/test_evals", "tests/test_harnesses/model_build"),
    ),
    (
        "src/slm_training/harness_core/",
        (
            "tests/test_harness_core",
            "tests/test_evals",
            "tests/test_harnesses/experiments",
            "tests/test_harnesses/model_build",
            "tests/test_lineage",
            "tests/test_versioning",
        ),
    ),
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
        (
            "tests/test_models",
            "tests/test_harnesses/model_build",
            # The tokenizer layout is a checkpoint-bound contract and the
            # tokenizers live under models/; keep its pin selected from here
            # even though the regression file sits with the DSL suites.
            "tests/test_dsl/test_tokenizer_grammar_invariants.py",
        ),
    ),
    (
        "src/slm_training/resources/versions.json",
        ("tests/test_versioning",),
    ),
    ("src/slm_training/runtime/", ("tests/test_runtime",)),
    ("src/slm_training/versioning.py", ("tests/test_versioning",)),
    ("src/slm_training/web/", ("tests/test_web",)),
    (
        "src/apps/openui_bridge/",
        ("tests/test_dsl", "tests/test_awwwards", "tests/test_regressions"),
    ),
    ("src/apps/design_md_bridge/", ("tests/test_dsl/design_md",)),
    ("src/apps/dashboard/", ("tests/test_web",)),
    ("src/apps/openui_preview/", ("tests/test_web",)),
)
CODE_SUFFIXES = {
    ".c",
    ".css",
    ".html",
    ".js",
    ".json",
    ".mjs",
    ".py",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
HOOK_TEST_FILE_LIMIT = 100


def changed_files(*, staged: bool, base_ref: str | None = None) -> list[str]:
    if base_ref:
        return sorted(
            path
            for path in _git(
                [
                    "git",
                    "diff",
                    "--name-only",
                    "--diff-filter=ACMRD",
                    f"{base_ref}...HEAD",
                    "--",
                ]
            ).splitlines()
            if path
        )
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
        if path in NON_PYTHON_LOCKFILES:
            # The CI install step validates lockfile integrity. Do not spend the
            # bounded Python-test budget on an unrelated full suite.
            continue
        if path.startswith("tests/") and path.endswith(".py"):
            if (ROOT / path).is_file():
                targets.add(path)
            continue
        if path.startswith(CASE_RESOURCE_PREFIX) and path.endswith(".json"):
            relative = Path(path.removeprefix(CASE_RESOURCE_PREFIX)).with_suffix(".py")
            test_path = Path("tests") / relative
            if (ROOT / test_path).is_file():
                targets.add(test_path.as_posix())
            continue
        if path in {
            f"{EVAL_RESOURCE_PREFIX}openui_ship_gates_v5.json",
            f"{EVAL_RESOURCE_PREFIX}openui_ship_gates_v6.json",
        }:
            targets.update(
                {
                    "tests/test_harness_core/test_gate_engine_golden.py",
                    "tests/test_harnesses/model_build/test_eval_gates.py",
                }
            )
            continue
        if path.startswith(EVAL_RESOURCE_PREFIX):
            targets.update({"tests/test_evals", "tests/test_harnesses/model_build"})
            continue
        if path == "src/slm_training/harnesses/model_build/ship_gates.py":
            targets.add("tests/test_harnesses/model_build/test_eval_gates.py")
            continue
        # Exact path entries own the mapping exclusively (narrow policy/script
        # hooks without dragging the broad scripts/ suite).
        exact = next(
            (suites for prefix, suites in SUITES_BY_PREFIX if prefix == path),
            None,
        )
        if exact is not None:
            targets.update(exact)
            continue
        matches = [
            suites for prefix, suites in SUITES_BY_PREFIX if path.startswith(prefix)
        ]
        if matches:
            targets.update(suite for suites in matches for suite in suites)
            continue
        if path.startswith("docs/") or Path(path).suffix in {
            ".md",
            ".rst",
            ".txt",
        }:
            continue
        if path.startswith(".github/workflows/"):
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
        path
        for path in paths
        if path.startswith("tests/")
        and path.endswith(".py")
        and Path(path).name.startswith("test_")
        and (ROOT / path).is_file()
    }
    return sorted(changed) if changed else select_tests(paths)


def hook_test_targets(paths: list[str]) -> list[str]:
    """Keep local hooks bounded; CI owns broad validation for large diffs."""
    if len(paths) > HOOK_TEST_FILE_LIMIT:
        return []
    return select_changed_tests(paths)


def check(
    paths: list[str],
    *,
    changed_tests_only: bool = False,
    staged: bool = False,
    test_shard_index: int = 0,
    test_shard_count: int = 1,
    base_ref: str | None = None,
) -> int:
    policy_errors = validate_repository()
    if policy_errors:
        print("repo-policy: failed")
        for error in policy_errors:
            print(f"- {error}")
        return 1
    stamp_command = [sys.executable, "-m", "scripts.verify_version_stamps", "--check"]
    if staged:
        stamp_command.append("--staged")
    if _run(stamp_command):
        return 1
    case_paths = [
        path
        for path in paths
        if path.startswith(CASE_RESOURCE_PREFIX)
        or path.startswith(EVAL_RESOURCE_PREFIX)
    ]
    if case_paths:
        case_command = [
            sys.executable,
            "-m",
            "scripts.refresh_test_cases",
            "--check",
        ]
        if base_ref:
            case_command.extend(["--base-ref", base_ref])
        resource_cases = [
            path for path in case_paths if path.startswith(CASE_RESOURCE_PREFIX)
        ]
        case_command.extend(resource_cases or ["--changed"])
        if _run(case_command):
            return 1
    # Cheap, no imports beyond stdlib: every configured harness must still
    # carry every repository law. Only re-checked when a surface changed.
    if any(path in AGENT_SURFACE_FILES for path in paths):
        if _run([sys.executable, "-m", "scripts.verify_agent_surfaces"]):
            return 1
    tests = hook_test_targets(paths) if changed_tests_only else select_tests(paths)
    python_paths = [
        path for path in paths if path.endswith(".py") and (ROOT / path).is_file()
    ]
    print(
        f"changed-check: {len(paths)} file(s), pytest targets: {', '.join(tests) or 'none'}"
    )
    if changed_tests_only and len(paths) > HOOK_TEST_FILE_LIMIT:
        print(
            f"changed-check: pytest deferred for large diff ({len(paths)} > "
            f"{HOOK_TEST_FILE_LIMIT} files); run the listed suites manually or in CI"
        )
    if python_paths and _run([sys.executable, "-m", "ruff", "check", *python_paths]):
        return 1
    if python_paths and _run([sys.executable, "-m", "py_compile", *python_paths]):
        return 1
    if tests and changed_tests_only and (len(tests) > 1 or test_shard_count > 1):
        if _run_changed_tests_parallel(
            tests,
            shard_index=test_shard_index,
            shard_count=test_shard_count,
        ):
            return 1
    elif tests and _run([sys.executable, "-m", "pytest", "-q", *tests]):
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
        if not any(
            target.startswith(parent.rstrip("/") + "/") for parent in targets - {target}
        )
    )


def _run(command: list[str]) -> int:
    print("+", " ".join(command))
    env = _pytest_worker_env() if command[1:4] == ["-m", "pytest", "-q"] else None
    return subprocess.run(command, cwd=ROOT, check=False, env=env).returncode


def _pytest_worker_env() -> dict[str, str]:
    """Keep parallel pytest workers from oversubscribing CI CPU threads."""
    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    return env


def _run_changed_tests_parallel(
    tests: list[str],
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> int:
    workers = _changed_test_workers()
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *tests],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env=_pytest_worker_env(),
        text=True,
    )
    if collected.returncode:
        print(collected.stdout, end="")
        print(collected.stderr, end="", file=sys.stderr)
        return collected.returncode
    nodes = [line for line in collected.stdout.splitlines() if "::" in line]
    nodes = _shard_test_nodes(nodes, shard_count)[shard_index]
    print(
        f"changed-check: test shard {shard_index + 1}/{shard_count} "
        f"owns {len(nodes)} node(s)"
    )
    if not nodes:
        return 0
    if len(nodes) < 2:
        return _run([sys.executable, "-m", "pytest", "-q", *nodes])
    # More batches than workers let the executor steal remaining work when
    # individual test costs differ sharply despite equal node counts.
    batch_count = min(len(nodes), workers * 2)
    batches = _shard_test_nodes(nodes, batch_count)
    commands = [
        [sys.executable, "-m", "pytest", "-q", *batch] for batch in batches if batch
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        return int(any(executor.map(_run, commands)))


def _changed_test_workers() -> int:
    if CHANGED_TEST_WORKERS is not None:
        return CHANGED_TEST_WORKERS
    from slm_training.levers import CHANGED_TEST_WORKERS as configured_workers

    return configured_workers


@functools.lru_cache(maxsize=1)
def _test_file_durations() -> dict[str, float]:
    """Committed per-file wall-clock weights; empty when absent or unreadable.

    Duration weights only reorder the shard packing, so a missing or corrupt
    table degrades to the historical count-balanced behaviour instead of
    failing the run.
    """
    try:
        payload = json.loads(TEST_DURATIONS_PATH.read_text(encoding="utf-8"))
        durations = payload["durations"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if not isinstance(durations, dict):
        return {}
    return {
        str(path): float(seconds)
        for path, seconds in durations.items()
        if isinstance(seconds, (int, float)) and float(seconds) > 0.0
    }


def _shard_test_nodes(
    nodes: list[str],
    workers: int,
    durations: dict[str, float] | None = None,
) -> list[list[str]]:
    """Pack test nodes into ``workers`` shards, heaviest node first.

    The objective is **minimising the slowest shard**, because the canonical
    per-job wall (``MAX_RUN_MINUTES``) is enforced per shard: a packing is
    good exactly when its worst shard fits.

    A measured file's weight is therefore spread across its own nodes rather
    than kept whole. Keeping files atomic (the original duration-aware
    packing) isolates a slow suite but makes the worst shard *equal to the
    single heaviest file* -- on the committed table that is 429s on one shard
    versus 143s when the same file's nodes are allowed to spread, i.e. the
    isolation objective actively defeats the wall the sharding exists to
    satisfy. Unmeasured files keep one unit of weight per node, so an empty
    weights table still reduces exactly to count-balanced round-robin.

    Node-level splitting cannot help a file whose weight sits in a single
    test; those are irreducible by any packing and need their own remedy.
    """
    table = _test_file_durations() if durations is None else durations
    by_file: dict[str, list[str]] = {}
    for node in nodes:
        by_file.setdefault(node.split("::", 1)[0], []).append(node)
    units: list[tuple[float, str, int, list[str]]] = []
    for path in sorted(by_file):
        group = sorted(
            by_file[path],
            key=lambda node: hashlib.sha256(node.encode()).digest(),
        )
        weight = table.get(path)
        # Spread a measured file's cost over its own nodes so LPT can split it
        # across shards; an unmeasured file falls back to one unit per node.
        per_node = 1.0 if weight is None else float(weight) / max(1, len(group))
        units.extend(
            (per_node, path, index, [node]) for index, node in enumerate(group)
        )
    # Longest-processing-time-first: the heaviest node picks the emptiest
    # shard, which is the standard greedy approximation for minimising the
    # makespan (worst shard).
    units.sort(key=lambda unit: (-unit[0], unit[1], unit[2]))
    batches: list[list[str]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for weight, _path, _index, group in units:
        target = min(range(workers), key=lambda shard: (loads[shard], shard))
        batches[target].extend(group)
        loads[target] += weight
    return batches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="check only staged paths")
    parser.add_argument(
        "--base-ref",
        help="check committed changes since this ref (for bounded CI validation)",
    )
    parser.add_argument(
        "--changed-tests-only",
        action="store_true",
        help="prefer explicitly changed test files; intended for latency-bounded hooks",
    )
    parser.add_argument(
        "--test-shard-index",
        type=int,
        default=0,
        help="zero-based changed-test CI shard index",
    )
    parser.add_argument(
        "--test-shard-count",
        type=int,
        default=1,
        help="number of disjoint changed-test CI shards",
    )
    parser.add_argument("--list", action="store_true", help="print selected tests without running")
    parser.add_argument(
        "--hook",
        action="store_true",
        help="emit an agent Stop-hook decision instead of regular output",
    )
    args = parser.parse_args(argv)
    if args.test_shard_count < 1:
        parser.error("--test-shard-count must be positive")
    if not 0 <= args.test_shard_index < args.test_shard_count:
        parser.error("--test-shard-index must be within --test-shard-count")
    paths = changed_files(staged=args.staged, base_ref=args.base_ref)
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
            print(
                json.dumps(
                    {"decision": "block", "reason": f"Changed tests failed:\n{detail}"}
                )
            )
        else:
            print(json.dumps({"decision": "allow"}))
        return 0
    return check(
        paths,
        changed_tests_only=args.changed_tests_only,
        staged=args.staged,
        test_shard_index=args.test_shard_index,
        test_shard_count=args.test_shard_count,
        base_ref=args.base_ref,
    )


if __name__ == "__main__":
    raise SystemExit(main())
