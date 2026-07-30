"""Validate or deterministically refresh external pytest expectations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tests.casefiles import (  # noqa: E402
    CASE_ROOT,
    REPO_ROOT,
    all_case_files,
    load_case_document,
    resource_path_for_test,
    test_path_for_resource,
)

REVIEW_SCHEMA = "case_resource_review/v1"
GATE_POLICY = REPO_ROOT / "src/slm_training/resources/evals/openui_ship_gates_v5.json"
EVAL_RESOURCE_ROOT = REPO_ROOT / "src/slm_training/resources/evals"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _changed_paths(base_ref: str | None) -> list[Path]:
    diff = (
        _git("diff", "--name-only", "--diff-filter=ACMRD", f"{base_ref}...HEAD", "--")
        if base_ref
        else _git("diff", "HEAD", "--name-only", "--diff-filter=ACMRD", "--")
    )
    paths = set(diff.splitlines())
    if not base_ref:
        paths.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    prefix = CASE_ROOT.relative_to(REPO_ROOT).as_posix() + "/"
    return sorted(REPO_ROOT / path for path in paths if path.startswith(prefix))


def _resolve_targets(
    raw_targets: list[str], *, changed: bool, base_ref: str | None
) -> list[Path]:
    if changed:
        return _changed_paths(base_ref)
    if not raw_targets:
        return all_case_files()
    resources: set[Path] = set()
    for raw in raw_targets:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if path.suffix == ".py":
            path = resource_path_for_test(path)
        if path.is_dir():
            resources.update(path.rglob("*.json"))
        else:
            resources.add(path)
    return sorted(resources)


def _case_rows(document: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (set_name, row["id"]): row
        for set_name, case_set in document["case_sets"].items()
        for row in case_set["cases"]
    }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _review(
    path: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    before_text: str,
    after_text: str,
) -> dict[str, Any]:
    old_rows = _case_rows(before)
    new_rows = _case_rows(after)
    removed = sorted(
        f"{group}/{case_id}" for group, case_id in old_rows.keys() - new_rows
    )
    if removed:
        raise ValueError(f"{path}: case removal or rename is forbidden: {removed}")
    changed_expected: list[str] = []
    changed_input: list[str] = []
    changed_invariant: list[str] = []
    added = sorted(
        f"{group}/{case_id}" for group, case_id in new_rows.keys() - old_rows
    )
    for key in old_rows.keys() & new_rows.keys():
        label = f"{key[0]}/{key[1]}"
        old, new = old_rows[key], new_rows[key]
        if old.get("expected") != new.get("expected"):
            changed_expected.append(label)
        if old.get("input", old.get("values")) != new.get("input", new.get("values")):
            changed_input.append(label)
        if old.get("invariant") != new.get("invariant"):
            changed_invariant.append(label)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "before_sha256": _sha(before_text),
        "after_sha256": _sha(after_text),
        "before_count": len(old_rows),
        "after_count": len(new_rows),
        "added": added,
        "changed_input": sorted(changed_input),
        "changed_expected": sorted(changed_expected),
        "changed_invariant": sorted(changed_invariant),
        "removed": removed,
    }


def _without_expected(document: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(document)
    for case_set in value["case_sets"].values():
        for row in case_set["cases"]:
            row.pop("expected", None)
    return value


def _run_pytest(test_paths: list[Path], *, refresh: bool) -> None:
    env = os.environ.copy()
    if refresh:
        env["SLM_REFRESH_TEST_CASES"] = "1"
    else:
        env.pop("SLM_REFRESH_TEST_CASES", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *(path.relative_to(REPO_ROOT).as_posix() for path in test_paths),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"pytest failed with exit code {result.returncode}")


def _base_document(base_ref: str, path: Path) -> tuple[dict[str, Any], str] | None:
    relative = path.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{relative}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"{relative} at {base_ref} is not a JSON object")
    return value, result.stdout


def _gate_weakenings(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if after.get("default_min_suite_n", 0) < before.get("default_min_suite_n", 0):
        failures.append("default_min_suite_n decreased")
    old_suites = before.get("suites") or {}
    new_suites = after.get("suites") or {}
    for suite, old_metrics in old_suites.items():
        if suite not in new_suites:
            failures.append(f"removed suite {suite}")
            continue
        for metric, old_threshold in old_metrics.items():
            if metric not in new_suites[suite]:
                failures.append(f"removed threshold {suite}.{metric}")
            elif new_suites[suite][metric] < old_threshold:
                failures.append(f"lowered threshold {suite}.{metric}")
    return failures


def _policy_and_eval_checks(base_ref: str) -> dict[str, Any]:
    from slm_training.harnesses.model_build.ship_gates import (
        load_ship_gate_policy,
    )

    current = load_ship_gate_policy(GATE_POLICY)
    policy_review: dict[str, Any] = {
        "path": GATE_POLICY.relative_to(REPO_ROOT).as_posix()
    }
    if base := _base_document(base_ref, GATE_POLICY):
        before, _ = base
        weakenings = _gate_weakenings(before, current)
        if weakenings:
            raise ValueError("ship-gate policy weakened: " + "; ".join(weakenings))
        policy_review["weakenings"] = []

    diff = _git(
        "diff",
        "--name-only",
        "--diff-filter=ACMRD",
        base_ref if base_ref == "HEAD" else f"{base_ref}...HEAD",
        "--",
        EVAL_RESOURCE_ROOT.relative_to(REPO_ROOT).as_posix(),
    )
    changed = [REPO_ROOT / relative for relative in diff.splitlines()]
    frozen_changed = [
        path
        for path in changed
        if path != GATE_POLICY and _base_document(base_ref, path) is not None
    ]
    if frozen_changed:
        names = [path.relative_to(REPO_ROOT).as_posix() for path in frozen_changed]
        raise ValueError(
            "frozen eval resources changed in place; add a new version: "
            + ", ".join(names)
        )
    return policy_review


def check(resources: list[Path], *, base_ref: str | None) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    for path in resources:
        if not path.is_file():
            raise ValueError(f"case resource is missing: {path}")
        current_text = path.read_text(encoding="utf-8")
        current = load_case_document(path)
        if base_ref and (base := _base_document(base_ref, path)):
            before, before_text = base
            reviews.append(
                _review(
                    path,
                    before,
                    current,
                    before_text=before_text,
                    after_text=current_text,
                )
            )
    return {
        "schema": REVIEW_SCHEMA,
        "mode": "check",
        "files": reviews,
        "gate_policy": _policy_and_eval_checks(base_ref or "HEAD"),
    }


def refresh(resources: list[Path]) -> dict[str, Any]:
    if not resources:
        return {"schema": REVIEW_SCHEMA, "mode": "refresh", "files": []}
    before_text = {path: path.read_text(encoding="utf-8") for path in resources}
    before = {path: load_case_document(path) for path in resources}
    tests = sorted({test_path_for_resource(path) for path in resources})
    missing_tests = [path for path in tests if not path.is_file()]
    if missing_tests:
        raise ValueError(f"case resources without test modules: {missing_tests}")

    _run_pytest(tests, refresh=True)
    after_first_text = {path: path.read_text(encoding="utf-8") for path in resources}
    after_first = {path: load_case_document(path) for path in resources}
    for path in resources:
        if _without_expected(before[path]) != _without_expected(after_first[path]):
            raise ValueError(f"{path}: refresh changed input, invariant, id, or schema")

    _run_pytest(tests, refresh=True)
    after_second_text = {path: path.read_text(encoding="utf-8") for path in resources}
    if after_first_text != after_second_text:
        raise ValueError("expectation refresh is non-deterministic")
    _run_pytest(tests, refresh=False)

    return {
        "schema": REVIEW_SCHEMA,
        "mode": "refresh",
        "files": [
            _review(
                path,
                before[path],
                after_first[path],
                before_text=before_text[path],
                after_text=after_first_text[path],
            )
            for path in resources
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="test modules or case resources")
    parser.add_argument("--check", action="store_true", help="validate without writing")
    parser.add_argument(
        "--changed", action="store_true", help="select changed resources"
    )
    parser.add_argument("--base-ref", help="compare stable case IDs with this git ref")
    parser.add_argument("--review-out", type=Path, help="also write the JSON review")
    args = parser.parse_args(argv)
    try:
        resources = _resolve_targets(
            args.targets, changed=args.changed, base_ref=args.base_ref
        )
        review = (
            check(resources, base_ref=args.base_ref)
            if args.check
            else refresh(resources)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"case-resources: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(review, indent=2) + "\n"
    print(rendered, end="")
    if args.review_out:
        args.review_out.parent.mkdir(parents=True, exist_ok=True)
        args.review_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
