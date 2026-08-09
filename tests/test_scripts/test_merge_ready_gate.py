"""The local zero-minute merge gate mirrors the dormant python-static CI job.

Hosted GHA triggers are disabled (docs/design/ci-minutes-and-speed-plan-
20260806.md); `scripts.verify_merge_ready` is the authoritative local gate.
These tests certify (1) parity with `.github/workflows/ci.yml` by
construction, (2) the documented failure semantics, and (3) the pre-push /
reporter surfaces.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

from scripts.repo_policy import _workflow_jobs
from scripts.verify_merge_ready import (
    ROOT,
    Step,
    main,
    merge_gate_steps,
    run_gate,
)

CI_YML = ROOT / ".github/workflows/ci.yml"


def _python_static_text() -> str:
    text = CI_YML.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for name, start, end in _workflow_jobs(text):
        if name == "python-static":
            return "".join(lines[start:end])
    raise AssertionError("python-static job missing from ci.yml")


def test_gate_mirrors_every_python_static_module_in_order() -> None:
    """Every `python -m scripts.X` run by python-static appears in the gate,
    in the workflow's order — parity with the dormant CI job by construction."""
    mentions = re.findall(r"python -m (scripts\.\w+)", _python_static_text())
    assert mentions, "python-static job stopped invoking scripts modules"
    # The workflow mentions some modules twice (pull_request vs push branches
    # of the same step); parity is per distinct invocation, order-preserving.
    ci_modules = list(dict.fromkeys(mentions))
    gate_cmds = [" ".join(step.cmd) for step in merge_gate_steps(fast=True)]
    gate_modules = [
        match.group(1)
        for cmd in gate_cmds
        if (match := re.search(r"-m (scripts\.\w+)", cmd))
    ]
    # Order-preserving containment: the CI sequence embeds in the gate sequence.
    iterator = iter(gate_modules)
    missing = [module for module in ci_modules if module not in iterator]
    assert not missing, f"gate lost python-static steps (or reordered): {missing}"


def test_gate_mirrors_ruff_and_compileall() -> None:
    static_text = _python_static_text()
    assert "ruff check ." in static_text
    assert "python -m compileall -q src scripts tests" in static_text
    gate_cmds = [" ".join(step.cmd) for step in merge_gate_steps(fast=True)]
    assert any("ruff check ." in cmd for cmd in gate_cmds)
    assert any("compileall -q src scripts tests" in cmd for cmd in gate_cmds)


def test_fast_skips_only_the_changed_test_execution() -> None:
    fast = merge_gate_steps(fast=True)
    full = merge_gate_steps(fast=False)
    assert [step.name for step in full] == [step.name for step in fast] + [
        "changed_tests"
    ]
    assert all(step.static for step in fast)
    assert not full[-1].static
    assert "--changed-tests-only" in full[-1].cmd


def _step(name: str, code: str, *, static: bool = True) -> Step:
    return Step(name, (sys.executable, "-c", code), static=static)


def test_run_gate_collects_all_static_failures_then_skips_tests() -> None:
    steps = (
        _step("red", "import sys; print('boom'); sys.exit(1)"),
        _step("still-runs", "pass"),
        _step("tests", "raise SystemExit('never ran')", static=False),
    )
    summary = run_gate(steps, budget_seconds=60)
    by_name = {record["name"]: record for record in summary["steps"]}
    assert summary["ok"] is False
    assert by_name["red"]["status"] == "failed"
    assert "boom" in by_name["red"]["output_tail"]
    # Cheap static steps keep running after a failure (one pass, all findings).
    assert by_name["still-runs"]["status"] == "ok"
    # The expensive test step is the only one skipped.
    assert by_name["tests"]["status"] == "skipped"


def test_run_gate_green_path_runs_everything() -> None:
    steps = (
        _step("a", "pass"),
        _step("b", "pass", static=False),
    )
    summary = run_gate(steps, budget_seconds=60)
    assert summary["ok"] is True
    assert [record["status"] for record in summary["steps"]] == ["ok", "ok"]
    assert summary["rule"].startswith("red verify_merge_ready")


def test_run_gate_records_timeout_as_failure() -> None:
    steps = (_step("sleepy", "import time; time.sleep(30)"),)
    summary = run_gate(steps, budget_seconds=0.5)
    assert summary["ok"] is False
    assert summary["steps"][0]["status"] == "timeout"


def test_cli_json_summary_shape(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "scripts.verify_merge_ready.merge_gate_steps",
        lambda *, fast: (_step("stub", "pass"),),
    )
    assert main(["--json", "--fast"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True and summary["fast"] is True
    (record,) = summary["steps"]
    assert set(record) == {"name", "cmd", "status", "seconds"}
    assert record["status"] == "ok"


def test_cli_exits_nonzero_and_reports_rule_on_red(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "scripts.verify_merge_ready.merge_gate_steps",
        lambda *, fast: (_step("stub", "import sys; sys.exit(3)"),),
    )
    assert main(["--fast"]) == 1
    out = capsys.readouterr().out
    assert "merge-ready: BLOCKED (stub)" in out
    assert "GHA silence is not approval" in out


def test_cli_rejects_budget_above_canonical_cap() -> None:
    with pytest.raises(SystemExit):
        main(["--fast", "--max-step-seconds", "999999"])


def test_pre_push_hook_runs_fast_gate_with_loud_escape_hatch() -> None:
    hook = ROOT / ".githooks/pre-push"
    assert os.access(hook, os.X_OK), "pre-push must be executable"
    text = hook.read_text(encoding="utf-8")
    assert "scripts.verify_merge_ready --fast" in text
    assert "SLM_SKIP_PREPUSH" in text
    assert "UNVERIFIED" in text, "escape hatch must warn loudly"
    assert "GHA silence is not" in text


def test_reporter_is_a_read_only_json_wrapper() -> None:
    reporter = ROOT / "scripts/report_merge_readiness.sh"
    assert os.access(reporter, os.X_OK), "reporter must be executable"
    text = reporter.read_text(encoding="utf-8")
    assert "verify_merge_ready --json" in text
    assert "READ ONLY" in text
    # Never mutates: no redirections into tracked paths, no git write commands.
    assert not re.search(r"\bgit (add|commit|push|mv|rm)\b", text)


def test_hook_and_gate_paths_exist() -> None:
    for relative in (
        "scripts/verify_merge_ready.py",
        "scripts/report_merge_readiness.sh",
        ".githooks/pre-push",
        ".githooks/pre-commit",
    ):
        assert (Path(ROOT) / relative).is_file(), relative
