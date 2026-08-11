"""CLI: run one frozen reasoning/revmath task (HARN-03 / SLM-536).

Owners/docs: docs/design/reverse-mathematics-computability.md +
.agents/skills/revmath. Lean/external checkers are optional experiments;
default --hermetic needs no external solver. Schemas: revmath_task/v1 only.

Hermetic example:
    PYTHONPATH=src uv run python -m scripts.run_revmath_task \\
      --task src/slm_training/resources/revmath/fixtures/hermetic_forward_theorem.task.json \\
      --hermetic --output-dir outputs/revmath/hermetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.reasoning.revmath.plugins import (
    RevmathCheckPlan,
    override_hermetic_mode,
    resolve_plugin,
)
from slm_training.harnesses.reasoning.revmath.report import build_revmath_report
from slm_training.harnesses.reasoning.revmath.runner import run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import parse_revmath_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Frozen RevmathTaskV1 JSON")
    parser.add_argument(
        "--lean-root",
        type=Path,
        default=None,
        help="Pinned Lean project root (ignored for --hermetic)",
    )
    parser.add_argument(
        "--hermetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use committed hermetic checker (default: true)",
    )
    parser.add_argument(
        "--hermetic-mode",
        default="ok",
        choices=("ok", "malformed", "incomplete", "refuted", "sleep"),
        help="Override hermetic checker mode for diagnostics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/revmath/runs"),
        help="Directory for result / replay / report JSON",
    )
    parser.add_argument(
        "--repair-history-ref",
        default=None,
        help="Optional immutable repair-history reference (HARN-09)",
    )
    parser.add_argument("--report-id", default=None)
    args = parser.parse_args(argv)

    raw = json.loads(args.task.read_text(encoding="utf-8"))
    task = parse_revmath_task(raw)

    plan_override = None
    if args.hermetic and args.hermetic_mode != "ok":
        plugin = resolve_plugin(task)
        if plugin is None:
            raise SystemExit(f"no plugin for task_kind={task.task_kind!r}")
        lean_root = args.lean_root or Path(".")
        plan = plugin.plan_check(task, lean_root=lean_root, hermetic=True)
        plan_override = RevmathCheckPlan(
            command=override_hermetic_mode(plan.command, args.hermetic_mode),
            cwd=plan.cwd,
            env=plan.env,
            requires_lean_tool=plan.requires_lean_tool,
            lean_project_root=plan.lean_project_root,
            checker_id=plan.checker_id,
        )

    record = run_revmath_task(
        task,
        lean_root=args.lean_root,
        hermetic=args.hermetic,
        repair_history_ref=args.repair_history_ref,
        plan_override=plan_override,
    )
    report = build_revmath_report(
        report_id=args.report_id or f"report.{task.task_id}",
        tasks=[task],
        results=[record.result],
        repair_ids=()
        if args.repair_history_ref is None
        else (args.repair_history_ref,),
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_record.json").write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "result.json").write_text(
        json.dumps(record.result.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (out / "replay_bundle.json").write_text(
        json.dumps(record.replay_bundle.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "result_id": record.result.result_id,
                "judgment": record.result.judgment,
                "task_identity_digest": task.identity_digest(),
                "replayable": record.replay_bundle.replayable,
                "bundle_hash": record.replay_bundle.bundle_hash,
                "output_dir": str(out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
