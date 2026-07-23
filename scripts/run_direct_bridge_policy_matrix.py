#!/usr/bin/env python3
"""Run SLM-197 matched direct legal-edit policy controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch

from slm_training.data.flow.bridge_corpus import load_corpus
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm197_direct_bridge_policy import (
    DEFAULT_CORPUS,
    DEFAULT_RECORDS,
    MATRIX_ARMS,
    _evaluate,
    _plan_hint,
    _schedule,
    _train_arm,
    run_matrix,
)
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_scorer import (
    DirectLegalEditPolicy,
    multi_positive_set_loss,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DESIGN_JSON = ROOT / "docs/design/iter-slm197-direct-bridge-policy-20260723.json"
DESIGN_MD = ROOT / "docs/design/iter-slm197-direct-bridge-policy-20260723.md"
DESIGN_AGENTV = ROOT / "docs/design/iter-slm197-direct-bridge-policy-agentv-20260723"


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "output-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    prefixes = {
        str(output_dir.resolve()): "output-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("output-dir://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for prefix, replacement in prefixes.items():
            content = content.replace(prefix, replacement)
        path.write_text(content, encoding="utf-8")


def _cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    measured = [
        run
        for arm in ("D2", "D3-linear", "D3-fourier", "D4", "D5")
        for run in report["arms"][arm]["runs"]
    ]
    return [
        {
            "id": "exact-membership",
            "criteria": "Every teacher-forced row uses the byte-identical SLM-196 exact candidate membership.",
            "pass": all(
                run["evaluation"]["candidate_membership"]["exact"] for run in measured
            ),
            "result": {"runs": len(measured)},
        },
        {
            "id": "unknown-mask",
            "criteria": "UNKNOWN candidates are never marked as explicit unsupported negatives.",
            "pass": not any(
                run["evaluation"]["candidate_membership"][
                    "unknown_as_explicit_negative"
                ]
                for run in measured
            ),
            "result": {"unknown_prevalence_retained": True},
        },
        {
            "id": "matched-capacity",
            "criteria": "All measured time/no-time/plan arms have identical parameter capacity.",
            "pass": report["matched_controls"]["parameter_count_equal"],
            "result": report["matched_controls"],
        },
        {
            "id": "free-running-legality",
            "criteria": "Free-running actions are selected only from newly enumerated live candidate sets.",
            "pass": all(
                run["free_running"]["all_actions_live_rate"] == 1.0
                for run in measured
            ),
            "result": {"runs": len(measured), "claim": "structural_contract"},
        },
        {
            "id": "honest-disposition",
            "criteria": "A non-publishable four-row fixture is reported as inconclusive and confirmation remains blocked.",
            "pass": (
                report["honest_verdict"] == "inconclusive_fixture_only"
                and report["confirmation"]["status"] == "blocked"
            ),
            "result": {
                "verdict": report["honest_verdict"],
                "confirmation": report["confirmation"],
            },
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SLM-197: direct legal-edit policy matrix",
        "",
        "**Status:** measured fixture wiring; powered experiment blocked.",
        "**Honest verdict:** `inconclusive_fixture_only`.",
        "**Ship claim:** none.",
        "",
        "## Decision",
        "",
        "The direct-policy contract, multi-positive set-mass objective, matched",
        "time encodings, checkpoint migration, and exact live-candidate decode are",
        "implemented and exercised. The requested powered D0-D5 comparison cannot",
        "run honestly because the committed SLM-196 corpus is non-publishable and",
        "contains only four rows across two targets; its X22/local-corruption controls",
        "are absent. D0 and D1 are therefore unavailable, not silently synthesized.",
        "",
        "## Matrix",
        "",
        "| Arm | Status | Seed runs | Dev top-1 positive range | Free-run target exact range |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for arm, description in MATRIX_ARMS.items():
        value = report["arms"][arm]
        if value["status"] == "unavailable":
            lines.append(f"| `{arm}` {description} | unavailable: {value['reason']} | 0 | — | — |")
            continue
        top1 = [
            run["evaluation"]["teacher_forced"]["top1_positive_rate"]
            for run in value["runs"]
        ]
        free = [
            run["free_running"]["target_exact_rate"] for run in value["runs"]
        ]
        lines.append(
            f"| `{arm}` {description} | fixture measured | {len(value['runs'])} | "
            f"{min(top1):.3f}–{max(top1):.3f} | {min(free):.3f}–{max(free):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Contract evidence",
            "",
            f"- Equal parameter capacity: `{report['matched_controls']['parameter_count_equal']}` "
            f"({report['matched_controls']['parameter_counts']}).",
            "- Time encodings use fixed-budget schedule progress; gold remaining distance",
            "  and bridge length never enter the scorer.",
            "- The likelihood is `logsumexp(all live) - logsumexp(certified positives)`.",
            "- UNKNOWN remains separately masked and is never used by an explicit negative loss.",
            "- Free-running decode re-enumerates exact candidates, verifies transition",
            "  certificates, replays the edit, and logs every state/action.",
            "- Plan conditioning is default-off; D4 alone receives the planner one-hot.",
            "",
            "## Recipe",
            "",
            f"- Device/backend: `{report['recipe']['device']}` / `{report['recipe']['backend']}`",
            f"- Steps/seeds: `{report['recipe']['steps']}` / `{report['recipe']['seeds']}`",
            f"- Train/dev rows: `{report['recipe']['train_rows']}` / `{report['recipe']['dev_rows']}`",
            f"- Independent targets: `{report['recipe']['independent_targets']}`",
            f"- Wall time: `{report['elapsed_seconds']:.3f}s` (cap `{report['recipe']['max_wall_minutes']}m`)",
            f"- AgentV: `{report['agentv']['summary']}`",
            "",
            "No checkpoint was written or promoted, so MODEL_CARD.md and the README",
            "checkpoint summary do not change.",
            "",
            "## Confirmation firewall",
            "",
        ]
    )
    lines.extend(f"- {reason}" for reason in report["confirmation"]["reasons"])
    lines.extend(
        [
            "",
            "`--confirm` fails closed until a publishable corpus and frozen confirmation",
            "manifest are supplied. This fixture result does not freeze a production baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_matrix(
    report: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    report["version_stamp"] = build_version_stamp(
        "harness.experiments.slm197_direct_bridge_policy"
    )
    DESIGN_AGENTV.mkdir(parents=True, exist_ok=True)
    report["agentv"] = _portable(
        publish_agentv_evaluation(
            DESIGN_AGENTV,
            name="slm197-direct-bridge-policy-fixture",
            claim="direct_bridge_policy_fixture_contract",
            cases=_cases(report),
        ),
        DESIGN_AGENTV,
    )
    _rewrite_agentv_paths(DESIGN_AGENTV)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    DESIGN_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    DESIGN_MD.write_text(_markdown(report), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--eval", action="store_true")
    mode.add_argument("--matrix", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--confirm", action="store_true")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/slm197"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=tuple(MATRIX_ARMS), default="D2")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-wall-minutes", type=float, default=2.8)
    parser.add_argument("--claim-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.describe:
        print(
            json.dumps(
                {
                    "schema": "DirectBridgePolicyMatrixDescriptionV1",
                    "arms": MATRIX_ARMS,
                    "modes": ["describe", "train", "eval", "matrix", "resume", "confirm"],
                    "hard_cap_minutes": 3,
                    "confirmation": "fail_closed",
                },
                indent=2,
            )
        )
        return 0
    if not 0 < args.max_wall_minutes <= 3:
        parser.error("--max-wall-minutes must be in (0, 3]")
    rows, candidate_sets, manifest = load_corpus(args.corpus_dir)
    if args.confirm:
        if not manifest.get("publishable") or args.claim_manifest is None:
            parser.error(
                "--confirm requires a publishable corpus and frozen --claim-manifest"
            )
        parser.error("confirmation execution requires the powered confirmation suite")
    if args.matrix:
        seeds = tuple(int(value) for value in args.seeds.split(",") if value)
        _write_matrix(
            run_matrix(
                corpus_dir=args.corpus_dir,
                records_path=args.records,
                seeds=seeds,
                steps=args.steps,
                learning_rate=args.learning_rate,
                max_wall_minutes=args.max_wall_minutes,
            ),
            args.output_dir,
        )
        return 0
    if args.checkpoint is None:
        parser.error("--checkpoint is required for train/eval/resume")
    train_rows = [row for row in rows if row.split == "train"]
    dev_rows = [row for row in rows if row.split == "dev"]
    if args.train:
        policy, training = _train_arm(
            args.arm,
            train_rows,
            candidate_sets,
            seed=0,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        policy.save(
            args.checkpoint,
            metadata={
                "arm": args.arm,
                "corpus_content_fingerprint": manifest["content_fingerprint"],
                "termination": {"name": "fixed_k", "k": 2, "max_steps": 2},
                "training": training,
            },
        )
        print(args.checkpoint)
        return 0
    policy = DirectLegalEditPolicy.from_checkpoint(args.checkpoint)
    if args.resume:
        batch = LegalEditBatch.pack(train_rows, candidate_sets)
        optimizer = torch.optim.Adam(policy.scorer.parameters(), lr=args.learning_rate)
        plan = _plan_hint(batch, train_rows) if args.arm == "D4" else None
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = multi_positive_set_loss(
                policy.scorer(
                    batch, schedule_progress=_schedule(train_rows), plan_hint=plan
                ),
                batch,
            )
            loss.backward()
            optimizer.step()
        policy.save(args.checkpoint, metadata={"resumed_steps": args.steps})
        print(args.checkpoint)
        return 0
    result = _evaluate(policy, dev_rows, candidate_sets, arm=args.arm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "eval.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
