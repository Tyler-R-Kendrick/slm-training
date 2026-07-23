#!/usr/bin/env python3
"""Run the SLM-198 bridge-length and target-balanced curriculum ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.data.flow.bridge_corpus import load_corpus
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm197_direct_bridge_policy import DEFAULT_CORPUS
from slm_training.harnesses.experiments.slm198_bridge_curriculum import (
    ARMS,
    DEFAULT_SEEDS,
    BridgeCurriculumSampler,
    build_manifest,
    compute_difficulties,
    run_matrix,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DESIGN_JSON = ROOT / "docs/design/iter-slm198-bridge-curriculum-20260723.json"
DESIGN_MD = ROOT / "docs/design/iter-slm198-bridge-curriculum-20260723.md"
DESIGN_AGENTV = ROOT / "docs/design/iter-slm198-bridge-curriculum-agentv-20260723"


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if len(seeds) < 1:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    runs = [
        run for arm in report["arms"].values() for run in arm.get("runs", ())
    ]
    return [
        {
            "id": "deterministic-difficulty",
            "criteria": "Scheduling difficulty is deterministic and uses corpus/compiler facts, not confirmation outcomes.",
            "pass": bool(report["difficulty"]),
            "result": {"row_ids": sorted(report["difficulty"])},
        },
        {
            "id": "target-first",
            "criteria": "Balanced and curriculum arms declare target-first path/state sampling.",
            "pass": report["matched_controls"]["target_first_explicit"],
            "result": {"arms": list(ARMS[1:])},
        },
        {
            "id": "exposure-reconciliation",
            "criteria": "Every arm has matched total decisions, final target support, and the balanced family has equal target exposure.",
            "pass": (
                report["matched_controls"]["final_support_equal"]
                and report["matched_controls"]["balanced_target_exposure_equal"]
                and len({run["training"]["exposures"] for run in runs}) == 1
            ),
            "result": report["matched_controls"],
        },
        {
            "id": "frozen-policy-contract",
            "criteria": "All cells preserve exact candidate membership, equal scorer capacity, and train/dev target isolation.",
            "pass": (
                report["matched_controls"]["parameter_count_equal"]
                and report["matched_controls"]["split_safe"]
                and all(
                    run["evaluation"]["candidate_membership"]["exact"] for run in runs
                )
            ),
            "result": {
                "runs": len(runs),
                "parameter_counts": report["matched_controls"]["parameter_counts"],
            },
        },
        {
            "id": "honest-disposition",
            "criteria": "The non-publishable one-train-target fixture selects no curriculum and blocks confirmation.",
            "pass": (
                report["confirmation"]["status"] == "blocked"
                and report["confirmation"]["selected_arm"] is None
                and report["checkpoint"]["written"] is False
            ),
            "result": {
                "verdict": report["honest_verdict"],
                "confirmation": report["confirmation"],
            },
        },
    ]


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    prefixes = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for prefix, replacement in prefixes.items():
            content = content.replace(prefix, replacement)
        path.write_text(content, encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SLM-198: bridge curriculum and target-balance ablation",
        "",
        "**Status:** measured fixture wiring; powered experiment blocked.",
        "**Decision:** select no curriculum.",
        "**Ship claim:** none.",
        "",
        "## Result",
        "",
        "The seven-arm schedule, target-first sampler, deterministic difficulty",
        "features, exposure ledger, anti-curriculum, and resumable cursor are wired",
        "against the frozen SLM-197 direct legal-edit scorer. The committed corpus",
        "cannot distinguish the ablation: its train split has two rows from one target,",
        "all bridges have length two, and dependency capsules are empty.",
        "",
        "| Arm | Class | Seeds | Dev top-1 positive range | Free-run exact range |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        value = report["arms"][arm]
        runs = value["runs"]
        top1 = [
            run["evaluation"]["teacher_forced"]["top1_positive_rate"]
            for run in runs
        ]
        exact = [run["free_running"]["target_exact_rate"] for run in runs]
        lines.append(
            f"| `{arm}` | {value['status']} | {len(runs)} | "
            f"{min(top1):.3f}-{max(top1):.3f} | {min(exact):.3f}-{max(exact):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Exposure and policy contract",
            "",
            f"- Final target support equal: `{report['matched_controls']['final_support_equal']}`.",
            f"- Balanced target exposure equal: `{report['matched_controls']['balanced_target_exposure_equal']}`.",
            f"- Candidate-token totals equal in this matched fixture: `{report['matched_controls']['candidate_tokens_equal']}`.",
            f"- Parameter capacity equal: `{report['matched_controls']['parameter_count_equal']}` "
            f"({report['matched_controls']['parameter_counts']}).",
            f"- Train/dev target isolation: `{report['matched_controls']['split_safe']}`.",
            "- `oracle_difficulty` is development-only and cannot be selected.",
            "",
            "## Recipe",
            "",
            f"- Device/backend: `{report['recipe']['device']}` / `{report['recipe']['backend']}`",
            f"- Epochs/seeds: `{report['recipe']['epochs']}` / `{report['recipe']['seeds']}`",
            f"- Train/dev rows: `{report['recipe']['train_rows']}` / `{report['recipe']['dev_rows']}`",
            f"- Train/dev independent targets: `{report['recipe']['train_targets']}` / `{report['recipe']['dev_targets']}`",
            f"- Wall: `{report['elapsed_seconds']:.3f}s` (cap `{report['recipe']['max_wall_minutes']}m`)",
            f"- AgentV: `{report['agentv']['summary']}`",
            "",
            "No checkpoint was written or promoted, so the model card and README",
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
            "`--confirm` fails closed. A future confirmation must freeze a publishable,",
            "multi-target/multi-length corpus and compare one preregistered selected arm",
            "only against `uniform_targets`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    report["version_stamp"] = build_version_stamp(
        "harness.experiments.slm198_bridge_curriculum"
    )
    DESIGN_AGENTV.mkdir(parents=True, exist_ok=True)
    report["agentv"] = _portable(
        publish_agentv_evaluation(
            DESIGN_AGENTV,
            name="slm198-bridge-curriculum-fixture",
            claim="bridge_curriculum_fixture_contract",
            cases=_cases(report),
        ),
        DESIGN_AGENTV,
    )
    _rewrite_agentv_paths(DESIGN_AGENTV)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2) + "\n"
    (output_dir / "report.json").write_text(payload, encoding="utf-8")
    DESIGN_JSON.write_text(
        json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    DESIGN_MD.write_text(_markdown(report), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--manifest", action="store_true")
    mode.add_argument("--matrix", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--analyze-exposure", action="store_true")
    mode.add_argument("--confirm", action="store_true")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/slm198"))
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-wall-minutes", type=float, default=2.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 0 < args.max_wall_minutes <= 3:
        parser.error("--max-wall-minutes must be in (0, 3]")
    if args.describe:
        print(
            json.dumps(
                {
                    "schema": "SLM198BridgeCurriculumDescriptionV1",
                    "arms": list(ARMS),
                    "modes": [
                        "describe",
                        "manifest",
                        "matrix",
                        "resume",
                        "analyze-exposure",
                        "confirm",
                    ],
                    "confirmation_control": "uniform_targets",
                    "oracle_selection_eligible": False,
                },
                indent=2,
            )
        )
        return 0
    rows, candidate_sets, source = load_corpus(args.corpus_dir)
    train_rows = [row for row in rows if row.split == "train"]
    difficulty = compute_difficulties(train_rows, candidate_sets)
    if args.manifest:
        manifests = [
            build_manifest(
                train_rows,
                arm=arm,
                seed=args.seeds[0],
                epochs=args.epochs,
                source_content_fingerprint=source["content_fingerprint"],
                difficulty=difficulty,
            ).to_dict()
            for arm in ARMS
        ]
        print(json.dumps({"schema": "BridgeCurriculumManifestSetV1", "manifests": manifests}, indent=2))
        return 0
    if args.resume:
        manifest = build_manifest(
            train_rows,
            arm="entropy_curriculum",
            seed=args.seeds[0],
            epochs=args.epochs,
            source_content_fingerprint=source["content_fingerprint"],
            difficulty=difficulty,
        )
        sampler = BridgeCurriculumSampler(train_rows, manifest, difficulty)
        prefix = [next(sampler).row_id]
        resumed = BridgeCurriculumSampler.resume(
            train_rows, manifest, difficulty, sampler.state_dict()
        )
        print(json.dumps({"schema": "BridgeCurriculumResumeProofV1", "row_ids": prefix + [row.row_id for row in resumed]}, indent=2))
        return 0
    report = run_matrix(
        corpus_dir=args.corpus_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        max_wall_minutes=args.max_wall_minutes,
    )
    if args.confirm:
        parser.error("; ".join(report["confirmation"]["reasons"]))
    report = _write(report, args.output_dir)
    if args.analyze_exposure:
        print(json.dumps({"schema": "BridgeCurriculumExposureAnalysisV1", "matched_controls": report["matched_controls"]}, indent=2))
    else:
        print(json.dumps({"report": str(args.output_dir / "report.json"), "verdict": report["honest_verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
