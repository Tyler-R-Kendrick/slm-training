#!/usr/bin/env python3
"""Run the SDE0-01 frozen E396/E479 decode-scaffolding factorial.

Example::

    python -m scripts.ablate_decode_scaffolding \
      --checkpoint outputs/runs/e396-balanced-type-head-continuation-r1/checkpoints/last.pt \
      --checkpoint-id e396-balanced-type-head-continuation-r1 \
      --checkpoint-sha256 feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0 \
      --suites rico_held,rico_dev,bench_small \
      --output-codec choice \
      --out-dir outputs/runs/sde0-01

Dry-run (no checkpoint required)::

    python -m scripts.ablate_decode_scaffolding --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.eval.ablate_decode_scaffolding import (
    AblateReport,
    ModelBuildConfig,
    build_stage_a_arms,
    run_stage_a,
    stage_a_needs_stage_b,
)


def _markdown_report(report: AblateReport) -> str:
    lines = [
        "# SDE0-01 decode-scaffolding × prompt-inventory factorial",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Checkpoint:* `{report.checkpoint_id}`  ",
        f"*Stage:* {report.stage}  ",
        f"*Timestamp:* {report.timestamp}  ",
        "",
        "## Honest caveat",
        "",
        "This is an eval-only ablation over a frozen checkpoint.  It decomposes",
        "the contribution of decode-time scaffolding from learned weights; it does",
        "not train a new model or make a ship-quality claim without full provenance.",
        "",
        "## Stage A arms",
        "",
        "| arm | content_floor | prompt_inventory | semantic_constraints | attempts | decode_path | best_of_n | compatible | notes |",
        "| --- | ------------- | ---------------- | -------------------- | -------- | ----------- | --------- | ---------- | ----- |",
    ]
    for arm in report.arms:
        factors = arm.factors.to_dict()
        notes = " ".join(arm.notes)
        lines.append(
            f"| {arm.arm_id} | {factors['content_floor']} | "
            f"{factors['prompt_inventory']} | {factors['semantic_constraints']} | "
            f"{factors['attempts']} | {arm.decode_path_id} | {arm.best_of_n} | "
            f"{arm.compatible} | {notes} |"
        )
    lines.append("")
    lines.append("## Hard gates")
    lines.append("")
    compatible = [a for a in report.arms if a.compatible]
    lines.append(f"- Compatible arms: {len(compatible)}/{len(report.arms)}")
    if compatible:
        lines.append("- **PASS** every arm resolved a legal config override set.")
    else:
        lines.append("- **FAIL** no compatible arms; check decode-path compatibility.")
    need_stage_b = stage_a_needs_stage_b(report.arms)
    lines.append(f"- Stage B recommended: {need_stage_b}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a frozen TwoTowerModel checkpoint.  Omit for dry-run.",
    )
    parser.add_argument(
        "--checkpoint-id",
        default="unknown",
        help="Stable checkpoint identifier.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default=None,
        help="SHA-256 of the checkpoint file for provenance.",
    )
    parser.add_argument(
        "--checkpoint-remote-uri",
        default=None,
        help="Remote URI (e.g. hf://bucket/path) for provenance.",
    )
    parser.add_argument(
        "--output-codec",
        default="choice",
        choices=["choice", "lexer", "compositional"],
        help="Output tokenizer identity of the checkpoint.",
    )
    parser.add_argument(
        "--suites",
        default="",
        help="Comma-separated suite names.  Empty means fixture mode.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/sde0-01"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe Stage A arms without running eval.",
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/runs/sde0-01/scratch",
        help="Run directory for ModelBuildConfig.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for evaluation.",
    )
    args = parser.parse_args(argv)

    suites = tuple(x.strip() for x in args.suites.split(",") if x.strip())

    if args.dry_run:
        arms = build_stage_a_arms()
        print(f"SDE0-01 Stage A dry-run: {len(arms)} arm(s)")
        for arm in arms:
            print(
                f"  {arm.arm_id}: factors={arm.factors.to_dict()} "
                f"decode_path={arm.decode_path_id} best_of_n={arm.best_of_n}"
            )
        return 0

    base_config = ModelBuildConfig(
        train_dir=Path("outputs/data/train/v1"),
        run_root=Path(args.run_dir),
        run_id="sde0-01-base",
        device=args.device,
        output_tokenizer=args.output_codec,
    )

    report = run_stage_a(
        base_config,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_remote_uri=args.checkpoint_remote_uri,
        checkpoint_path=args.checkpoint,
        output_codec=args.output_codec,
        suites=suites,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"sde0_01_{report.run_id}.json"
    json_path.write_text(report.to_json(indent=2), encoding="utf-8")
    md_path = out_dir / f"sde0_01_{report.run_id}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")

    compatible = sum(1 for a in report.arms if a.compatible)
    print(f"Wrote {json_path} and {md_path}")
    print(f"Compatible arms: {compatible}/{len(report.arms)}")
    print(f"Stage B recommended: {stage_a_needs_stage_b(report.arms)}")
    return 0 if compatible == len(report.arms) else 1


if __name__ == "__main__":
    raise SystemExit(main())
