#!/usr/bin/env python3
"""Run the SLM-144 SPV1-01 archetype + role-set predictor fixture matrix.

Example:
  python -m scripts.run_slm144_plan_predictor_fixture --mode fixture
  python -m scripts.run_slm144_plan_predictor_fixture --mode plan-only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.harnesses.experiments.slm144_plan_predictor_matrix import (
    MATRIX_SET,
    MATRIX_VERSION,
    Slm144Arm,
    Slm144Manifest,
    Slm144Report,
    Slm144Row,
    build_slm144_manifest,
    render_markdown,
    run_fixture_matrix,
)
from slm_training.models.semantic_plan_predictor import (
    PlanTrainingExample,
    build_role_set_target,
    featurize_program_spec,
)
from slm_training.versioning import build_version_stamp

__all__ = ["main"]

_DESIGN_JSON = "docs/design/iter-slm144-plan-predictor-20260720.json"
_DESIGN_MD = "docs/design/iter-slm144-plan-predictor-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _examples_from_corpus(
    corpus_split: list[tuple[Any, Any]],
    family_vocab: dict[str, int],
    role_vocab: dict[str, int],
    archetype_vocab: dict[str, int],
    max_len: int,
) -> list[PlanTrainingExample]:
    examples: list[PlanTrainingExample] = []
    for spec, plan in corpus_split:
        if plan.archetype.id is None:
            continue
        archetype_label = archetype_vocab[plan.archetype.id]
        role_ids = [slot.role_id for slot in plan.role_slots]
        features = featurize_program_spec(spec, family_vocab)
        role_mask = build_role_set_target(role_ids, role_vocab, len(role_vocab))
        sorted_roles = sorted(
            {role_vocab[r] for r in role_ids if r in role_vocab},
            key=lambda i: i,
        )
        padded = (sorted_roles + [-1] * max_len)[:max_len]
        serialized = torch.tensor(padded, dtype=torch.long)
        examples.append(
            PlanTrainingExample(
                example_id=spec.id,
                input_features=features,
                archetype_label=archetype_label,
                role_set_mask=role_mask,
                serialized_roles=serialized,
                source_plan=plan,
                program_spec=spec,
            )
        )
    return examples


def _build_vocabs(
    corpus: dict[str, list[tuple[Any, Any]]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
    families: set[str] = set()
    roles: set[str] = set()
    archetypes: set[str] = set()
    max_roles = 0
    for split in corpus.values():
        for _spec, plan in split:
            if plan.archetype.id:
                archetypes.add(plan.archetype.id)
            role_ids = [slot.role_id for slot in plan.role_slots]
            roles.update(role_ids)
            max_roles = max(max_roles, len(role_ids))
            for slot in plan.role_slots:
                if slot.component_family:
                    families.add(slot.component_family)
    family_vocab = {f: i for i, f in enumerate(sorted(families))}
    role_vocab = {r: i for i, r in enumerate(sorted(roles))}
    archetype_vocab = {a: i for i, a in enumerate(sorted(archetypes))}
    return family_vocab, role_vocab, archetype_vocab, max_roles


def _build_manifest_report(
    mode: str,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    corpus = build_fixture_plan_corpus(count=64, seed=0)
    family_vocab, role_vocab, archetype_vocab, max_roles = _build_vocabs(corpus)
    manifest = build_slm144_manifest()

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm144PlanPredictorManifestV1",
            "matrix_set": manifest.matrix_set,
            "matrix_version": manifest.matrix_version,
            "status": "plan_only",
            "claim_class": "wiring",
            "corpus": {
                "train": len(corpus["train"]),
                "val": len(corpus["val"]),
                "families": len(family_vocab),
                "roles": len(role_vocab),
                "archetypes": len(archetype_vocab),
            },
            "manifest": manifest.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm144_plan_predictor",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_slm144_plan_predictor_fixture --mode plan-only"
        return payload, command

    max_len = max(1, max_roles)
    train_examples = _examples_from_corpus(
        corpus["train"], family_vocab, role_vocab, archetype_vocab, max_len
    )
    val_examples = _examples_from_corpus(
        corpus["val"], family_vocab, role_vocab, archetype_vocab, max_len
    )

    report = run_fixture_matrix(
        train_examples,
        val_examples,
        run_id="slm144_fixture",
        output_dir=output_dir,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm144_plan_predictor_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        manifest_data = payload["manifest"]
        manifest = Slm144Manifest(
            matrix_set=manifest_data.get("matrix_set", MATRIX_SET),
            matrix_version=manifest_data.get("matrix_version", MATRIX_VERSION),
            hypothesis=manifest_data.get("hypothesis", ""),
            arms=[Slm144Arm(**arm) for arm in manifest_data.get("arms", [])],
        )
        lines = [
            "# SLM-144 / SPV1-01: Archetype + role-set predictor plan",
            "",
            "**Claim class:** wiring / fixture only  ",
            "**Run date:** 2026-07-20  ",
            "**Machine-readable result:** [`iter-slm144-plan-predictor-20260720.json`](iter-slm144-plan-predictor-20260720.json)",
            "",
            "This is a plan-only manifest. The fixture corpus and arm definitions "
            "are wired; run `--mode fixture` to execute the CPU train/eval matrix.",
            "",
            "## Manifest",
            "",
            f"Hypothesis: {manifest.hypothesis}",
            "",
            "| Arm | Archetype | Role set |",
            "| --- | --- | --- |",
        ]
        for arm in manifest.arms:
            lines.append(
                f"| {arm.arm_id} | {arm.archetype_source} | {arm.role_set_source} |"
            )
        lines.extend(["", "## Exact command", "", f"```bash\n{command}\n```", ""])
        return "\n".join(lines)

    manifest_data = payload["manifest"]
    report = Slm144Report(
        matrix_set=payload["matrix_set"],
        matrix_version=payload["matrix_version"],
        run_id=payload["run_id"],
        status=payload["status"],
        manifest=Slm144Manifest(
            matrix_set=manifest_data.get("matrix_set", MATRIX_SET),
            matrix_version=manifest_data.get("matrix_version", MATRIX_VERSION),
            hypothesis=manifest_data.get("hypothesis", ""),
            arms=[Slm144Arm(**arm) for arm in manifest_data.get("arms", [])],
        ),
        rows=[Slm144Row(**row) for row in payload["rows"]],
        version_stamp=payload.get("version_stamp", {}),
    )
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-144 SPV1-01 plan-predictor fixture matrix",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture trains and evaluates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm144-fixture-<YYYYMMDD>)",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm144-fixture-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_manifest_report(args.mode, output_dir)
    payload["schema"] = "Slm144PlanPredictorReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm144_plan_predictor_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    json_path = root / _DESIGN_JSON
    md_path = root / _DESIGN_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report_text, encoding="utf-8")

    command_line = command
    if args.output_dir is not None:
        command_line += f" --output-dir {output_dir}"
    md_path.write_text(_build_markdown(payload, command_line), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
