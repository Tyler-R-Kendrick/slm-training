#!/usr/bin/env python3
"""Extract SemanticPlanV1 gold plans from OpenUI records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.semantic_plan import (
    OpenUISemanticPlanExtractor,
    plan_factor_fingerprints,
)
from slm_training.dsl.pack import get_pack


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="Input records JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Output plans JSONL")
    parser.add_argument("--coverage", type=Path, help="Optional coverage report JSON")
    parser.add_argument("--dsl", default="openui", help="DSL pack id")
    args = parser.parse_args(argv)

    pack = get_pack(args.dsl)
    extractor = OpenUISemanticPlanExtractor()
    rows = _jsonl(args.records)

    plans: list[dict[str, Any]] = []
    coverage = {
        "n": 0,
        "archetype_ids": set(),
        "role_families": set(),
        "symbols": 0,
        "bindings": 0,
        "unknown_reasons": [],
    }

    for row in rows:
        # Records are ExampleRecord-shaped; build a minimal ProgramSpec.
        spec = ProgramSpec.from_openui(
            id=str(row.get("id") or row.get("record_id") or "unknown"),
            openui=str(row.get("openui") or row.get("canonical_openui") or ""),
            facts={},
            program_family_id=str(row.get("program_family_id") or "openui"),
            lineage_id=str(row.get("lineage_id") or "extract"),
            split_group_id=str(row.get("split_group_id") or "extract"),
            split=str(row.get("split") or "train"),
        )
        plan = extractor.extract(spec, pack)
        fingerprints = plan_factor_fingerprints(plan)
        payload = {
            "schema": "SemanticPlanV1",
            "record_id": spec.id,
            "plan": plan.to_dict(),
            "fingerprints": fingerprints,
        }
        plans.append(payload)

        coverage["n"] += 1
        if plan.archetype.id:
            coverage["archetype_ids"].add(plan.archetype.id)
        for slot in plan.role_slots:
            if slot.component_family:
                coverage["role_families"].add(slot.component_family)
        coverage["symbols"] += len(plan.symbols)
        coverage["bindings"] += len(plan.bindings)
        if plan.confidence_calibration.abstention_reason:
            coverage["unknown_reasons"].append(plan.confidence_calibration.abstention_reason)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for payload in plans:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    if args.coverage:
        coverage["archetype_ids"] = sorted(coverage["archetype_ids"])
        coverage["role_families"] = sorted(coverage["role_families"])
        args.coverage.parent.mkdir(parents=True, exist_ok=True)
        args.coverage.write_text(
            json.dumps(coverage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"plans": len(plans), "coverage": str(args.coverage)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
