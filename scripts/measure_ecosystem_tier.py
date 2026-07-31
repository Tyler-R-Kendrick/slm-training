#!/usr/bin/env python3
"""Measure OpenUI ecosystem-library impact separate from production core.

Splits generation metrics and formal preflight success into
``production_core`` vs ``ecosystem_library`` tiers. Does not train, evaluate
models, or claim ship readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.dsl.ecosystem_tier import check_registry, write_registry
from slm_training.evals.ecosystem_tier_scoreboard import (
    build_ecosystem_tier_scoreboard,
    scoreboard_markdown,
    write_scoreboard,
)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return data


def _load_rows(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        return rows
    data = json.loads(text)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [row for row in data["rows"] if isinstance(row, dict)]
    raise SystemExit(f"expected list or {{rows: [...]}} in {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="JSON object of aggregated generation metrics",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=None,
        help="JSON/JSONL per-row generation metrics to aggregate",
    )
    parser.add_argument(
        "--formal-statuses",
        type=Path,
        default=None,
        help="JSON map of formal module → status (proved/unknown/...)",
    )
    parser.add_argument(
        "--probe-lean",
        action="store_true",
        help="Run a bounded lake build and mark formal modules proved on success",
    )
    parser.add_argument(
        "--previous-ecosystem-size",
        type=int,
        default=None,
        help="Prior ecosystem inventory size for growth annotation",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs/design/ecosystem-tier-scoreboard.json"),
        help="Scoreboard JSON path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/design/ecosystem-tier-scoreboard.md"),
        help="Scoreboard markdown path",
    )
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help="Rewrite the committed ecosystem tier classification registry",
    )
    parser.add_argument(
        "--check-registry",
        action="store_true",
        help="Fail if the committed registry drifts from live constants",
    )
    parser.add_argument(
        "--fixture-demo",
        action="store_true",
        help="Mark the scoreboard as fixture/demo (not ship)",
    )
    args = parser.parse_args(argv)

    if args.write_registry:
        payload = write_registry()
        print(f"wrote registry fingerprint={payload['fingerprint']}")
    if args.check_registry:
        check_registry()
        print("ecosystem tier registry: ok")

    scoreboard = build_ecosystem_tier_scoreboard(
        generation_metrics=_load_json(args.metrics),
        generation_rows=_load_rows(args.rows),
        formal_statuses=_load_json(args.formal_statuses),
        probe_lean=args.probe_lean,
        previous_ecosystem_library_size=args.previous_ecosystem_size,
        recipe={
            "fixture_demo": bool(args.fixture_demo),
            "probe_lean": bool(args.probe_lean),
            "metrics_path": str(args.metrics) if args.metrics else None,
            "rows_path": str(args.rows) if args.rows else None,
        },
    )
    write_scoreboard(args.out_json, scoreboard)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(scoreboard_markdown(scoreboard), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "core_formal": scoreboard["production_core"]["formal_preflight"][
                    "rollup"
                ],
                "ecosystem_formal": scoreboard["ecosystem_library"][
                    "formal_preflight"
                ]["rollup"],
                "ecosystem_library_size": scoreboard["separation"][
                    "ecosystem_library_size"
                ],
                "separation_holds": scoreboard["separation"]["separation_holds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
