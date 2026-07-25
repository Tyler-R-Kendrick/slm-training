#!/usr/bin/env python3
"""Run the SLM-292 (AP-010) semantic-contrast objective fixture-scale smoke.

Trains matched control (``semantic_contrast_loss_weight=0.0``) and treatment
(``semantic_contrast_loss_weight>0.0``) TwoTowerModel arms -- identical seed,
architecture, training records, step count, and decode settings -- and writes
the required evidence (loss weight, margin, mutation sampling, positive/
negative distances, generation outcomes) to
``docs/design/iter-slm292-semantic-contrast-smoke-<date>.{json,md}``.

**This is fixture-scale wiring evidence, not the AP-010 promotion claim.**
See the written markdown's disclosure section.

Example:
  python -m scripts.run_slm292_semantic_contrast_smoke
  python -m scripts.run_slm292_semantic_contrast_smoke --steps 40 --seed 7
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm292_semantic_contrast_smoke import (
    render_markdown,
    run_smoke,
)
from slm_training.versioning import build_version_stamp

__all__ = ["main"]

_DESIGN_JSON = Path("docs/design/iter-slm292-semantic-contrast-smoke-20260725.json")
_DESIGN_MD = Path("docs/design/iter-slm292-semantic-contrast-smoke-20260725.md")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--batch-pairs", type=int, default=6)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--treatment-weight", type=float, default=1.0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=None,
        help="Override the SLM-290 corpus path (default: packaged openui_hard_valid_v1)",
    )
    parser.add_argument("--json-out", type=Path, default=_DESIGN_JSON)
    parser.add_argument("--md-out", type=Path, default=_DESIGN_MD)
    args = parser.parse_args(argv)

    report = run_smoke(
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        d_model=args.d_model,
        batch_pairs=args.batch_pairs,
        margin=args.margin,
        treatment_weight=args.treatment_weight,
        corpus_path=args.corpus_path,
        sampling_seed=args.sampling_seed,
    )
    payload: dict[str, Any] = report.to_dict()
    payload["generated_at"] = _now()
    payload["command"] = "python -m scripts.run_slm292_semantic_contrast_smoke"
    payload["version_stamp"] = build_version_stamp(
        "harness.experiments.slm292_semantic_contrast_smoke",
        "model.twotower",
        "data.semantic_contrast",
    )

    import json

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"control final loss={report.control_final_loss:.4f} "
        f"treatment final loss={report.treatment_final_loss:.4f}"
    )
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
