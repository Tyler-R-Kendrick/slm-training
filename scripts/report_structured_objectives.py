"""No-model-update objective report for LDI3-01 structured objectives (SLM-128).

Runs the Legal-Set FTPO (pairwise + mass), TAB-PO-inspired barrier, and
TBPO-inspired ratio control over a small frozen fixture corpus and emits a
deterministic JSON report of objective/component values, barrier-active and
erosion metrics, and raw-vs-legal-space probabilities. **No model is updated and
no quality claim is made** — the objectives are *adapted*, not reproduced, from
the cited work.

    python scripts/report_structured_objectives.py --out docs/design/ldi3-01-structured-objective-report-20260718.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from slm_training.harnesses.preference.decision_events_v2 import ObjectiveView
from slm_training.harnesses.preference.structured_objectives import (
    StructuredObjectiveConfig,
    structured_decision_loss,
)

# One frozen, admitted fixture state: 5 legal actions, 2 verified good, 2 bad.
_LEGAL = (0, 1, 2, 3, 4)
_VIEW = ObjectiveView(
    good_action_ids=(0, 1),
    bad_action_ids=(2, 3),
    ambiguous_action_ids=(4,),
    unobserved_action_ids=(),
    weights=((0, 1.0), (1, 0.5), (2, 1.0), (3, 1.0)),
    materializer_id="fixture",
    materializer_config_hash="frozen",
)
_LOGITS = torch.tensor([0.4, -0.3, 0.1, -0.6, 0.2])
_REFERENCE = torch.tensor([0.8, 0.1, -0.2, -0.4, 0.0])


def _run(config: StructuredObjectiveConfig, **kwargs: Any) -> dict[str, Any]:
    loss, metrics = structured_decision_loss(
        _LOGITS, _VIEW, legal_action_ids=_LEGAL, config=config, **kwargs
    )
    return {"config_fingerprint": config.fingerprint(), "loss": float(loss), "metrics": metrics}


def build_report() -> dict[str, Any]:
    mask = torch.tensor([1.0, 0.0])  # only good action 0 is verified-critical
    full_probs = torch.softmax(_LOGITS, dim=-1)
    legal_probs = torch.softmax(_LOGITS.index_select(0, torch.tensor(_LEGAL)), dim=-1)
    return {
        "note": "Adapted objectives (not reproduced / not SOTA). No model update; no quality claim.",
        "fixture": {"legal": list(_LEGAL), "view": _VIEW.to_dict()},
        "raw_vs_legal_probability": {
            "full_vocab_probs": [round(x, 6) for x in full_probs.tolist()],
            "legal_space_probs": [round(x, 6) for x in legal_probs.tolist()],
        },
        "objectives": {
            "legal_set_ftpo_pairwise": _run(
                StructuredObjectiveConfig(name="legal_set_ftpo", variant="pairwise")
            ),
            "legal_set_ftpo_mass": _run(
                StructuredObjectiveConfig(name="legal_set_ftpo", variant="mass")
            ),
            "tab_barrier": _run(
                StructuredObjectiveConfig(name="tab_barrier", barrier_p=0.3),
                reference_logits=_REFERENCE,
                critical_good_mask=mask,
            ),
            "tbpo_inspired": _run(
                StructuredObjectiveConfig(name="tbpo_inspired"),
                reference_logits=_REFERENCE,
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
