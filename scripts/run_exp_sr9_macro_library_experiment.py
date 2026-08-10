#!/usr/bin/env python3
"""RSP-005 (SLM-480): run the locked EXP-SR-9 prospective certified
macro/library-learning experiment and persist honest evidence.

Thin wrapper around
``harnesses.experiments.symbolic_macro_library.run_campaign``: generates a
small deterministic in-process symbolic-regression corpus, mines a macro
library exclusively from its ``corpus_split=="train"`` records, and
measures ``macro_library_size_reduction_rate`` (the locked EXP-SR-9 primary
endpoint) for three matched arms (``no_macros``, ``frequency_macros``,
``mdl_macros``) on the frozen, never-mined-from ``corpus_split=="test"``
slice.

This experiment's claim_class is ``fixture`` -- diagnostic evidence for
later structural-search work, never a checkpoint promotion.

Writes ``docs/design/exp-sr-9-macro-library-experiment.json`` and the
companion markdown (``documenting-experiment-results`` convention).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.symbolic_macro_library import (
    ARM_IDS,
    Rsp005CampaignV1,
    run_campaign,
)


def run_experiment() -> dict[str, Any]:
    campaign = Rsp005CampaignV1()
    with tempfile.TemporaryDirectory() as tmp:
        return run_campaign(campaign, root=Path(tmp))


def _markdown(report: dict[str, Any]) -> str:
    arms = report["arms"]
    lines = [
        "# EXP-SR-9: prospective certified macro/library-learning experiment",
        "",
        f"Campaign manifest: `{report['manifest_sha256']}` (claim_class=`{report['claim_class']}`).",
        f"Catalogue manifest: `{report['catalogue_manifest_sha256']}` (`{report['catalogue_id']}`).",
        "",
        report["scope_disclaimer"],
        "",
        f"- Train records: {report['n_train_records']}",
        f"- Held-out (frozen, never mined from) test records: {report['n_test_records']}",
        f"- Candidate subtrees observed while mining: {report['candidate_count']}",
        "",
        "## Arms (prospective `macro_library_size_reduction_rate` on the held-out slice)",
        "",
        "| Arm | Reduction rate | Library size | Selection policy |",
        "| --- | --- | --- | --- |",
    ]
    for arm_id in ARM_IDS:
        arm = arms[arm_id]
        lines.append(
            f"| `{arm_id}` | {arm['reduction_rate']:.6f} | {arm['library_size']} | `{arm['selection_policy']}` |"
        )
    lines += [
        "",
        f"- Primary metric: `{report['primary_metric']}`, minimum_effect={report['minimum_effect']}",
        f"- Leakage audit: `is_clean`={report['leakage_audit']['is_clean']}",
        f"- Expansion-equivalence: {len(report['expansion_equivalence'])} admitted mdl_macros checked, "
        f"all_verified={report['expansion_equivalence_all_verified']}",
        f"- Falsifier holds: {report['falsifier_holds']}",
        "",
        f"**Authorized (diagnostic signal only, never a promotion):** `{report['authorized']}`",
        "",
        "Full detail: `docs/design/exp-sr-9-macro-library-experiment.json`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json", type=Path, default=Path("docs/design/exp-sr-9-macro-library-experiment.json")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("docs/design/exp-sr-9-macro-library-experiment.md")
    )
    args = parser.parse_args(argv)

    report = run_experiment()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "authorized": report["authorized"],
                "falsifier_holds": report["falsifier_holds"],
                "mdl_reduction_rate": report["arms"]["mdl_macros"]["reduction_rate"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
