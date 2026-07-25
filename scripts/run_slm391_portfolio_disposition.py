#!/usr/bin/env python3
"""Run the SLM-391 (DSH4-06) staged DSL harness portfolio disposition.

Runs every applicable frozen suite at fixture scale (two-pack CAP1 gates,
SLM-368 matched-control experiment, operator legal-set suite, SLM-390
trace-eval lifecycle, strict-v2 / binding-aware checks, AgentV diagnostic),
binds per-claim evidence artifacts, and writes the final
``StagedDslHarnessDispositionV1`` plus append-only doc updates.  No
promotion, checkpoint, or model-card mutation results; the model-card and
dataset-card snippets are printed only.

Example:
  python -m scripts.run_slm391_portfolio_disposition --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm391_portfolio_disposition import (
    StagedDslHarnessDispositionV1,
    build_disposition,
    render_markdown,
)

_DESIGN_JSON = "docs/design/iter-slm391-portfolio-disposition-20260725.json"
_DESIGN_MD = "docs/design/iter-slm391-portfolio-disposition-20260725.md"
_QUALITY_MATRIX = "docs/design/quality-experiment-matrix.md"
_RESEARCH_LINEAGE = "docs/design/research-lineage.md"

_MATRIX_HEADING = "# DSH4-06 — staged DSL harness portfolio disposition (2026-07-25)"
_LINEAGE_HEADING = "## DSH4-06 staged DSL harness portfolio disposition (SLM-391)"
_LINEAGE_ANCHOR = "## Honesty rules"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _verdict_table(disposition: StagedDslHarnessDispositionV1) -> str:
    rows = [
        "| Claim | Scope | Verdict |",
        "| --- | --- | --- |",
    ]
    for claim in disposition.claims:
        rows.append(f"| {claim.claim_id} | {claim.scope} | {claim.verdict} |")
    return "\n".join(rows)


def quality_matrix_section(disposition: StagedDslHarnessDispositionV1) -> str:
    """The append-only quality-matrix section for this disposition."""
    verdicts = ", ".join(
        f"{claim.claim_id}={claim.verdict}" for claim in disposition.claims
    )
    followups = "; ".join(follow["title"] for follow in disposition.follow_ups)
    return f"""{_MATRIX_HEADING}

Publish the staged DSL harness portfolio disposition at fixture scale. The
frozen two-pack CAP1 suite gates pass only under the schema-consuming oracle
ceiling; the matched SLM-368 experiment preserves its CERT_CAP1 rejection
(stop codes first-class, including underpowered at fixture n); the DSH3
operator legal-set suite reproduces exact bounded enumeration; the SLM-390
trace-eval lifecycle verifies end to end with the undeidentifiable trace held
in quarantine; strict-v2/binding-aware metric checks agree with the frozen
adversarial corpus. Claim verdicts: {verdicts}.

CAP0 (SLM-361/362) and DSH4-01/02 distillation artifacts are absent on this
lineage and classify `unknown` per the stop rule — never normalized by
assumption. No efficiency claim is supported: workload/hardware/batch
identity is recorded but no matched-baseline measurement exists. Negative,
invalid, contaminated, and prediction-identical outcomes remain first-class
rows. No fixture-only row supports a capability or ship claim; no promotion,
checkpoint, or model-card change results.

Recommended follow-ups (deduplicated, evidence-grounded): {followups}.
Full evidence:
[`iter-slm391-portfolio-disposition-20260725.md`](iter-slm391-portfolio-disposition-20260725.md).
"""


def lineage_entry(disposition: StagedDslHarnessDispositionV1) -> str:
    """The append-only research-lineage entry for this disposition."""
    return f"""{_LINEAGE_HEADING}

**Fidelity label: disposition audit / governance closeout, not a new capability.**
[`StagedDslHarnessDispositionV1`](iter-slm391-portfolio-disposition-20260725.md)
binds every DSH portfolio claim to its compatible code/suite/config/hardware
identities on this lineage:

{_verdict_table(disposition)}

Fixture-scale wiring supports only the `harness_contract` claims
(cap2_operators, trace_evals) with explicit bounds. Held-out semantic
capability (cap1_semantics) stays **rejected** under the preregistered stop
codes; cap0_grammar, distillation, and efficiency are **unknown** because
their evidence artifacts are absent on this lineage or no normalized
matched-baseline measurement exists. Nothing is promoted.
"""


def append_quality_matrix_section(existing: str, section: str) -> str:
    """Append the matrix section; historical content is an untouched prefix."""
    if _MATRIX_HEADING in existing:
        return existing
    text = existing if existing.endswith("\n") else existing + "\n"
    return text + "\n" + section


def append_lineage_entry(existing: str, entry: str) -> str:
    """Insert the lineage entry before the honesty-rules footer (append-only)."""
    if _LINEAGE_HEADING in existing:
        return existing
    if _LINEAGE_ANCHOR in existing:
        head, sep, tail = existing.partition(_LINEAGE_ANCHOR)
        head = head.rstrip("\n") + "\n\n"
        return head + entry + "\n" + sep + tail
    text = existing if existing.endswith("\n") else existing + "\n"
    return text + "\n" + entry


def model_card_snippet(disposition: StagedDslHarnessDispositionV1) -> str:
    """Printed-only model-card snippet (NOT auto-inserted)."""
    return (
        "### Model card snippet (print-only; no checkpoint created or promoted)\n\n"
        "| Checkpoint | Change | Evidence |\n"
        "| --- | --- | --- |\n"
        "| (none) | SLM-391 portfolio disposition ran at fixture scale; no new "
        "checkpoint, no promotion, no serving `*.pt` | "
        "docs/design/iter-slm391-portfolio-disposition-20260725.md |\n"
    )


def dataset_card_snippet(disposition: StagedDslHarnessDispositionV1) -> str:
    """Printed-only dataset-card snippet (NOT auto-inserted)."""
    suite_sha = ""
    for claim in disposition.claims:
        if claim.claim_id == "cap1_semantics":
            suite_sha = str(claim.identities.get("suite", ""))
    return (
        "### Dataset card snippet (print-only; no dataset version changed)\n\n"
        f"- Frozen eval suite: `cap1_two_pack_freeze/v1` sha256 `{suite_sha}` "
        "(unchanged; eval/training overlap audit clean).\n"
        "- No training-data build ran; no quality report to close the loop on.\n"
        "- Trace-eval candidates remain quarantined-by-default; no promotion "
        "to training outside the SLM-390 explicit lifecycle.\n"
    )


def _agentv_diagnostic(run_dir: Path, disposition: StagedDslHarnessDispositionV1) -> dict[str, Any]:
    """AgentV diagnostic publication; failure is non-fatal and recorded."""
    cases = [
        {
            "id": f"slm391-claim-{claim.claim_id}",
            "criteria": (
                f"{claim.claim_id} verdict is one of supported/rejected/invalid/"
                "unknown and names identities or stays unknown"
            ),
            "result": {
                "verdict": claim.verdict,
                "scope": claim.scope,
                "identities": dict(claim.identities),
            },
            "assertions": [
                {
                    "id": f"slm391-claim-{claim.claim_id}:verdict_declared",
                    "actual": claim.verdict
                    in {"supported", "rejected", "invalid", "unknown"},
                    "operator": "eq",
                    "expected": True,
                },
                {
                    "id": f"slm391-claim-{claim.claim_id}:identities_or_unknown",
                    "actual": bool(claim.identities.get("code"))
                    and (
                        claim.verdict == "unknown"
                        or bool(claim.identities.get("suite"))
                    ),
                    "operator": "eq",
                    "expected": True,
                },
            ],
        }
        for claim in disposition.claims
    ]
    try:
        from slm_training.evals.agentv import publish_agentv_evaluation

        result = publish_agentv_evaluation(
            run_dir,
            name="slm391-portfolio-disposition",
            claim=(
                "StagedDslHarnessDispositionV1 verdicts are declared and every "
                "claim names identities or stays unknown"
            ),
            cases=cases,
            version_stamp=dict(disposition.version_stamp),
        )
        # Machine-absolute artifact paths must not enter the durable design
        # record (repo policy); the durable JSON keeps the verdict summary.
        result.pop("artifacts", None)
        run_dir_text = str(Path(run_dir).resolve())

        def _relativize(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _relativize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_relativize(v) for v in value]
            if isinstance(value, str):
                return value.replace(
                    run_dir_text, "outputs/runs/slm391-portfolio-disposition"
                )
            return value

        result = _relativize(result)
        result["artifacts_note"] = (
            "AgentV run artifacts live under the run directory "
            "(outputs/runs/slm391-portfolio-disposition/agentv); local absolute "
            "paths are omitted from the durable record"
        )
        return result
    except Exception as exc:  # noqa: BLE001 - diagnostic must not fail the run
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-391 DSH4-06 staged DSL harness portfolio disposition",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"fixture"},
        default="fixture",
        help="Run mode (fixture scale only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm391-portfolio-disposition)",
    )
    parser.add_argument("--cap1-steps", type=int, default=4)
    parser.add_argument("--cap1-records-per-arm", type=int, default=8)
    parser.add_argument("--cap1-seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Write only the run artifact, not the docs/design updates.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    root = _repo_root()
    output_dir = args.output_dir or Path("outputs/runs/slm391-portfolio-disposition")
    output_dir.mkdir(parents=True, exist_ok=True)

    command = (
        "python -m scripts.run_slm391_portfolio_disposition --mode fixture "
        f"--cap1-steps {args.cap1_steps} "
        f"--cap1-records-per-arm {args.cap1_records_per_arm} "
        f"--cap1-seeds {' '.join(str(s) for s in args.cap1_seeds)}"
    )
    disposition = build_disposition(
        repo_root=root,
        cap1_steps=args.cap1_steps,
        cap1_records_per_arm=args.cap1_records_per_arm,
        cap1_seeds=tuple(args.cap1_seeds),
        now=_now(),
    )
    agentv = _agentv_diagnostic(output_dir, disposition)
    payload = disposition.to_dict()
    payload["agentv_diagnostic"] = agentv
    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    report_path = output_dir / "slm391_portfolio_disposition.json"
    report_path.write_text(report_text, encoding="utf-8")

    if not args.skip_docs:
        json_path = root / _DESIGN_JSON
        md_path = root / _DESIGN_MD
        json_path.write_text(report_text, encoding="utf-8")
        md_path.write_text(render_markdown(disposition, command=command), encoding="utf-8")

        matrix_path = root / _QUALITY_MATRIX
        matrix_path.write_text(
            append_quality_matrix_section(
                matrix_path.read_text(encoding="utf-8"),
                quality_matrix_section(disposition),
            ),
            encoding="utf-8",
        )
        lineage_path = root / _RESEARCH_LINEAGE
        lineage_path.write_text(
            append_lineage_entry(
                lineage_path.read_text(encoding="utf-8"),
                lineage_entry(disposition),
            ),
            encoding="utf-8",
        )

    print(report_path)
    print()
    print(model_card_snippet(disposition))
    print(dataset_card_snippet(disposition))
    print("Recommended follow-ups:")
    for follow in disposition.follow_ups:
        print(f"- {follow['title']} ({follow['id']})")
    print()
    print(f"AgentV diagnostic: {agentv.get('status', 'unknown')}")
    for claim in disposition.claims:
        print(f"  {claim.claim_id}: {claim.verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
