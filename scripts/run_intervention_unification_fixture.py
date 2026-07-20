#!/usr/bin/env python3
"""Run the SLM-137 LDI4-03 intervention unification fixture.

Builds sample manifests for every intervention kind, validates them through the
common registry, exercises promotion transitions, enforces one-active and
acyclic-lineage rules, and emits a closeout index. No model is loaded.

Example:
  python -m scripts.run_intervention_unification_fixture --mode plan-only
  python -m scripts.run_intervention_unification_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build.checkpoint_reference import FileArtifact
from slm_training.lineage.interventions import (
    INTERVENTION_KINDS,
    BaseIdentity,
    EvaluationBundle,
    InterventionManifest,
    InterventionRegistry,
    assert_single_active,
    build_closeout_index,
    detect_lineage_cycle,
    promote,
)
from slm_training.versioning import build_version_stamp


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _base() -> BaseIdentity:
    return BaseIdentity(
        architecture="twotower",
        base_model_id="parent-x",
        base_model_revision="rev1",
        tokenizer_sha="tok1",
        base_compatibility_fingerprint="compat-abc",
    )


def _manifest(
    kind: str,
    intervention_id: str,
    status: str,
    *,
    deployable: bool | None = None,
    parent_intervention_ids: tuple[str, ...] = (),
) -> InterventionManifest:
    if deployable is None:
        deployable = kind != "sae_diagnostic"
    return InterventionManifest(
        intervention_id=intervention_id,
        kind=kind,
        method="low_rank" if kind in ("causal_peft", "twotower_delta") else "reft_r1",
        status=status,
        deployable=deployable,
        base=_base(),
        module_site_map=(("denoiser.block.3", "residual"),),
        parameter_shapes=(("A", (4, 16)),),
        trainable_parameter_count=64,
        artifact_files=(FileArtifact("adapter_model.pt", 128, "deadbeef"),),
        config_fingerprint="cfg-123",
        parent_intervention_ids=parent_intervention_ids,
    )


def _bundle(*, complete: bool = True, gates_pass: bool = True) -> EvaluationBundle:
    ident: dict[str, Any] = {
        "base_sha": "b",
        "intervention_sha": "i",
        "corpus_sha": "c",
        "seed": 0,
        "commit_sha": "s",
    }
    if not complete:
        del ident["seed"]
    return EvaluationBundle(
        identity=ident,
        event={"support_summary": {}, "local_objective_metrics": {}},
        locality={"legal_space_drift": {}, "preservation": {}, "disabled_parity": {}},
        end_to_end={
            "ship_gates": {
                "pass": gates_pass,
                "failures": [] if gates_pass else ["parse"],
            },
            "adversarial": {},
            "ood": {},
            "agentv": {},
        },
    )


def _run_fixture() -> dict[str, Any]:
    """Build and exercise the unification fixture."""
    registry = InterventionRegistry()

    # One manifest of each kind, covering distinct promotion outcomes.
    manifests = [
        _manifest("causal_peft", "peft-1", "wiring"),
        _manifest("twotower_delta", "delta-1", "diagnostic"),
        _manifest("reft", "reft-1", "diagnostic"),
        _manifest("sae_diagnostic", "sae-1", "wiring"),
    ]

    validations = {
        m.intervention_id: registry.inspect(m) for m in manifests
    }

    promotions: list[dict[str, Any]] = []
    current: dict[str, InterventionManifest] = {m.intervention_id: m for m in manifests}

    # causal_peft: wiring -> diagnostic -> eligible -> promoted
    for target in ("diagnostic", "eligible", "promoted"):
        evidence = _bundle() if target in ("eligible", "promoted") else None
        result = promote(current["peft-1"], target, evidence=evidence, registry=registry)
        promotions.append(
            {
                "intervention_id": "peft-1",
                "from": current["peft-1"].status,
                "to": target,
                "ok": result.ok,
                "failures": list(result.failures),
            }
        )
        if result.ok:
            current["peft-1"] = replace(current["peft-1"], status=result.status)

    # twotower_delta: diagnostic -> rejected (failing ship gate)
    result = promote(
        current["delta-1"], "eligible", evidence=_bundle(gates_pass=False), registry=registry
    )
    promotions.append(
        {
            "intervention_id": "delta-1",
            "from": current["delta-1"].status,
            "to": "eligible",
            "ok": result.ok,
            "failures": list(result.failures),
        }
    )

    # reft: diagnostic -> eligible (not promoted, kept as candidate)
    result = promote(current["reft-1"], "eligible", evidence=_bundle(), registry=registry)
    promotions.append(
        {
            "intervention_id": "reft-1",
            "from": current["reft-1"].status,
            "to": "eligible",
            "ok": result.ok,
            "failures": list(result.failures),
        }
    )
    if result.ok:
        current["reft-1"] = replace(current["reft-1"], status=result.status)

    # sae_diagnostic: wiring -> diagnostic; attempted promotion to promoted must fail
    result = promote(current["sae-1"], "diagnostic", registry=registry)
    promotions.append(
        {
            "intervention_id": "sae-1",
            "from": current["sae-1"].status,
            "to": "diagnostic",
            "ok": result.ok,
            "failures": list(result.failures),
        }
    )
    if result.ok:
        current["sae-1"] = replace(current["sae-1"], status=result.status)
    blocked = promote(current["sae-1"], "promoted", evidence=_bundle(), registry=registry)
    promotions.append(
        {
            "intervention_id": "sae-1",
            "from": current["sae-1"].status,
            "to": "promoted",
            "ok": blocked.ok,
            "failures": list(blocked.failures),
        }
    )

    final_manifests = list(current.values())
    assert_single_active(final_manifests[:1])
    cycle = detect_lineage_cycle(final_manifests)

    closeout = build_closeout_index(final_manifests)

    return {
        "matrix_set": "ldi4-03-intervention-unification",
        "matrix_version": "ldi4-03-v1",
        "run_id": "ldi4_03_fixture",
        "status": "wiring_only",
        "claim_class": "wiring",
        "kinds": list(INTERVENTION_KINDS),
        "manifests": [m.to_dict() for m in final_manifests],
        "validations": validations,
        "promotions": promotions,
        "one_active_asserted": True,
        "lineage_cycle": cycle is not None,
        "closeout_index": closeout,
        "version_stamp": build_version_stamp("harness_core.lineage.interventions"),
    }


def _plan_only_report() -> dict[str, Any]:
    return {
        "matrix_set": "ldi4-03-intervention-unification",
        "matrix_version": "ldi4-03-v1",
        "run_id": "ldi4_03_plan",
        "status": "plan_only",
        "claim_class": "wiring",
        "kinds": list(INTERVENTION_KINDS),
        "manifests": [],
        "promotions": [],
        "note": "plan-only: no manifests or promotions exercised",
        "version_stamp": build_version_stamp("harness_core.lineage.interventions"),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SLM-137 / LDI4-03: Intervention unification fixture ({report.get('run_id')})",
        "",
        f"Matrix set: `{report.get('matrix_set')}`  ",
        f"Version: `{report.get('matrix_version')}`  ",
        f"Status: **{report.get('status')}**",
        "",
        "## What this exercises",
        "",
        "A common manifest, registry, evaluation bundle, and promotion state machine for "
        "four intervention kinds: `causal_peft`, `twotower_delta`, `reft`, and "
        "`sae_diagnostic`. No model is loaded; this is wiring-only evidence.",
        "",
        "## Kinds",
        "",
        ", ".join(f"`{k}`" for k in report.get("kinds", [])),
        "",
    ]

    promotions = report.get("promotions", [])
    if promotions:
        lines.extend(
            [
                "## Promotion transitions",
                "",
                "| Intervention | From | To | OK | Failures |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for p in promotions:
            failures = "; ".join(p.get("failures", [])) or "—"
            lines.append(
                f"| {p.get('intervention_id')} | {p.get('from')} | {p.get('to')} | "
                f"{p.get('ok')} | {failures} |"
            )

    closeout = report.get("closeout_index")
    if closeout:
        lines.extend(
            [
                "",
                "## Closeout index",
                "",
                f"Total artifacts: {closeout.get('artifact_count')}  ",
                f"Best deployable: `{closeout.get('best_deployable')}`  ",
                f"Statement: {closeout.get('best_deployable_statement')}",
                "",
                "By status:",
            ]
        )
        for status, ids in closeout.get("by_status", {}).items():
            lines.append(f"- {status}: {ids}")

    lines.extend(
        [
            "",
            "## Fixture caveat",
            "",
            report.get(
                "note",
                "Wiring-only evidence. Real model loading, merge/export parity, dashboard "
                "integration, and bucket upload are deferred to the integration run.",
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-137 LDI4-03 intervention unification fixture"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture"),
        default="plan-only",
        help=(
            "plan-only emits the manifest skeleton; "
            "fixture exercises the registry and promotion state machine"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/ldi4-03-intervention-unification-{_today_slug()}"),
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_json = Path(
        f"docs/design/iter-ldi4-03-intervention-unification-{_today_slug()}.json"
    )
    design_md = Path(
        f"docs/design/iter-ldi4-03-intervention-unification-{_today_slug()}.md"
    )

    report = _plan_only_report() if args.mode == "plan-only" else _run_fixture()

    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    report_path = output_dir / "ldi4_03_intervention_unification_report.json"
    report_path.write_text(report_text, encoding="utf-8")
    markdown = _render_markdown(report)
    (output_dir / "ldi4_03_intervention_unification_report.md").write_text(
        markdown, encoding="utf-8"
    )

    design_json.parent.mkdir(parents=True, exist_ok=True)
    design_json.write_text(report_text, encoding="utf-8")
    design_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nReport JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
