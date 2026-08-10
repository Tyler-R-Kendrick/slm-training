#!/usr/bin/env python3
"""RSP-009 (SLM-490): publish the cross-experiment disposition for every
EXP-SR mechanism (exp-sr-1 through exp-sr-12).

Reuses SGS-009's mechanism_disposition_report.py schema/builders exactly
(no forked disposition mechanism), one structured
``MechanismDispositionRecordV1`` per catalogue family -- the natural unit
matching "every EXP-SR mechanism has one structured disposition" (the
catalogue *is* the mechanism registry). Every record is built through
``build_mechanism_disposition_record``, which fails closed on this issue's
own two hardest rules: a mechanism cannot be ``adopt_primary``/
``adopt_optional`` without ``measured``/``promotable``/``ship`` evidence,
and ``evidence_class="fixture"`` can never pair with ``default_state="on"``.
Every experiment this initiative ran stayed fixture-scale (mock external
judges, synthetic human raters, small in-process corpora) -- so no
mechanism here is disposed ``adopt_primary``/``adopt_optional``: that would
violate this module's own validator, not just this script's judgment.

    python -m scripts.publish_exp_sr_disposition
    python -m scripts.publish_exp_sr_disposition --check

``--check`` re-derives the report from this script's own record table and
verifies it against the last-published JSON (byte-identical after
stripping the report's own hash), matching the ``documenting-experiment-
results`` convention of a durable, reproducible evidence artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.mechanism_disposition_report import (
    Disposition,
    MechanismDispositionRecordV1,
    build_disposition_report,
    build_mechanism_disposition_record,
    disposition_report_integrity_ok,
)

EVIDENCE_CUTOFF = "2026-08-10"


@dataclass(frozen=True)
class _MechanismSpec:
    family_id: str
    mechanism_id: str
    owner_module: str
    authority_class: str
    default_state: str
    evidence_class: str
    disposition: Disposition
    rationale: str
    activation_evidence: str = ""
    applicable_packs: tuple[str, ...] = ()
    matched_controls: bool = False
    semantic_effect: str = ""
    cost_cold_effect: str = ""
    safety_invariants: tuple[str, ...] = ()
    external_dependencies: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


MECHANISM_SPECS: tuple[_MechanismSpec, ...] = (
    _MechanismSpec(
        family_id="exp-sr-1",
        mechanism_id="oracle_localization",
        owner_module="src/slm_training/data/semantic_plan/oracle.py",
        authority_class="oracle-diagnostic",
        default_state="n/a",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        matched_controls=True,
        semantic_effect=(
            "roles/bindings factors show a material causal ceiling "
            "(oracle_factor_localization_precision=1.0 on the authorized "
            "subset); archetype/topology do not clear minimum_effect=0.03"
        ),
        applicable_packs=("openui",),
        issues=("SLM-467", "SLM-476"),
        rationale=(
            "SIE-002/EXP-SR-1 localized which semantic-plan factors have a "
            "material causal ceiling on a VCE-008-cleaned fixture pool "
            "(no-op/shuffled/destructive matched controls). Fixture-scale "
            "(claim_class=fixture): cannot promote a default. Authorizes "
            "only roles/bindings for later learned-predictor work -- "
            "archetype/style_layout stays an explicit placeholder, never a "
            "claimed authorization."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-2",
        mechanism_id="prompt_factor_predictor",
        owner_module="src/slm_training/models/prompt_factor_predictor.py",
        authority_class="advisory-learned",
        default_state="off",
        evidence_class="rejected",
        disposition=Disposition.REJECT,
        matched_controls=True,
        semantic_effect=(
            "requirement_predictor_f1=0.0 (learned arm); delta vs frequency "
            "baseline=-0.448; dominated_by_simpler_control=True"
        ),
        applicable_packs=("openui",),
        issues=("SLM-477", "SLM-482"),
        rationale=(
            "SIE-004/EXP-SR-2 evaluated the SIE-003 predictor seam end-to-"
            "end against no-predictor/deterministic/frequency/retrieval "
            "controls: the learned predictor is dominated by the simpler "
            "frequency baseline. Per this closeout's own rule ('when "
            "mechanisms are equivalent within uncertainty, prefer the "
            "simpler/lower-cost owner'), the predictor is rejected -- stays "
            "advisory-learned, off, never entered any hard-prune path."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-3",
        mechanism_id="evaluator_calibration_protocol",
        owner_module="src/slm_training/evals/evaluator_calibration_protocol.py",
        authority_class="evaluation-only",
        default_state="n/a",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        matched_controls=True,
        semantic_effect=(
            "evaluator_human_agreement_kappa=0.0 on 12 labeled-mock pairs; "
            "below minimum_effect=0.1 and the catalogue's le-0.0 kill gate"
        ),
        external_dependencies=(
            "real external-judge API credentials",
            "real human raters",
        ),
        issues=("SLM-478", "SLM-483"),
        rationale=(
            "SIE-006/EXP-SR-3 ran VCE-010's calibration protocol through "
            "the locked exp-sr-3 identity, but this environment has no "
            "real external-judge credentials or human raters (SLM-483's "
            "own external-blocked escape hatch): both the external-judge "
            "and human-rater arms are labeled mocks. The kappa=0.0 result "
            "is honest evidence about the MOCK fixture, not the real "
            "evaluator's true calibration -- retained as diagnostic-only "
            "pending a genuine external-judge/blinded-human run, never "
            "rejected on synthetic data alone."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-4",
        mechanism_id="semantic_repair_witness_cegis",
        owner_module="src/slm_training/harnesses/experiments/rsp001_cegis_repair.py",
        authority_class="advisory-learned",
        default_state="off",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        matched_controls=True,
        semantic_effect=(
            "verified_repair_yield=1.0 (witness_cegis arm) vs regenerate "
            "baseline, delta=+0.375 on the fixture corpus"
        ),
        applicable_packs=("openui",),
        issues=("SLM-488",),
        rationale=(
            "RSP-001/EXP-SR-4 shows a clear positive fixture-scale signal "
            "for witness-guided CEGIS repair over regenerate/edit-distance/ "
            "learned-scorer controls. claim_class=fixture structurally "
            "blocks adopt_optional under this schema's own validator "
            "(fixture-only evidence cannot adopt) regardless of the source "
            "doc's own 'adopt-optional' recommendation language -- retained "
            "as diagnostic, default lever stays explicitly off."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-5",
        mechanism_id="compute_state_controller",
        owner_module="src/slm_training/harnesses/experiments/sie008_voc_controller.py",
        authority_class="advisory-learned",
        default_state="off",
        evidence_class="rejected",
        disposition=Disposition.REJECT,
        matched_controls=True,
        semantic_effect=(
            "compute_value_regret=0.5036 (symbolic_rule arm, lower is "
            "better) vs hand_threshold control, delta=-0.2906 (worse)"
        ),
        issues=("SLM-473", "SLM-484"),
        rationale=(
            "SIE-008/EXP-SR-5 replayed the symbolic value-of-compute "
            "controller against hand_threshold/linear_scorer/gold_oracle "
            "arms on the SIE-007 substrate: the symbolic rule loses to "
            "both simpler controls. Rejected; capacity/compute claims stay "
            "advisory-learned and size-matched per goal-invariant VI."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-6",
        mechanism_id="bounded_search_neural_prior",
        owner_module="src/slm_training/harnesses/experiments/rsp002_bounded_search.py",
        authority_class="advisory-learned",
        default_state="off",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        matched_controls=True,
        semantic_effect=(
            "bounded_search_hit_rate=1.0 (neural_prior arm) vs 0.0 "
            "(left_to_right baseline), delta=1.0 on the fixture corpus"
        ),
        issues=("SLM-489",),
        rationale=(
            "RSP-002/EXP-SR-6 shows a clear positive fixture-scale signal "
            "for neural-prior search ordering over the existing decode "
            "baseline and a model-free enumerative control. Fixture-only "
            "evidence blocks adoption under this schema; retained as "
            "diagnostic, default stays off pending real reachability-"
            "matched, multi-seed evidence."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-7",
        mechanism_id="packed_semantic_summary",
        owner_module="src/slm_training/dsl/grammar/fastpath/packed_semantic_summary.py",
        authority_class="evaluation-only",
        default_state="off",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        matched_controls=True,
        semantic_effect=(
            "static_summary_domain_parity_rate=1.0 (9/9 domain checks); "
            "packed-summary cold-path improvement +82.5ms"
        ),
        applicable_packs=("openui",),
        issues=("SLM-485",),
        rationale=(
            "RSP-003/EXP-SR-7 measured exact domain parity against the "
            "live oracle on this fixture, with a real cold-path cost win. "
            "The source evidence doc's own recommendation label was "
            "'inconclusive_fixture' (cost/promotion posture, not measurement "
            "doubt) -- the underlying result is an unambiguous fixture-scale "
            "positive, so retain_diagnostic reflects it more accurately "
            "than inconclusive; excluded from production default/checkpoint "
            "regardless, per fixture-only evidence."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-8",
        mechanism_id="symbolic_egraph",
        owner_module="src/slm_training/harnesses/experiments/symbolic_egraph.py",
        authority_class="evaluation-only",
        default_state="off",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        matched_controls=True,
        semantic_effect=(
            "egraph_certified_equivalence_rate=1.0; matches canonical-hash "
            "dedup's unique-form set and Pareto frontier exactly"
        ),
        applicable_packs=("symbolic_regression",),
        issues=("SLM-479",),
        rationale=(
            "RSP-004/EXP-SR-8: e-graph saturation/extraction is fully "
            "certified but produces no frontier improvement over plain "
            "canonicalize_expr -- mathematically expected, since SRP-005's "
            "current REWRITE_RULES are each complexity-non-increasing and "
            "confluent, so both mechanisms provably converge to the same "
            "fixed point. Per SLM-479's own acceptance criterion, stays "
            "default-off regardless of fixture win; no production "
            "canonicalization path imports this module."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-9",
        mechanism_id="symbolic_macro_library",
        owner_module="src/slm_training/harnesses/experiments/symbolic_macro_library.py",
        authority_class="evaluation-only",
        default_state="off",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        matched_controls=True,
        semantic_effect=(
            "macro_library_size_reduction_rate=0.194 (mdl_macros); clears "
            "minimum_effect=0.02 vs no_macros but ties frequency_macros "
            "exactly on this small fixture vocabulary"
        ),
        applicable_packs=("symbolic_regression",),
        issues=("SLM-480",),
        rationale=(
            "RSP-005/EXP-SR-9: the MDL net-gain objective clears the "
            "no-macros baseline but shows no measured advantage over the "
            "simpler frequency-only control on this small fixture "
            "vocabulary -- an honest, unresolved comparison, not a clean "
            "win for the MDL-specific claim. All 16 admitted macros passed "
            "independent expansion-equivalence re-verification; retained "
            "diagnostic, evaluation-only, imported by no production "
            "canonicalization path."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-10",
        mechanism_id="quality_diversity_corpus",
        owner_module="src/slm_training/harnesses/experiments/rsp006_quality_diversity.py",
        authority_class="evaluation-only",
        default_state="off",
        evidence_class="rejected",
        disposition=Disposition.REJECT,
        matched_controls=True,
        semantic_effect=(
            "corpus_topology_coverage=0.7222 (quality_diversity arm), "
            "exactly tied with the random_matched control"
        ),
        issues=("SLM-481",),
        rationale=(
            "RSP-006/EXP-SR-10: the quality-diversity generation strategy "
            "does not beat matched-compute random rejection sampling -- "
            "an exact tie, not a directional loss, but the falsifier "
            "('doesn't exceed matched-compute rejection sampling') fires "
            "on a tie by its own preregistered wording. Rejected; the "
            "simpler random_matched control is preferred per this "
            "closeout's own equivalence rule."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-11",
        mechanism_id="pysr_srbench_adapter",
        owner_module="src/slm_training/dsl/symbolic_expr_pysr_adapter.py",
        authority_class="evaluation-only",
        default_state="n/a",
        evidence_class="scratch",
        disposition=Disposition.BLOCKED,
        external_dependencies=("pysr/Julia runtime", "SRBench comparison protocol"),
        issues=("SLM-486",),
        rationale=(
            "RSP-007/EXP-SR-11 was never executed: no locked exp-sr-11 "
            "campaign or docs/design evidence artifact exists in this "
            "repository as of this closeout's evidence cutoff. SRP-011's "
            "isolated adapter (symbolic_expr_pysr_adapter.py) is tested in "
            "isolation only (mocked/golden output; never imports pysr/"
            "Julia), which is not the same as a matched external-benchmark "
            "comparison. Disposed blocked rather than inferred from "
            "isolated unit coverage or silently omitted; a follow-up issue "
            "to actually run (or formally external-block) EXP-SR-11 is "
            "filed alongside this report."
        ),
    ),
    _MechanismSpec(
        family_id="exp-sr-12",
        mechanism_id="second_pack_portability",
        owner_module="src/slm_training/harnesses/experiments/rsp008_second_pack_portability.py",
        authority_class="evaluation-only",
        default_state="n/a",
        evidence_class="rejected",
        disposition=Disposition.REJECT,
        matched_controls=True,
        semantic_effect=(
            "second_pack_portability_parity_rate=0.909 (10/11 checks); "
            "3 required shared-seam forks found: dsl_pack_registry_shadow, "
            "an OpenUI-pinned language contract, and a non-conformant "
            "SyGuS-inspired seam"
        ),
        applicable_packs=("openui", "symbolic_regression"),
        issues=("SLM-474", "SLM-487"),
        rationale=(
            "RSP-008/EXP-SR-12: zero-fork cross-pack certification is "
            "falsified -- the fork inventory IS the evidence the "
            "falsifier fired correctly, satisfying this closeout's own "
            "rule that cross-pack claims require actual replication, not "
            "an inference from single-pack success. Rejected as a "
            "cross-pack certification claim; OpenUI and symbolic-regression "
            "pack claims remain scoped separately."
        ),
    ),
)


def build_records() -> tuple[MechanismDispositionRecordV1, ...]:
    return tuple(
        build_mechanism_disposition_record(
            mechanism_id=spec.mechanism_id,
            owner_module=spec.owner_module,
            owner_version="v1",
            default_state=spec.default_state,
            authority_class=spec.authority_class,
            evidence_cutoff=EVIDENCE_CUTOFF,
            evidence_class=spec.evidence_class,
            disposition=spec.disposition,
            rationale=spec.rationale,
            activation_evidence=spec.activation_evidence,
            applicable_packs=spec.applicable_packs,
            matched_controls=spec.matched_controls,
            semantic_effect=spec.semantic_effect,
            cost_cold_effect=spec.cost_cold_effect,
            safety_invariants=spec.safety_invariants,
            external_dependencies=spec.external_dependencies,
            issues=spec.issues,
        )
        for spec in MECHANISM_SPECS
    )


def _markdown(report_dict: dict[str, Any], table_markdown: str) -> str:
    rejected = [r for r in report_dict["records"] if r["disposition"] == "reject"]
    blocked = [r for r in report_dict["records"] if r["disposition"] == "blocked"]
    retained = [r for r in report_dict["records"] if r["disposition"] == "retain_diagnostic"]
    lines = [
        "# EXP-SR cross-experiment disposition (RSP-009 / SLM-490)",
        "",
        "**Schema:** `mechanism_disposition_report/v1` (SGS-009, "
        "`harnesses/experiments/mechanism_disposition_report.py` -- reused, "
        "not forked)",
        "**Matrix set:** exp-sr-1 through exp-sr-12 "
        "(`harnesses/experiments/exp_sr_catalogue.py`)",
        f"**Evidence cutoff:** `{EVIDENCE_CUTOFF}`",
        "**Claim class:** every disposed mechanism stayed fixture-scale "
        "(mock external judges, synthetic human raters, small in-process "
        "corpora); none is `adopt_primary`/`adopt_optional` -- that "
        "combination with `evidence_class=fixture` is structurally "
        "rejected by `build_mechanism_disposition_record`'s own validator, "
        "not just this report's judgment.",
        "",
        "## Canonical artifact pair",
        "",
        f"- `docs/design/exp-sr-disposition-{EVIDENCE_CUTOFF.replace('-', '')}.json` "
        "-- the structured `MechanismDispositionReportV1` (hash-verified).",
        f"- `docs/design/exp-sr-disposition-{EVIDENCE_CUTOFF.replace('-', '')}.md` "
        "-- this document.",
        "",
        "## Executive finding",
        "",
        f"Of 12 registered EXP-SR families, {len(retained)} are retained as "
        f"honest positive-or-mixed diagnostic signals "
        f"({', '.join(r['mechanism_id'] for r in retained)}), "
        f"{len(rejected)} are rejected on measured evidence "
        f"({', '.join(r['mechanism_id'] for r in rejected)}), and "
        f"{len(blocked)} never executed and is disposed blocked "
        f"({', '.join(r['mechanism_id'] for r in blocked)}). No mechanism "
        "reached adopt_primary/adopt_optional: every campaign this "
        "initiative ran stayed fixture-scale, and fixture-only evidence "
        "cannot adopt (this closeout's own preregistered rule, enforced "
        "structurally by the disposition schema's validator). Production "
        "changes to any default/champion pointer occur only through the "
        "normal model-lineage process, never from this report directly.",
        "",
        "## Evidence chronology",
        "",
    ]
    for spec in MECHANISM_SPECS:
        issues = ", ".join(spec.issues)
        lines.append(f"- `{spec.family_id}` ({issues}): `{spec.mechanism_id}`")
    lines += [
        "",
        "## Mechanism disposition table",
        "",
        table_markdown,
        "",
        "## Cross-pack summary",
        "",
        "- OpenUI-pack mechanisms (exp-sr-1, exp-sr-2, exp-sr-4, exp-sr-6, "
        "exp-sr-7) and symbolic-regression-pack mechanisms (exp-sr-8, "
        "exp-sr-9) are each scoped and disposed within their own pack; "
        "none of their results are extrapolated across packs.",
        "- The one genuine cross-pack claim (exp-sr-12, second-pack "
        "portability) was rejected on real replication evidence (3 "
        "required shared-seam forks) rather than assumed from single-pack "
        "success -- satisfying this closeout's own 'cross-pack claims "
        "require actual replication' rule.",
        "",
        "## Rejected mechanisms",
        "",
    ]
    for record in rejected:
        lines.append(f"- `{record['mechanism_id']}`: {record['rationale']}")
    lines += [
        "",
        "## Blocked mechanisms",
        "",
    ]
    for record in blocked:
        lines.append(f"- `{record['mechanism_id']}`: {record['rationale']}")
    lines += [
        "",
        "## Reproducibility",
        "",
        "```",
        "python -m scripts.publish_exp_sr_disposition --check",
        "```",
        "",
        "## Limitations",
        "",
        "- Every measured result here is fixture-scale: small in-process "
        "corpora, mocked external judges, synthetic human raters, seeds "
        "0-2 at most. None of these dispositions is a statistically-"
        "powered, production-scale claim.",
        "- exp-sr-3 (evaluator calibration) and exp-sr-11 (PySR/SRBench "
        "benchmark) both remain genuinely open: neither has been measured "
        "against real external evidence (real judge/human raters; a real "
        "external tool run). Follow-up issues are filed for both rather "
        "than treating the mock/blocked results as final answers.",
        "- This report does not itself change any production default, "
        "champion pointer, or checkpoint -- every disposed mechanism's "
        "`default_state` stays `off`/`n/a`, matching each source "
        "experiment's own explicit lever state.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(f"docs/design/exp-sr-disposition-{EVIDENCE_CUTOFF.replace('-', '')}.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(f"docs/design/exp-sr-disposition-{EVIDENCE_CUTOFF.replace('-', '')}.md"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    report = build_disposition_report(build_records())
    if not disposition_report_integrity_ok(report):
        raise RuntimeError("disposition report failed its own integrity check")

    report_dict = report.to_dict()
    expected_json = json.dumps(report_dict, indent=2) + "\n"
    expected_md = _markdown(report_dict, report.to_markdown())

    if args.check:
        if not args.out_json.is_file() or args.out_json.read_text(encoding="utf-8") != expected_json:
            print(f"stale disposition JSON: {args.out_json}")
            return 1
        if not args.out_md.is_file() or args.out_md.read_text(encoding="utf-8") != expected_md:
            print(f"stale disposition Markdown: {args.out_md}")
            return 1
        print(f"ok {report.report_hash}")
        return 0

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(expected_json, encoding="utf-8")
    args.out_md.write_text(expected_md, encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"report_hash={report.report_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
