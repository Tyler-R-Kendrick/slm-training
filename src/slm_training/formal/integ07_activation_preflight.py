"""INTEG-07 / SLM-561: activation-preflight recall + false-skip benchmark.

Replays frozen known-activating / provably-inactive / unknown / adversarial
cases through the existing INTEG-02 ``admit_mechanism_treatment`` seam
(KERN-08 certificates). No parallel admission path.

A false skip of a known activating case is release-blocking. Unknown and
identity-mismatched evidence must effective-run; beyond-fixture skips require
exact corpus / trigger-theorem / cost-model identity match, and bind the live
HARN-11 hermetic corpus digest into the report.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.autoresearch.preflight.mechanism_no_effect import (
    ADMISSION_SCHEMA,
    admit_mechanism_treatment,
)
from slm_training.formal.mechanism_trigger import (
    DECODE_UNIT_WORK,
    NEURAL_FORWARD_ONLY,
    NamedCostModel,
    THEOREM_NECESSARY,
)
from slm_training.harnesses.reasoning.revmath.corpus import (
    load_fixture_corpus_manifest,
)
from slm_training.versioning import build_version_stamp

REPLAY_SCHEMA = "integ07_activation_preflight_replay/v1"
REPORT_SCHEMA = "integ07_activation_preflight_recall_report/v1"
REPLAY_RELPATH = (
    "src/slm_training/resources/formal/integ07_activation_preflight_replay.v1.json"
)
DEFAULT_RESULTS_RELPATH = "docs/design/integ-07-activation-preflight-recall-results.json"

_REPO = Path(__file__).resolve().parents[3]

_COST_MODELS = {
    NEURAL_FORWARD_ONLY.id: NEURAL_FORWARD_ONLY,
    DECODE_UNIT_WORK.id: DECODE_UNIT_WORK,
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    cohort: str
    ground_truth: str
    expect_disposition: str
    raw_disposition: str
    effective_disposition: str
    identity_ok: bool
    ok: bool
    false_skip: bool
    theorem_ref: str | None
    corpus_id: str
    locked_corpus_id: str
    certificate_sha256: str | None
    treatment_compute_units: int
    compute_saved: int
    reasons: tuple[str, ...]
    adversarial_kind: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "cohort": self.cohort,
            "ground_truth": self.ground_truth,
            "expect_disposition": self.expect_disposition,
            "raw_disposition": self.raw_disposition,
            "effective_disposition": self.effective_disposition,
            "identity_ok": self.identity_ok,
            "ok": self.ok,
            "false_skip": self.false_skip,
            "theorem_ref": self.theorem_ref,
            "corpus_id": self.corpus_id,
            "locked_corpus_id": self.locked_corpus_id,
            "certificate_sha256": self.certificate_sha256,
            "treatment_compute_units": self.treatment_compute_units,
            "compute_saved": self.compute_saved,
            "reasons": list(self.reasons),
            "adversarial_kind": self.adversarial_kind,
            "detail": self.detail,
        }


def replay_path(*, root: Path | None = None) -> Path:
    return (root or _REPO) / REPLAY_RELPATH


def load_replay(*, root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(replay_path(root=root).read_text(encoding="utf-8"))
    if payload.get("schema") != REPLAY_SCHEMA:
        raise ValueError(f"expected schema {REPLAY_SCHEMA!r}")
    return payload


def _cost_model(case: Mapping[str, Any]) -> NamedCostModel:
    mid = str(case.get("cost_model") or NEURAL_FORWARD_ONLY.id)
    version = int(case.get("cost_model_version") or 1)
    base = _COST_MODELS.get(mid)
    if base is None:
        return NamedCostModel(mid, version)
    if version == base.version:
        return base
    return NamedCostModel(mid, version)


def identity_matches(
    *,
    case: Mapping[str, Any],
    admission_corpus_id: str,
    admission_theorem_ref: str | None,
    admission_mechanism_id: str,
    cost_model: NamedCostModel,
) -> tuple[bool, str]:
    """Beyond-fixture skips require exact locked corpus/theorem/model identity."""

    locked_corpus = str(case.get("locked_corpus_id") or case.get("corpus_id") or "")
    if admission_corpus_id != locked_corpus:
        return (
            False,
            f"corpus mismatch: admission={admission_corpus_id!r} "
            f"locked={locked_corpus!r}",
        )
    expected_mech = str(case.get("mechanism_id") or "")
    if admission_mechanism_id != expected_mech:
        return (
            False,
            f"mechanism mismatch: admission={admission_mechanism_id!r} "
            f"expected={expected_mech!r}",
        )
    expected_thm = case.get("expected_theorem_ref", case.get("theorem_ref"))
    if expected_thm is None:
        if admission_theorem_ref is not None:
            return False, "expected null theorem_ref but admission carried one"
    else:
        if admission_theorem_ref != str(expected_thm):
            return (
                False,
                f"theorem mismatch: admission={admission_theorem_ref!r} "
                f"expected={expected_thm!r}",
            )
    expected_ver = case.get("expected_cost_model_version")
    if expected_ver is not None and int(cost_model.version) != int(expected_ver):
        return (
            False,
            f"cost-model version mismatch: admission={cost_model.version} "
            f"expected={expected_ver}",
        )
    expected_model = case.get("cost_model")
    if expected_model is not None and cost_model.id != str(expected_model):
        return (
            False,
            f"cost-model id mismatch: admission={cost_model.id!r} "
            f"expected={expected_model!r}",
        )
    return True, "identity ok"


def effective_disposition(
    *,
    raw_disposition: str,
    identity_ok: bool,
) -> str:
    """Skip is only effective when raw skip binds exact locked identities."""

    if raw_disposition == "skip_no_effect" and identity_ok:
        return "skip_no_effect"
    return "run"


def evaluate_case(
    case: Mapping[str, Any],
    *,
    revmath_corpus_id: str,
    revmath_corpus_sha256: str,
) -> CaseResult:
    model = _cost_model(case)
    theorem_arg = case.get("theorem_ref")
    kwargs: dict[str, Any] = {
        "mechanism_id": str(case["mechanism_id"]),
        "corpus_id": str(case["corpus_id"]),
        "observations": list(case["observations"]),
        "scan_complete": bool(case.get("scan_complete", False)),
        "cost_model": model,
        "claim_class": "fixture",
        "campaign_id": "camp.integ07",
        "experiment_id": f"exp.integ07.{case['id']}",
        "manifest_sha256": revmath_corpus_sha256,
    }
    if theorem_arg is not None:
        kwargs["theorem_ref"] = str(theorem_arg)
    else:
        kwargs["theorem_ref"] = THEOREM_NECESSARY

    admission = admit_mechanism_treatment(**kwargs)
    id_ok, id_detail = identity_matches(
        case=case,
        admission_corpus_id=admission.corpus_id,
        admission_theorem_ref=admission.theorem_ref,
        admission_mechanism_id=admission.mechanism_id,
        cost_model=model,
    )
    # Bind HARN-11 live digest when the case claims the hermetic corpus.
    if str(case.get("locked_corpus_id")) == revmath_corpus_id:
        if str(case.get("corpus_id")) != revmath_corpus_id:
            id_ok = False
            id_detail = (
                f"corpus mismatch vs live HARN-11 {revmath_corpus_id!r}: "
                f"admission corpus_id={admission.corpus_id!r}"
            )

    raw = admission.disposition
    effective = effective_disposition(raw_disposition=raw, identity_ok=id_ok)
    expect = str(case["expect_disposition"])
    units = int(case.get("treatment_compute_units") or 0)
    false_skip = (
        str(case.get("ground_truth")) == "activating" and effective == "skip_no_effect"
    )
    ok = effective == expect and not false_skip
    saved = units if effective == "skip_no_effect" else 0
    detail_parts = [id_detail, *admission.reasons]
    if case.get("bind_revmath_entry_id"):
        detail_parts.append(
            f"revmath_entry={case['bind_revmath_entry_id']} "
            f"corpus_sha256={revmath_corpus_sha256[:12]}"
        )
    return CaseResult(
        case_id=str(case["id"]),
        cohort=str(case["cohort"]),
        ground_truth=str(case["ground_truth"]),
        expect_disposition=expect,
        raw_disposition=raw,
        effective_disposition=effective,
        identity_ok=id_ok,
        ok=ok,
        false_skip=false_skip,
        theorem_ref=admission.theorem_ref,
        corpus_id=admission.corpus_id,
        locked_corpus_id=str(case.get("locked_corpus_id") or case["corpus_id"]),
        certificate_sha256=(
            admission.evidence_lineage.get("certificate_sha256")
            if admission.evidence_lineage
            else None
        ),
        treatment_compute_units=units,
        compute_saved=saved,
        reasons=tuple(admission.reasons),
        adversarial_kind=(
            str(case["adversarial_kind"]) if case.get("adversarial_kind") else None
        ),
        detail="; ".join(detail_parts),
    )


def _rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 1.0
    return numer / denom


def run_benchmark(*, root: Path | None = None) -> dict[str, Any]:
    base = root or _REPO
    replay = load_replay(root=base)
    manifest = load_fixture_corpus_manifest()
    revmath_id = str(manifest.corpus_id)
    revmath_sha = str(manifest.corpus_sha256)

    declared = replay.get("revmath_corpus") or {}
    if str(declared.get("corpus_id")) != revmath_id:
        raise ValueError(
            f"replay locked revmath corpus_id {declared.get('corpus_id')!r} "
            f"!= live HARN-11 {revmath_id!r}"
        )

    # Positive/negative fixture presence (hermetic corpus coverage).
    entry_ids = {entry.entry_id for entry in manifest.entries}
    missing_pos = [
        eid
        for eid in declared.get("positive_entry_ids") or []
        if eid not in entry_ids
    ]
    missing_neg = [
        eid
        for eid in declared.get("negative_entry_ids") or []
        if eid not in entry_ids
    ]
    if missing_pos or missing_neg:
        raise ValueError(
            f"HARN-11 fixture gaps positive={missing_pos} negative={missing_neg}"
        )

    results = [
        evaluate_case(
            case,
            revmath_corpus_id=revmath_id,
            revmath_corpus_sha256=revmath_sha,
        )
        for case in replay["cases"]
    ]

    activating = [r for r in results if r.cohort == "known_activating"]
    inactive = [r for r in results if r.cohort == "provably_inactive"]
    unknown = [r for r in results if r.cohort == "unknown_fallback"]
    adversarial = [r for r in results if r.cohort == "adversarial"]

    activating_run = sum(1 for r in activating if r.effective_disposition == "run")
    inactive_skip = sum(
        1 for r in inactive if r.effective_disposition == "skip_no_effect"
    )
    fallback_cases = unknown + [
        r
        for r in adversarial
        if r.ground_truth in {"unknown", "activating"}
        or r.adversarial_kind
        in {
            "incomplete_scan",
            "stale_trigger",
            "corpus_mismatch",
            "mechanism_version_mismatch",
        }
    ]
    fallback_run = sum(1 for r in fallback_cases if r.effective_disposition == "run")
    false_skips = [r.case_id for r in results if r.false_skip]
    compute_saved = sum(r.compute_saved for r in results)
    compute_eligible = sum(r.treatment_compute_units for r in inactive)

    recall = _rate(activating_run, len(activating))
    specificity = _rate(inactive_skip, len(inactive))
    unknown_fallback_rate = _rate(fallback_run, len(fallback_cases))

    required_adv = {
        "incomplete_scan",
        "stale_trigger",
        "corpus_mismatch",
        "mechanism_version_mismatch",
    }
    seen_adv = {r.adversarial_kind for r in adversarial if r.adversarial_kind}
    missing_adv = sorted(required_adv - seen_adv)

    recall_min = float(
        (replay.get("acceptance") or {}).get("known_activating_recall_min", 1.0)
    )
    passed = (
        recall >= recall_min
        and not false_skips
        and all(r.ok for r in results)
        and not missing_adv
        and unknown_fallback_rate == 1.0
    )

    theorem_identities = sorted(
        {
            r.theorem_ref
            for r in results
            if r.theorem_ref and r.effective_disposition == "skip_no_effect"
        }
    )
    corpus_identities = sorted({r.locked_corpus_id for r in results})

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "integ": "INTEG-07",
        "issue": "SLM-561",
        "admission_schema": ADMISSION_SCHEMA,
        "replay_schema": REPLAY_SCHEMA,
        "passed": passed,
        "metrics": {
            "known_activating_recall": recall,
            "known_activating_n": len(activating),
            "provably_inactive_specificity": specificity,
            "provably_inactive_n": len(inactive),
            "unknown_fallback_rate": unknown_fallback_rate,
            "unknown_fallback_n": len(fallback_cases),
            "compute_saved": compute_saved,
            "compute_eligible_inactive": compute_eligible,
            "false_skip_count": len(false_skips),
            "false_skip_case_ids": false_skips,
        },
        "acceptance": {
            "known_activating_recall_min": recall_min,
            "recall_gate_passed": recall >= recall_min and not false_skips,
            "unknown_must_run": unknown_fallback_rate == 1.0,
            "adversarial_kinds_required": sorted(required_adv),
            "adversarial_kinds_seen": sorted(x for x in seen_adv if x),
            "adversarial_kinds_missing": missing_adv,
        },
        "identities": {
            "revmath_corpus_id": revmath_id,
            "revmath_corpus_sha256": revmath_sha,
            "trigger_theorem_refs": theorem_identities,
            "locked_corpus_ids": corpus_identities,
            "kernel": replay.get("kernel"),
            "admission": replay.get("admission"),
        },
        "cases": [r.to_dict() for r in results],
        "cohort_counts": {
            "known_activating": len(activating),
            "provably_inactive": len(inactive),
            "unknown_fallback": len(unknown),
            "adversarial": len(adversarial),
        },
    }
    report["version_stamp"] = build_version_stamp("formal.objects")
    return report


def assert_benchmark_passed(report: Mapping[str, Any]) -> None:
    if not report.get("passed"):
        metrics = report.get("metrics") or {}
        raise AssertionError(
            "INTEG-07 activation-preflight recall gate failed: "
            f"recall={metrics.get('known_activating_recall')!r} "
            f"false_skips={metrics.get('false_skip_case_ids')!r} "
            f"acceptance={report.get('acceptance')!r}"
        )
    metrics = report["metrics"]
    if float(metrics["known_activating_recall"]) < 1.0:
        raise AssertionError("known-activating recall must be 100%")
    if metrics.get("false_skip_count"):
        raise AssertionError(
            f"false skips are release-blocking: {metrics.get('false_skip_case_ids')}"
        )


def write_report(
    report: Mapping[str, Any],
    *,
    path: Path | None = None,
    root: Path | None = None,
) -> Path:
    target = path or ((root or _REPO) / DEFAULT_RESULTS_RELPATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def required_adversarial_kinds(replay: Mapping[str, Any] | None = None) -> set[str]:
    _ = replay
    return {
        "incomplete_scan",
        "stale_trigger",
        "corpus_mismatch",
        "mechanism_version_mismatch",
    }


__all__ = [
    "DEFAULT_RESULTS_RELPATH",
    "REPLAY_RELPATH",
    "REPLAY_SCHEMA",
    "REPORT_SCHEMA",
    "assert_benchmark_passed",
    "effective_disposition",
    "evaluate_case",
    "identity_matches",
    "load_replay",
    "replay_path",
    "required_adversarial_kinds",
    "run_benchmark",
    "write_report",
]
