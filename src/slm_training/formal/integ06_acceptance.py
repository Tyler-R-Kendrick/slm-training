"""INTEG-06 / SLM-573: adversarial end-to-end acceptance orchestration.

Assembles existing EVID-11 / KERN-07/08/11 / HARN-09/11 / INTEG-01/02/03/05
gates into one release-blocking matrix. Runners call live public APIs and
canonical surfaces (including ``run_revmath_profile`` / CampaignStore) —
no third evidence stack and no test-only shortcuts.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.autoresearch.campaign_formal_evidence import (
    formal_status_from_authority,
)
from slm_training.autoresearch.preflight.mechanism_no_effect import (
    admit_mechanism_treatment,
)
from slm_training.formal import mutation_suite as evid11
from slm_training.formal.judgment import ClassificationInputsV1, classify
from slm_training.formal.mechanism_trigger import (
    DECODE_UNIT_WORK,
    NEURAL_FORWARD_ONLY,
    SINGLETON_BYPASS,
    Observation,
    TriggerEvidence,
    emit_dominance_certificate,
    observation_from_singleton_bypass,
)
from slm_training.formal.refutation_authority import (
    BindingIdsV1,
    evidence_authorizes_removal,
    make_exact_replay_evidence,
)
from slm_training.formal.singleton_neural_policy import (
    SingletonProofV1,
    mutation_rejects_incomplete_singleton_domain,
    observe_singleton_runtime,
    singleton_bypass_neural_trace,
    singleton_policy_neural_cost_zero,
    singleton_proof_valid,
)
from slm_training.formal.trace_refinement import (
    build_supported_fixture_trace,
    check_trace_refinement,
    inject_illegal_commit,
    inject_omitted_model_forwards,
    inject_reordered_events,
    inject_stale_state_domain,
    reseal_trace,
)
from slm_training.harnesses.reasoning.revmath.corpus import (
    demonstrate_mutation_failures,
    load_fixture_corpus_manifest,
)
from slm_training.harnesses.reasoning.revmath.profile_binding import (
    RevmathProfilePlanV1,
    run_revmath_profile,
)
from slm_training.harnesses.reasoning.revmath.schemas import parse_revmath_task
from slm_training.harnesses.reasoning.revmath.self_healing import (
    reject_forbidden_repair,
)
from slm_training.models.decode_stats import DecodeStats
from slm_training.versioning import build_version_stamp

MATRIX_SCHEMA = "integ06_adversarial_acceptance_matrix/v1"
REPORT_SCHEMA = "integ06_adversarial_acceptance_report/v1"
MATRIX_RELPATH = (
    "src/slm_training/resources/formal/integ06_adversarial_matrix.v1.json"
)
DEFAULT_RESULTS_RELPATH = "docs/design/integ-06-adversarial-acceptance-results.json"

_REPO = Path(__file__).resolve().parents[3]
_REVMATH_FIXTURES = _REPO / "src/slm_training/resources/revmath/fixtures"

# Frozen repair mutations that must never become self-healing authority.
_FORBIDDEN_SELF_HEAL = (
    "weaken_theorem",
    "add_axiom",
    "increase_budget",
    "change_corpus",
    "proposition",
    "allowed_assumption_ids",
    "max_wall_seconds",
    "corpus_sha256",
)

# HARN-11 corpus classes covering forward / reversal / counterexample /
# quantitative-bound / unknown / invalid semantic slots (SLM-573 S10).
_REQUIRED_REVMATH_OUTCOMES = {
    "constructivization_success": "witnessed",  # forward
    "counterexample_checked": "refuted",
    "quantitative_bound_success": "witnessed",
    "timeout": "unknown",
    "malformed_evidence": "invalid",
}
_REQUIRED_REVMATH_PRESENCE = (
    "reversal_equivalence",  # reversal (report-shaped; no single outcome tag)
    "ablation_success",  # forward ablation family
)


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    scenario: str
    gate: str
    layer: str
    expect_reject: bool
    rejected: bool
    detail: str
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "scenario": self.scenario,
            "gate": self.gate,
            "layer": self.layer,
            "expect_reject": self.expect_reject,
            "rejected": self.rejected,
            "detail": self.detail,
            "ok": self.ok,
        }


def matrix_path(*, root: Path | None = None) -> Path:
    base = root or _REPO
    return base / MATRIX_RELPATH


def load_matrix(*, root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(matrix_path(root=root).read_text(encoding="utf-8"))
    if payload.get("schema") != MATRIX_SCHEMA:
        raise ValueError(f"expected schema {MATRIX_SCHEMA!r}")
    return payload


def _reject(detail: str) -> tuple[bool, str]:
    """Adversarial path: gate fired (mutation rejected) → rejected=True."""

    return True, detail


def _accept(detail: str) -> tuple[bool, str]:
    """Positive path success, or adversarial leak (rejected=False)."""

    return False, detail


def _sha(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# --- S01 singleton ---------------------------------------------------------


def pos_complete_singleton_zero_neural() -> tuple[bool, str]:
    proof = SingletonProofV1(status="complete", candidates=((9,),), forced_token=9)
    if not singleton_proof_valid(proof):
        return _reject("complete singleton failed validity")
    if not singleton_policy_neural_cost_zero(proof):
        return _reject("complete singleton policy cost != 0")
    if singleton_bypass_neural_trace(proof):
        return _reject("bypass trace must be empty (zero neural work)")
    stats = DecodeStats(forwards_count=0, tokens_emitted=1, dfa_sync_count=1)
    observed = observe_singleton_runtime(proof, stats)
    if not (
        observed.proves_zero_neural_work
        and observed.observed_forwards_count == 0
        and observed.projected_neural_cost == 0
    ):
        return _reject(f"runtime refinement mismatch: {observed}")
    return _accept("complete singleton → zero neural work + matching runtime")


def adv_incomplete_singleton_must_not_force() -> tuple[bool, str]:
    if not mutation_rejects_incomplete_singleton_domain():
        return _accept("incomplete singleton mutation helper failed")
    proof = SingletonProofV1(
        status="incomplete", candidates=((0,),), forced_token=0
    )
    if singleton_proof_valid(proof) or singleton_policy_neural_cost_zero(proof):
        return _accept("incomplete singleton incorrectly forced / zero-cost")
    return _reject("incomplete singleton correctly refused force")


# --- S02 refutation --------------------------------------------------------


def pos_exact_replay_authorizes_removal() -> tuple[bool, str]:
    return evid11.positive_exact_replay_authorizes()


def adv_timeout_skipped_incomplete_preserve_candidate() -> tuple[bool, str]:
    # Reuse EVID-11 gate: timeout/skipped/missing must not authorize removal.
    return evid11.mutation_timeout_missing_tool_strip_evidence()


# --- S03 encoding ----------------------------------------------------------


def pos_honest_encoding_authorizes() -> tuple[bool, str]:
    return evid11.positive_honest_encoding_authorizes()


def adv_valid_cert_wrong_encoding_rejected() -> tuple[bool, str]:
    return evid11.mutation_certificate_wrong_encoding()


# --- S04 binding -----------------------------------------------------------


def pos_sealed_binding_holds() -> tuple[bool, str]:
    return evid11.positive_sealed_binding_holds()


def adv_same_name_changed_proposition_rejected() -> tuple[bool, str]:
    return evid11.mutation_same_name_proposition_to_true()


# --- S05 drift -------------------------------------------------------------


def adv_source_toolchain_axiom_drift_rejected() -> tuple[bool, str]:
    details: list[str] = []
    for name, runner in (
        ("toolchain", evid11.mutation_toolchain_lock_drift),
        ("source_tree", evid11.mutation_source_tree_drift),
        ("axiom", evid11.mutation_unexpected_axiom),
    ):
        rejected, detail = runner()
        if not rejected:
            return _accept(f"{name} drift leaked: {detail}")
        details.append(f"{name}:{detail}")
    return _reject("; ".join(details))


# --- S06 runtime trace -----------------------------------------------------


def pos_frozen_trace_refines() -> tuple[bool, str]:
    trace = build_supported_fixture_trace()
    result = check_trace_refinement(
        trace,
        fixture_id="bundle_stats_mechanism",
        expected_cost=28,
        expected_forwards=11,
    )
    if result.accepted and result.reject_reason is None:
        return _accept("frozen INTEG-01/KERN-11 fixture refines")
    return _reject(f"frozen fixture failed: {result.reject_reason}")


def adv_illegal_reorder_stale_hidden_forward_rejected() -> tuple[bool, str]:
    injectors = (
        ("illegal_commit", inject_illegal_commit),
        ("reordered_events", inject_reordered_events),
        ("stale_state_domain", inject_stale_state_domain),
        ("omitted_model_forwards", inject_omitted_model_forwards),
    )
    details: list[str] = []
    base = build_supported_fixture_trace()
    for expected_reason, injector in injectors:
        mutated = reseal_trace(injector(base))
        result = check_trace_refinement(
            mutated,
            fixture_id="bundle_stats_mechanism",
            expected_cost=28,
            expected_forwards=11,
        )
        if result.accepted or result.reject_reason != expected_reason:
            return _accept(
                f"{expected_reason} leaked: accepted={result.accepted} "
                f"reason={result.reject_reason!r}"
            )
        details.append(expected_reason)
    return _reject("rejected:" + ",".join(details))


# --- S07 self-healing ------------------------------------------------------


def adv_forbidden_self_heal_mutations_rejected() -> tuple[bool, str]:
    details: list[str] = []
    for alias in _FORBIDDEN_SELF_HEAL:
        try:
            reject_forbidden_repair(alias)
        except Exception as exc:  # RevmathSchemaError — fail-closed is success
            details.append(f"{alias}:{type(exc).__name__}")
            continue
        return _accept(f"forbidden repair {alias!r} was allowed")
    return _reject("; ".join(details))


# --- S08 trigger preflight -------------------------------------------------


def _absent_obs(*, input_id: str = "integ06/absent"):
    return observation_from_singleton_bypass(
        input_id=input_id,
        is_singleton=False,
        baseline_output=1,
        enabled_output=1,
        baseline_cost=2,
        enabled_cost=2,
    )


def pos_absent_trigger_skips_treatment() -> tuple[bool, str]:
    admission = admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="integ06/s08",
        observations=[_absent_obs()],
        scan_complete=True,
        campaign_id="camp.integ06.s08",
        experiment_id="exp.integ06.s08",
    )
    if admission.disposition != "skip_no_effect":
        return _reject(f"expected skip_no_effect, got {admission.disposition}")
    return _accept("absent trigger → skip_no_effect")


def adv_unknown_incomplete_trigger_runs_treatment() -> tuple[bool, str]:
    unknown = Observation(
        input_id="integ06/unknown",
        baseline_output=1,
        enabled_output=1,
        baseline_cost=2,
        enabled_cost=2,
        trigger=TriggerEvidence.UNKNOWN,
    )
    incomplete = admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="integ06/s08-incomplete",
        observations=[_absent_obs(input_id="integ06/incomplete")],
        scan_complete=False,
        campaign_id="camp.integ06.s08",
        experiment_id="exp.integ06.s08u",
    )
    unknown_adm = admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="integ06/s08-unknown",
        observations=[unknown],
        scan_complete=True,
        campaign_id="camp.integ06.s08",
        experiment_id="exp.integ06.s08u2",
    )
    if incomplete.disposition != "run" or unknown_adm.disposition != "run":
        return _accept(
            f"unsafe skip: incomplete={incomplete.disposition} "
            f"unknown={unknown_adm.disposition}"
        )
    return _reject("unknown/incomplete correctly force run")


# --- S09 dominance identity ------------------------------------------------


def pos_dominance_certificate_emits() -> tuple[bool, str]:
    obs = observation_from_singleton_bypass(
        input_id="integ06/dom",
        is_singleton=False,
        baseline_output=1,
        enabled_output=1,
        baseline_cost=2,
        enabled_cost=5,
    )
    cert = emit_dominance_certificate(
        mechanism=SINGLETON_BYPASS,
        cost_model=NEURAL_FORWARD_ONLY,
        corpus_id="integ06/dom",
        corpus=[obs],
    )
    payload = cert.to_dict()
    if not payload.get("dominates_locally"):
        return _reject("dominance certificate missing dominates_locally")
    if payload.get("corpus_id") != "integ06/dom":
        return _reject("dominance corpus_id mismatch")
    if payload.get("cost_model") != NEURAL_FORWARD_ONLY.tag:
        return _reject("dominance cost_model mismatch")
    return _accept(f"dominance sealed digest={_sha(payload)[:16]}")


def _dominance_identity_ok(
    sealed: Mapping[str, Any],
    *,
    corpus_id: str,
    cost_model_tag: str,
    corpus_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if sealed.get("corpus_id") != corpus_id:
        return False
    if sealed.get("cost_model") != cost_model_tag:
        return False
    live_corpus = [dict(row) for row in corpus_rows]
    sealed_corpus = list(sealed.get("corpus") or [])
    return sealed_corpus == live_corpus


def adv_dominance_identity_drift_invalidates() -> tuple[bool, str]:
    obs = observation_from_singleton_bypass(
        input_id="integ06/dom-drift",
        is_singleton=False,
        baseline_output=1,
        enabled_output=1,
        baseline_cost=2,
        enabled_cost=5,
    )
    cert = emit_dominance_certificate(
        mechanism=SINGLETON_BYPASS,
        cost_model=NEURAL_FORWARD_ONLY,
        corpus_id="integ06/dom-sealed",
        corpus=[obs],
    )
    sealed = cert.to_dict()
    drifts = (
        ("corpus_id", "integ06/dom-other", NEURAL_FORWARD_ONLY.tag, [obs.to_dict()]),
        ("cost_model", "integ06/dom-sealed", DECODE_UNIT_WORK.tag, [obs.to_dict()]),
        (
            "corpus_row",
            "integ06/dom-sealed",
            NEURAL_FORWARD_ONLY.tag,
            [{**obs.to_dict(), "baseline_cost": 99}],
        ),
    )
    details: list[str] = []
    for label, cid, model_tag, rows in drifts:
        if _dominance_identity_ok(
            sealed, corpus_id=cid, cost_model_tag=model_tag, corpus_rows=rows
        ):
            return _accept(f"dominance identity drift {label!r} still matched")
        details.append(label)
    if not _dominance_identity_ok(
        sealed,
        corpus_id="integ06/dom-sealed",
        cost_model_tag=NEURAL_FORWARD_ONLY.tag,
        corpus_rows=[obs.to_dict()],
    ):
        return _accept("honest dominance identity incorrectly invalidated")
    return _reject("invalidated:" + ",".join(details))


# --- S10 revmath classifications -------------------------------------------


def _formal_status_for_outcome(expected: str) -> tuple[str, str]:
    """Return (formal_status, judgment_outcome) via campaign mapping surface."""

    if expected == "witnessed":
        judgment = classify(
            ClassificationInputsV1(
                vss_verdict="supported",
                has_witness_digest=True,
                replay_checked=True,
                replay_ok=True,
            ),
            checked=True,
        )
        status = formal_status_from_authority(
            {
                "authority_class": "checked_semantic",
                "judgment": judgment.to_dict(),
            }
        )
        return status, judgment.outcome.value
    if expected == "refuted":
        binding = BindingIdsV1(
            state_id="s10",
            problem_id="s10",
            source_id="vss",
            tool_id="python_replay",
        )
        evidence = make_exact_replay_evidence(binding, evidence_digest="s10")
        if not evidence_authorizes_removal(evidence, binding):
            raise AssertionError("refutation evidence failed authorization")
        judgment = classify(
            ClassificationInputsV1(
                vss_verdict="unsupported",
                exhausted=True,
                replay_checked=True,
                replay_ok=True,
                refutation_evidence=evidence,
                expected_binding=binding,
            ),
            checked=True,
        )
        status = formal_status_from_authority(
            {
                "authority_class": "checked_semantic",
                "judgment": judgment.to_dict(),
            }
        )
        return status, judgment.outcome.value
    if expected == "unknown":
        judgment = classify(
            ClassificationInputsV1(vss_verdict="unsupported", timed_out=True),
            checked=True,
        )
        status = formal_status_from_authority(
            {
                "authority_class": "checked_semantic",
                "judgment": judgment.to_dict(),
            }
        )
        return status, judgment.outcome.value
    # invalid
    judgment = classify(
        ClassificationInputsV1(malformed_input=True),
        checked=True,
    )
    status = formal_status_from_authority(
        {
            "authority_class": "invalid",
            "judgment": judgment.to_dict(),
        }
    )
    return status, judgment.outcome.value


def _formal_status_for_outcome(expected: str) -> tuple[str, str]:
    """Return (formal_status, judgment_outcome) via campaign mapping surface."""

    if expected == "witnessed":
        judgment = classify(
            ClassificationInputsV1(
                vss_verdict="supported",
                has_witness_digest=True,
                replay_checked=True,
                replay_ok=True,
            ),
            checked=True,
        )
        status = formal_status_from_authority(
            {
                "authority_class": "checked_semantic",
                "judgment": judgment.to_dict(),
            }
        )
        return status, judgment.outcome.value
    if expected == "refuted":
        binding = BindingIdsV1(
            state_id="s10",
            problem_id="s10",
            source_id="vss",
            tool_id="python_replay",
        )
        evidence = make_exact_replay_evidence(binding, evidence_digest="s10")
        if not evidence_authorizes_removal(evidence, binding):
            raise AssertionError("refutation evidence failed authorization")
        judgment = classify(
            ClassificationInputsV1(
                vss_verdict="unsupported",
                exhausted=True,
                replay_checked=True,
                replay_ok=True,
                refutation_evidence=evidence,
                expected_binding=binding,
            ),
            checked=True,
        )
        status = formal_status_from_authority(
            {
                "authority_class": "checked_semantic",
                "judgment": judgment.to_dict(),
            }
        )
        return status, judgment.outcome.value
    if expected == "unknown":
        judgment = classify(
            ClassificationInputsV1(vss_verdict="unsupported", timed_out=True),
            checked=True,
        )
        status = formal_status_from_authority(
            {
                "authority_class": "checked_semantic",
                "judgment": judgment.to_dict(),
            }
        )
        return status, judgment.outcome.value
    judgment = classify(
        ClassificationInputsV1(malformed_input=True),
        checked=True,
    )
    status = formal_status_from_authority(
        {
            "authority_class": "invalid",
            "judgment": judgment.to_dict(),
        }
    )
    return status, judgment.outcome.value


def pos_revmath_classes_through_campaign() -> tuple[bool, str]:
    manifest = load_fixture_corpus_manifest()
    by_class = {e.corpus_class: e for e in manifest.entries}
    missing = [
        name
        for name, expected in _REQUIRED_REVMATH_OUTCOMES.items()
        if name not in by_class or by_class[name].expected_outcome != expected
    ]
    missing.extend(
        name for name in _REQUIRED_REVMATH_PRESENCE if name not in by_class
    )
    if missing:
        return _reject(f"corpus class/outcome mismatch: {missing}")

    outcome_to_authority = {
        "witnessed": "proved",
        "refuted": "refuted",
        "unknown": "unknown",
        "invalid": "invalid",
    }
    for name, expected in _REQUIRED_REVMATH_OUTCOMES.items():
        try:
            status, outcome = _formal_status_for_outcome(expected)
        except AssertionError as exc:
            return _reject(str(exc))
        want = outcome_to_authority[expected]
        if status != want or outcome != expected:
            return _reject(
                f"{name}: formal_status={status!r} outcome={outcome!r} "
                f"want status={want!r} outcome={expected!r}"
            )

    task_path = _REVMATH_FIXTURES / "ablation_necessary.task.json"
    task = parse_revmath_task(json.loads(task_path.read_text(encoding="utf-8")))
    with tempfile.TemporaryDirectory(prefix="integ06-s10-") as tmp:
        store_root = Path(tmp) / "campaigns"
        plan = RevmathProfilePlanV1(
            campaign_id=task.campaign.campaign_id,
            experiment_id="exp.integ06.s10.profile",
            claim_class="fixture",
        )
        profile = run_revmath_profile(
            [task],
            store_root=store_root,
            plan=plan,
            hermetic=True,
        )
        if not profile.manifest_sha256:
            return _reject("profile missing manifest_sha256")
        if profile.disposition.get("evidence_class") != "fixture":
            return _reject(f"unexpected disposition: {profile.disposition}")
        store_path = Path(profile.store_root)
        if not (store_path / "decode_stats.jsonl").is_file():
            return _reject("campaign store missing decode_stats.jsonl")
        digest = profile.manifest_sha256[:16]
    return _accept(
        f"classes retained; campaign manifest={digest} "
        f"corpus={manifest.corpus_sha256[:16]}"
    )


def adv_revmath_class_mutation_rejected() -> tuple[bool, str]:
    manifest = load_fixture_corpus_manifest()
    mutation_entries = [
        e for e in manifest.entries if e.corpus_class.startswith("mutation_")
    ]
    if not mutation_entries:
        return _accept("no HARN-11 mutation entries in corpus")
    details: list[str] = []
    for entry in mutation_entries:
        hits = demonstrate_mutation_failures(entry)
        if not hits:
            return _accept(f"{entry.corpus_class} mutation did not hit a gate")
        details.append(f"{entry.corpus_class}:{len(hits)}")
    return _reject("; ".join(details))


# --- suite runner ----------------------------------------------------------


_RUNNERS: dict[str, Callable[[], tuple[bool, str]]] = {
    "pos_complete_singleton_zero_neural": pos_complete_singleton_zero_neural,
    "pos_exact_replay_authorizes_removal": pos_exact_replay_authorizes_removal,
    "pos_honest_encoding_authorizes": pos_honest_encoding_authorizes,
    "pos_sealed_binding_holds": pos_sealed_binding_holds,
    "pos_frozen_trace_refines": pos_frozen_trace_refines,
    "pos_absent_trigger_skips_treatment": pos_absent_trigger_skips_treatment,
    "pos_dominance_certificate_emits": pos_dominance_certificate_emits,
    "pos_revmath_classes_through_campaign": pos_revmath_classes_through_campaign,
    "adv_incomplete_singleton_must_not_force": adv_incomplete_singleton_must_not_force,
    "adv_timeout_skipped_incomplete_preserve_candidate": (
        adv_timeout_skipped_incomplete_preserve_candidate
    ),
    "adv_valid_cert_wrong_encoding_rejected": adv_valid_cert_wrong_encoding_rejected,
    "adv_same_name_changed_proposition_rejected": (
        adv_same_name_changed_proposition_rejected
    ),
    "adv_source_toolchain_axiom_drift_rejected": (
        adv_source_toolchain_axiom_drift_rejected
    ),
    "adv_illegal_reorder_stale_hidden_forward_rejected": (
        adv_illegal_reorder_stale_hidden_forward_rejected
    ),
    "adv_forbidden_self_heal_mutations_rejected": (
        adv_forbidden_self_heal_mutations_rejected
    ),
    "adv_unknown_incomplete_trigger_runs_treatment": (
        adv_unknown_incomplete_trigger_runs_treatment
    ),
    "adv_dominance_identity_drift_invalidates": (
        adv_dominance_identity_drift_invalidates
    ),
    "adv_revmath_class_mutation_rejected": adv_revmath_class_mutation_rejected,
}


def _run_case(entry: Mapping[str, Any], *, expect_reject: bool) -> CaseOutcome:
    runner_name = str(entry["runner"])
    runner = _RUNNERS.get(runner_name)
    if runner is None:
        return CaseOutcome(
            case_id=str(entry["id"]),
            scenario=str(entry.get("scenario", "")),
            gate=str(entry["gate"]),
            layer=str(entry.get("layer", "")),
            expect_reject=expect_reject,
            rejected=False,
            detail=f"unknown runner {runner_name!r}",
            ok=False,
        )
    rejected, detail = runner()
    if expect_reject:
        ok = rejected
    else:
        ok = not rejected
    return CaseOutcome(
        case_id=str(entry["id"]),
        scenario=str(entry.get("scenario", "")),
        gate=str(entry["gate"]),
        layer=str(entry.get("layer", "")),
        expect_reject=expect_reject,
        rejected=rejected if expect_reject else (not ok),
        detail=detail,
        ok=ok,
    )


def run_suite(*, root: Path | None = None) -> dict[str, Any]:
    """Execute the matrix; adversarial cases reject, positives pass."""

    matrix = load_matrix(root=root)
    required = list(matrix["required_scenarios"])
    adv_outcomes = [
        _run_case(entry, expect_reject=True)
        for entry in matrix["adversarial_cases"]
    ]
    pos_outcomes = [
        _run_case(entry, expect_reject=False)
        for entry in matrix["positive_controls"]
    ]
    covered = sorted(
        {item.scenario for item in adv_outcomes}
        | {item.scenario for item in pos_outcomes}
    )
    missing = sorted(set(required) - set(covered))
    no_destructive_authority = all(item.ok for item in adv_outcomes) and all(
        item.ok for item in pos_outcomes
    )
    all_ok = no_destructive_authority and not missing
    return {
        "schema": REPORT_SCHEMA,
        "integ": "INTEG-06",
        "issue": "SLM-573",
        "passed": all_ok,
        "adversarial_count": len(adv_outcomes),
        "positive_count": len(pos_outcomes),
        "covered_scenarios": covered,
        "missing_scenarios": missing,
        "adversarial_cases": [item.to_dict() for item in adv_outcomes],
        "positive_controls": [item.to_dict() for item in pos_outcomes],
        "no_failure_becomes_destructive_authority": no_destructive_authority,
        "version_stamp": build_version_stamp("formal.objects"),
    }


def coverage_table(report: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """Scenario → gate coverage rows for docs and PR reports."""

    payload = report or run_suite()
    rows: list[dict[str, str]] = []
    for item in list(payload["adversarial_cases"]) + list(
        payload["positive_controls"]
    ):
        rows.append(
            {
                "case": item["case_id"],
                "scenario": item["scenario"],
                "gate": item["gate"],
                "layer": item["layer"],
                "expect_reject": str(item["expect_reject"]),
                "ok": str(item["ok"]),
            }
        )
    return rows


def gate_coverage_by_scenario(
    report: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """One row per required scenario with primary adversarial + positive gates."""

    payload = report or run_suite()
    matrix = load_matrix()
    by_scenario: dict[str, dict[str, str]] = {
        sid: {
            "scenario": sid,
            "adversarial_gate": "",
            "positive_gate": "",
            "layer": "",
            "ok": "True",
        }
        for sid in matrix["required_scenarios"]
    }
    for item in payload["adversarial_cases"]:
        row = by_scenario[item["scenario"]]
        row["adversarial_gate"] = item["gate"]
        row["layer"] = item["layer"]
        if not item["ok"]:
            row["ok"] = "False"
    for item in payload["positive_controls"]:
        row = by_scenario[item["scenario"]]
        row["positive_gate"] = item["gate"]
        if not row["layer"]:
            row["layer"] = item["layer"]
        if not item["ok"]:
            row["ok"] = "False"
    return [by_scenario[sid] for sid in matrix["required_scenarios"]]


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def assert_suite_passed(report: Mapping[str, Any]) -> None:
    if report.get("passed"):
        return
    failures = [
        item
        for item in list(report.get("adversarial_cases", []))
        + list(report.get("positive_controls", []))
        if not item.get("ok")
    ]
    missing = report.get("missing_scenarios") or []
    raise AssertionError(
        "INTEG-06 adversarial acceptance failed: "
        f"failures={failures!r} missing_scenarios={missing!r}"
    )


__all__ = [
    "DEFAULT_RESULTS_RELPATH",
    "MATRIX_SCHEMA",
    "REPORT_SCHEMA",
    "CaseOutcome",
    "assert_suite_passed",
    "coverage_table",
    "gate_coverage_by_scenario",
    "load_matrix",
    "matrix_path",
    "run_suite",
    "write_report",
]
