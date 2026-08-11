"""EVID-11 / SLM-559: formal-evidence mutation + red-team acceptance suite.

Assembles existing EVID-06/07/09/10 gates into a release-blocking matrix.
Does not invent a third evidence stack — runners call the live APIs.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.formal.authority import (
    CheckerResultRefV1,
    FormalAuthorityV2,
    build_formal_authority_v2,
    from_formal_object_v1,
)
from slm_training.formal.checker_capability import (
    CHECKER_TRUST_DOMAINS,
    SEMANTIC_AUTHORITY_POLICY,
    CheckerRunView,
    capability_satisfied_by,
    evaluate_acceptance_policy,
    semantic_authority_requires_distinct_trust_domains,
)
from slm_training.formal.encoding_adapter import (
    CnfRefEncodingAdapter,
    encoding_authorizes_semantic_result,
    fixture_problems,
    mint_checked_certificate_via_encoding,
    mutate_encoded_flip_first_literal,
)
from slm_training.formal.judgment import (
    ClassificationInputsV1,
    JudgmentOutcome,
    classify,
    from_exact_closure_query,
)
from slm_training.formal.objects import export_closure_law, export_lean_claim
from slm_training.formal.refutation_authority import (
    BindingIdsV1,
    LegacyExhaustionFlagV1,
    evidence_authorizes_removal,
    legacy_exhaustion_authorizes_removal,
    make_exact_replay_evidence,
)
from slm_training.formal.theorem_binding import (
    LeanTheoremBindingV1,
    seal_theorem_binding,
    verify_theorem_binding,
)
from slm_training.versioning import build_version_stamp

MATRIX_SCHEMA = "formal_evidence_mutation_matrix/v1"
REPORT_SCHEMA = "formal_evidence_mutation_suite_report/v1"
MATRIX_RELPATH = (
    "src/slm_training/resources/formal/evid11_mutation_matrix.v1.json"
)
DEFAULT_RESULTS_RELPATH = "docs/design/formal-evidence-mutation-results.json"

_FORBIDDEN_SOURCE = re.compile(r"\b(?:sorry|admit|axiom)\b")

_THEOREM_ID = "ExactClosure.closePass_subset"
_MODULE = "LeverProofLean.ExactClosure"
_LOCAL = "closePass_subset"

_BINDING = BindingIdsV1(
    state_id="evid11-state",
    problem_id="evid11-problem",
    source_id="vss",
    tool_id="python_replay",
)


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    family: str
    gate: str
    capability: str
    expect_reject: bool
    rejected: bool
    detail: str
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "gate": self.gate,
            "capability": self.capability,
            "expect_reject": self.expect_reject,
            "rejected": self.rejected,
            "detail": self.detail,
            "ok": self.ok,
        }


def matrix_path(*, root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / MATRIX_RELPATH


def load_matrix(*, root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(matrix_path(root=root).read_text(encoding="utf-8"))
    if payload.get("schema") != MATRIX_SCHEMA:
        raise ValueError(f"expected schema {MATRIX_SCHEMA!r}")
    return payload


def _seed_project(root: Path, *, proposition: str = "True") -> None:
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.30.0\n", encoding="utf-8")
    (root / "lakefile.toml").write_text('name = "stub"\n', encoding="utf-8")
    (root / "lake-manifest.json").write_text("{}\n", encoding="utf-8")
    path = root.joinpath(*_MODULE.split(".")).with_suffix(".lean")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"theorem {_LOCAL} : {proposition} := trivial\n", encoding="utf-8")


def _seal(root: Path, *, axioms: tuple[str, ...] = ()) -> LeanTheoremBindingV1:
    return seal_theorem_binding(
        fq_name=_THEOREM_ID,
        module=_MODULE,
        lean_root=root,
        axiom_footprint=axioms,
    )


def _reject(detail: str) -> tuple[bool, str]:
    return True, detail


def _accept(detail: str) -> tuple[bool, str]:
    return False, detail


def mutation_same_name_proposition_to_true() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_project(root, proposition="∀ (live : Nat), live = live")
        binding = _seal(root)
        _seed_project(root, proposition="True")
        violations = verify_theorem_binding(
            binding, lean_root=root, catalog_module=_MODULE, live_axiom_footprint=()
        )
        if any("proposition mutation" in item for item in violations):
            return _reject("; ".join(violations))
        return _accept(f"expected proposition rejection, got {violations!r}")


def mutation_changed_conclusion() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_project(root, proposition="True")
        binding = _seal(root)
        _seed_project(root, proposition="False")
        violations = verify_theorem_binding(
            binding, lean_root=root, live_axiom_footprint=()
        )
        if any("proposition mutation" in item for item in violations):
            return _reject("; ".join(violations))
        return _accept(f"expected proposition rejection, got {violations!r}")


def mutation_toolchain_lock_drift() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_project(root)
        binding = _seal(root)
        (root / "lake-manifest.json").write_text('{"packages":[]}\n', encoding="utf-8")
        violations = verify_theorem_binding(
            binding, lean_root=root, live_axiom_footprint=()
        )
        if any("lake manifest" in item for item in violations):
            return _reject("; ".join(violations))
        return _accept(f"expected lake manifest drift, got {violations!r}")


def mutation_source_tree_drift() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_project(root, proposition="True")
        binding = _seal(root)
        lean = root.joinpath(*_MODULE.split(".")).with_suffix(".lean")
        lean.write_text(
            f"theorem {_LOCAL} : True := trivial\n-- drift marker\n",
            encoding="utf-8",
        )
        violations = verify_theorem_binding(
            binding, lean_root=root, live_axiom_footprint=()
        )
        if any("source-tree digest drift" in item for item in violations):
            return _reject("; ".join(violations))
        return _accept(f"expected source-tree drift, got {violations!r}")


def mutation_catalog_module_redirection() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_project(root)
        binding = _seal(root)
        violations = verify_theorem_binding(
            binding,
            lean_root=root,
            catalog_module="LeverProofLean.OtherModule",
            live_axiom_footprint=(),
        )
        if any("catalog redirection" in item for item in violations):
            return _reject("; ".join(violations))
        return _accept(f"expected catalog redirection, got {violations!r}")


def mutation_unexpected_axiom() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_project(root)
        binding = _seal(root, axioms=())
        violations = verify_theorem_binding(
            binding,
            lean_root=root,
            live_axiom_footprint=("classical.choice",),
        )
        if any("unexpected axiom" in item or "axiom" in item for item in violations):
            return _reject("; ".join(violations))
        return _accept(f"expected axiom rejection, got {violations!r}")


def mutation_forbidden_proof_escape_token() -> tuple[bool, str]:
    forged = "theorem escape : True := by sorry\n"
    if _FORBIDDEN_SOURCE.search(forged):
        return _reject("forbidden token 'sorry' matched")
    return _accept("forbidden-source regex missed sorry")


def mutation_forged_exhaustion_replay_flags() -> tuple[bool, str]:
    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="unsupported",
            exhausted=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    if (
        judgment.outcome is JudgmentOutcome.UNKNOWN
        and not judgment.authorizes_removal()
    ):
        return _reject("forged exhaustion/replay flags did not authorize removal")
    return _accept(f"forged flags leaked authority: {judgment.outcome}")


def mutation_legacy_exhaustion_telemetry() -> tuple[bool, str]:
    flag = LegacyExhaustionFlagV1(
        exhausted=True, replay_ok=True, coverage_complete=True
    )
    if legacy_exhaustion_authorizes_removal(flag) is False:
        return _reject("legacy exhaustion telemetry correctly inert")
    return _accept("legacy exhaustion incorrectly authorized removal")


def mutation_skipped_checker_as_success() -> tuple[bool, str]:
    evidence = make_exact_replay_evidence(_BINDING, evidence_digest="evid11-digest")
    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="unsupported",
            refutation_evidence=evidence,
            expected_binding=_BINDING,
        ),
        checked=True,
    )
    obj = export_closure_law("removable_only_replay_valid_unsupported")
    try:
        from_formal_object_v1(
            obj,
            judgment=judgment,
            checker_results=(
                CheckerResultRefV1(backend="python_structural", status="pass"),
                CheckerResultRefV1(backend="python_replay", status="skipped"),
            ),
        )
    except ValueError as exc:
        if "checked semantic" in str(exc):
            return _reject(str(exc))
        return _accept(f"wrong rejection: {exc}")
    return _accept("skipped checker was accepted as checked semantic")


def mutation_timeout_missing_tool_strip_evidence() -> tuple[bool, str]:
    evidence = make_exact_replay_evidence(_BINDING, evidence_digest="evid11-digest")
    details: list[str] = []
    for extra in (
        {"timed_out": True},
        {"missing_tool": True},
        {"skipped_replay": True},
    ):
        judgment = from_exact_closure_query(
            verdict="unsupported",
            replay_ok=True,
            replay_checked=True,
            checked=True,
            exhausted=True,
            refutation_evidence=evidence,
            expected_binding=_BINDING,
            **extra,
        )
        if judgment.outcome not in {
            JudgmentOutcome.UNKNOWN,
            JudgmentOutcome.INVALID,
        } or judgment.authorizes_removal():
            return _accept(f"failure mode leaked authority under {extra}: {judgment}")
        details.append(f"{sorted(extra)}->{judgment.outcome.value}")
    return _reject("; ".join(details))


def mutation_stale_problem_identity() -> tuple[bool, str]:
    stale = BindingIdsV1(
        state_id=_BINDING.state_id,
        problem_id="stale-problem",
        source_id=_BINDING.source_id,
        tool_id=_BINDING.tool_id,
    )
    evidence = make_exact_replay_evidence(stale, evidence_digest="digest")
    if evidence_authorizes_removal(evidence, _BINDING) is False:
        return _reject("stale problem_id rejected")
    return _accept("stale problem_id authorized removal")


def mutation_certificate_wrong_encoding() -> tuple[bool, str]:
    adapter = CnfRefEncodingAdapter()
    problem = fixture_problems()["unsat_contradiction"]
    result, evidence = adapter.build_evidence(problem)
    assert evidence is not None and result.encoded is not None
    mutated = mutate_encoded_flip_first_literal(result.encoded)
    if encoding_authorizes_semantic_result(
        evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=mutated.digest(),
    ):
        return _accept("mutated encoding still authorized")
    return _reject("wrong encoding digest blocked authority")


def mutation_stale_problem_digest_on_mint() -> tuple[bool, str]:
    adapter = CnfRefEncodingAdapter()
    problem = fixture_problems()["unsat_contradiction"]
    _, evidence = adapter.build_evidence(problem)
    assert evidence is not None
    blocked = mint_checked_certificate_via_encoding(
        _BINDING,
        encoding=evidence,
        expected_problem_digest="0" * 64,
        certificate_checked=True,
        encoded_digest_observed=evidence.encoded_digest,
    )
    if blocked is None:
        return _reject("stale problem digest blocked mint")
    return _accept("stale problem digest minted certificate")


def mutation_shared_parser_family_fault() -> tuple[bool, str]:
    """Both structural+reference 'pass' but share a trust domain — no authority."""

    backends = ("python_structural", "python_reference")
    if semantic_authority_requires_distinct_trust_domains(backends):
        return _accept("shared-family backends incorrectly counted as diverse")
    # Distinct domains still authorize at the diversity gate.
    if not semantic_authority_requires_distinct_trust_domains(
        ("python_structural", "lean_kernel")
    ):
        return _accept("distinct trust domains failed positive diversity check")
    report = evaluate_acceptance_policy(
        (
            CheckerRunView(backend="python_structural", ok=True),
            CheckerRunView(backend="python_reference", ok=True),
        ),
        SEMANTIC_AUTHORITY_POLICY,
    )
    if report.accepted or report.verification_kind != "redundant_conformance":
        return _accept(
            f"production semantic policy misclassified shared family: {report.to_dict()}"
        )
    return _reject(
        "shared python_structural/reference trust domain denied semantic authority"
    )


def mutation_shared_serialization_common_mode() -> tuple[bool, str]:
    """Shared FormalObject serialization fault hits both nominal Python backends."""

    from slm_training.formal.checkers import (
        check_python_reference,
        check_python_structural,
    )
    from slm_training.formal.objects import FormalObjectV1

    obj = export_lean_claim("Judgment.timeout_never_refuted")
    raw = obj.to_dict()
    digest = raw["content_digest"]
    flipped = ("0" if digest[0] != "0" else "1") + digest[1:]
    raw["content_digest"] = flipped
    try:
        FormalObjectV1.from_dict(raw)
        parse_rejected = False
    except ValueError:
        parse_rejected = True
    if not parse_rejected:
        return _accept("corrupted digest still parsed as FormalObjectV1")

    structural = check_python_structural(obj)
    reference = check_python_reference(obj)
    if not (structural.ok and reference.ok):
        return _accept("honest object failed shared structural family checkers")
    if CHECKER_TRUST_DOMAINS[structural.backend] != CHECKER_TRUST_DOMAINS[
        reference.backend
    ]:
        return _accept("structural/reference unexpectedly split trust domains")
    report = evaluate_acceptance_policy(
        (
            CheckerRunView(backend=structural.backend, ok=True),
            CheckerRunView(backend=reference.backend, ok=True),
        ),
        SEMANTIC_AUTHORITY_POLICY,
    )
    if report.accepted:
        return _accept("common-mode structural pair conferred semantic authority")
    return _reject(
        "shared serialization family common-mode success blocked for semantic authority"
    )


def mutation_unadvertised_capability_claim() -> tuple[bool, str]:
    """A checker cannot satisfy a capability it does not advertise."""

    from slm_training.formal.checker_capability import AcceptancePolicyV1

    if capability_satisfied_by("runtime_refinement", backend="python_structural"):
        return _accept("python_structural incorrectly advertises runtime_refinement")
    report = evaluate_acceptance_policy(
        (CheckerRunView(backend="python_structural", ok=True),),
        AcceptancePolicyV1(
            policy_id="probe_runtime_refinement/v1",
            required_capabilities=frozenset({"runtime_refinement"}),
            min_distinct_trust_domains=1,
            require_independent_verification=False,
        ),
        claimed_capabilities=("runtime_refinement",),
    )
    if report.accepted or "runtime_refinement" not in report.missing_capabilities:
        return _accept(f"unadvertised capability accepted: {report.to_dict()}")
    return _reject("unadvertised runtime_refinement rejected by policy")


def mutation_v2_content_digest_mismatch() -> tuple[bool, str]:
    obj = export_lean_claim("Judgment.timeout_never_refuted")
    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="supported",
            has_witness_digest=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    authority = from_formal_object_v1(obj, judgment=judgment)
    raw = authority.to_dict()
    raw["content_digest"] = "0" * 64
    try:
        FormalAuthorityV2.from_dict(raw)
    except ValueError as exc:
        if "content_digest mismatch" in str(exc):
            return _reject(str(exc))
        return _accept(f"wrong rejection: {exc}")
    return _accept("tampered content_digest was accepted")


def mutation_v1_v2_migration_tamper() -> tuple[bool, str]:
    """Forge checked_semantic over a timeout/unknown judgment during migration."""

    judgment = classify(ClassificationInputsV1(timed_out=True), checked=False)
    obj = export_lean_claim("Judgment.timeout_never_refuted")
    try:
        build_formal_authority_v2(
            claim_id=obj.object_id,
            surface="formal_object",
            judgment=judgment,
            v1_artifact=obj.to_dict(),
            v1_content_digest=obj.content_digest,
            authority_class="checked_semantic",
        )
    except ValueError as exc:
        if "authority_class" in str(exc):
            return _reject(str(exc))
        return _accept(f"wrong rejection: {exc}")
    return _accept("migration tamper accepted checked_semantic over timeout")


def positive_exact_replay_authorizes() -> tuple[bool, str]:
    evidence = make_exact_replay_evidence(_BINDING, evidence_digest="ok-digest")
    if evidence_authorizes_removal(evidence, _BINDING):
        return _accept("exact replay authorized")  # positive: not a rejection
    return _reject("positive exact replay failed to authorize")


def positive_honest_encoding_authorizes() -> tuple[bool, str]:
    adapter = CnfRefEncodingAdapter()
    problem = fixture_problems()["unsat_contradiction"]
    result, evidence = adapter.build_evidence(problem)
    assert evidence is not None and result.encoded is not None
    ok = encoding_authorizes_semantic_result(
        evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=result.encoded.digest(),
    )
    if ok:
        return _accept("honest encoding authorized")
    return _reject("positive honest encoding failed to authorize")


def positive_sealed_binding_holds() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_project(root)
        binding = _seal(root)
        violations = verify_theorem_binding(
            binding, lean_root=root, catalog_module=_MODULE, live_axiom_footprint=()
        )
        if not violations:
            return _accept("sealed binding holds")
        return _reject(f"positive binding failed: {violations}")


_RUNNERS: dict[str, Callable[[], tuple[bool, str]]] = {
    "mutation_same_name_proposition_to_true": mutation_same_name_proposition_to_true,
    "mutation_changed_conclusion": mutation_changed_conclusion,
    "mutation_toolchain_lock_drift": mutation_toolchain_lock_drift,
    "mutation_source_tree_drift": mutation_source_tree_drift,
    "mutation_catalog_module_redirection": mutation_catalog_module_redirection,
    "mutation_unexpected_axiom": mutation_unexpected_axiom,
    "mutation_forbidden_proof_escape_token": mutation_forbidden_proof_escape_token,
    "mutation_forged_exhaustion_replay_flags": mutation_forged_exhaustion_replay_flags,
    "mutation_legacy_exhaustion_telemetry": mutation_legacy_exhaustion_telemetry,
    "mutation_skipped_checker_as_success": mutation_skipped_checker_as_success,
    "mutation_timeout_missing_tool_strip_evidence": mutation_timeout_missing_tool_strip_evidence,
    "mutation_stale_problem_identity": mutation_stale_problem_identity,
    "mutation_certificate_wrong_encoding": mutation_certificate_wrong_encoding,
    "mutation_stale_problem_digest_on_mint": mutation_stale_problem_digest_on_mint,
    "mutation_shared_parser_family_fault": mutation_shared_parser_family_fault,
    "mutation_shared_serialization_common_mode": mutation_shared_serialization_common_mode,
    "mutation_unadvertised_capability_claim": mutation_unadvertised_capability_claim,
    "mutation_v2_content_digest_mismatch": mutation_v2_content_digest_mismatch,
    "mutation_v1_v2_migration_tamper": mutation_v1_v2_migration_tamper,
    "positive_exact_replay_authorizes": positive_exact_replay_authorizes,
    "positive_honest_encoding_authorizes": positive_honest_encoding_authorizes,
    "positive_sealed_binding_holds": positive_sealed_binding_holds,
}


def _run_case(entry: Mapping[str, Any], *, expect_reject: bool) -> CaseOutcome:
    runner_name = str(entry["runner"])
    runner = _RUNNERS.get(runner_name)
    if runner is None:
        return CaseOutcome(
            case_id=str(entry["id"]),
            family=str(entry.get("family", "positive")),
            gate=str(entry["gate"]),
            capability=str(entry["capability"]),
            expect_reject=expect_reject,
            rejected=False,
            detail=f"unknown runner {runner_name!r}",
            ok=False,
        )
    rejected, detail = runner()
    # Positive controls use expect_reject=False: "rejected" means the *gate*
    # fired; for positives we invert — runner returns rejected=False on success
    # via _accept, and rejected=True via _reject when the positive fails.
    if expect_reject:
        ok = rejected
    else:
        ok = not rejected
    return CaseOutcome(
        case_id=str(entry["id"]),
        family=str(entry.get("family", "positive")),
        gate=str(entry["gate"]),
        capability=str(entry["capability"]),
        expect_reject=expect_reject,
        rejected=rejected if expect_reject else (not ok),
        detail=detail,
        ok=ok,
    )


def run_suite(*, root: Path | None = None) -> dict[str, Any]:
    """Execute the matrix; every mutation must reject, positives must pass."""

    matrix = load_matrix(root=root)
    required = list(matrix["required_families"])
    mutation_outcomes = [
        _run_case(entry, expect_reject=True) for entry in matrix["mutations"]
    ]
    positive_outcomes = [
        _run_case(entry, expect_reject=False) for entry in matrix["positive_controls"]
    ]
    covered = sorted({item.family for item in mutation_outcomes})
    missing_families = sorted(set(required) - set(covered))
    all_ok = (
        all(item.ok for item in mutation_outcomes)
        and all(item.ok for item in positive_outcomes)
        and not missing_families
    )
    return {
        "schema": REPORT_SCHEMA,
        "evid": "EVID-11",
        "issue": "SLM-559",
        "passed": all_ok,
        "mutation_count": len(mutation_outcomes),
        "positive_count": len(positive_outcomes),
        "covered_families": covered,
        "missing_families": missing_families,
        "mutations": [item.to_dict() for item in mutation_outcomes],
        "positive_controls": [item.to_dict() for item in positive_outcomes],
        "structural_consistency_alone_confers_authority": False,
        "version_stamp": build_version_stamp("formal.objects"),
    }


def coverage_table(report: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """Mutation id → gate/capability rows for docs and PR reports."""

    payload = report or run_suite()
    rows: list[dict[str, str]] = []
    for item in payload["mutations"]:
        rows.append(
            {
                "mutation": item["case_id"],
                "family": item["family"],
                "gate": item["gate"],
                "capability": item["capability"],
                "rejected": str(item["rejected"]),
                "ok": str(item["ok"]),
            }
        )
    return rows


def write_report(
    report: Mapping[str, Any],
    path: Path,
) -> Path:
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
        for item in list(report.get("mutations", []))
        + list(report.get("positive_controls", []))
        if not item.get("ok")
    ]
    missing = report.get("missing_families") or []
    raise AssertionError(
        "formal evidence mutation suite failed: "
        f"failures={failures!r} missing_families={missing!r}"
    )


__all__ = [
    "CHECKER_TRUST_DOMAINS",
    "DEFAULT_RESULTS_RELPATH",
    "MATRIX_SCHEMA",
    "REPORT_SCHEMA",
    "CaseOutcome",
    "assert_suite_passed",
    "coverage_table",
    "load_matrix",
    "matrix_path",
    "run_suite",
    "semantic_authority_requires_distinct_trust_domains",
    "write_report",
]
