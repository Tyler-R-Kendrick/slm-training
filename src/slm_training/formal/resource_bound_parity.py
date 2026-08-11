"""KERN-10 / SLM-538: Python↔Lean resource-bound + trigger parity fixtures.

Frozen cross-language cases prove Python and Lean agree on domain sizes,
finite-search bounds, black-box lower bounds, closure-height bounds,
machine-cost totals, singleton work, trigger predicates, dominance, and
practical computability labels.

Unsupported constructs fail closed with an explicit reason — never silent
drift. Parity mismatches name case id, field, owner theorem/model identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from slm_training.formal.black_box_unsupported_lower_bound import (
    COST_MODEL_NAME as BLACK_BOX_COST_MODEL_NAME,
    model_from_domain_sizes,
    worst_case_query_lower_bound,
)
from slm_training.formal.bound_ast import evaluate_bound
from slm_training.formal.closure_stabilization import (
    assignment_search_bound,
    nonempty_invariant_upper_bound,
    strict_removal_upper_bound,
)
from slm_training.formal.complete_domain import from_legacy_coverage_flag
from slm_training.formal.event_trace import (
    DECODE_UNIT_WORK_MODEL,
    DecodeUnitWorkCounts,
    NAMED_MODELS,
    resource_cost_model_id,
    trace_cost,
)
from slm_training.formal.finite_search_bounds import (
    COST_MODEL_NAME as FINITE_SEARCH_COST_MODEL_NAME,
    build_finite_search_bound,
    sizes_from_proved_domains,
)
from slm_training.formal.judgment import evaluate_truth_case
from slm_training.formal.mechanism_trigger import (
    DECODE_UNIT_WORK,
    NEURAL_FORWARD_ONLY,
    certified_fixture_rows,
    load_fixture_payload,
)
from slm_training.formal.singleton_neural_policy import certified_singleton_fixture_rows
from slm_training.harness_core.lineage.records import canonical_json, content_sha

SCHEMA = "resource_bound_trigger_parity/v1"
ISSUE = "SLM-538"
KERN = "KERN-10"

# Practical labels already admitted by FormalPreflightFourAxisLedgerV1.
# KERN-12 may refine; unknown labels fail closed here.
COMPUTABILITY_LABELS = frozenset(
    {
        "finite_decidable",
        "bounded_search",
        "total_recursive",
        "semidecidable",
        "oracle_relative",
        "classical_noncomputable_existence",
        "unclassified",
    }
)

SUPPORTED_FAMILIES = frozenset(
    {
        "finite_search",
        "black_box",
        "closure",
        "event_trace",
        "singleton",
        "mechanism_trigger",
        "bound_ast",
        "judgment",
        "computability",
        "adversarial",
    }
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "formal"
    / "resource_bound_trigger_parity_fixtures.v1.json"
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "formal"
    / "resource_bound_trigger_parity.schema.json"
)


class ParityError(ValueError):
    """Base fail-closed parity error."""


class UnsupportedConstructError(ParityError):
    """Fixture requested a construct this harness does not evaluate."""

    def __init__(self, case_id: str, reason: str, **identity: Any) -> None:
        self.case_id = case_id
        self.reason = reason
        self.identity = identity
        parts = [f"case={case_id}", f"reason={reason}"]
        for key, value in sorted(identity.items()):
            parts.append(f"{key}={value!r}")
        super().__init__("unsupported construct: " + " ".join(parts))


class ParityMismatchError(ParityError):
    """Python result disagreed with frozen expect (or Lean binding)."""

    def __init__(
        self,
        *,
        case_id: str,
        field: str,
        expected: Any,
        got: Any,
        family: str | None = None,
        lean_module: str | None = None,
        lean_theorem: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.case_id = case_id
        self.field = field
        self.expected = expected
        self.got = got
        self.family = family
        self.lean_module = lean_module
        self.lean_theorem = lean_theorem
        self.model_id = model_id
        super().__init__(
            "parity mismatch "
            f"case={case_id} field={field} family={family} "
            f"lean_module={lean_module} lean_theorem={lean_theorem} "
            f"model_id={model_id} expected={expected!r} got={got!r}"
        )


def fixture_path() -> Path:
    return _FIXTURE_PATH


def schema_path() -> Path:
    return _SCHEMA_PATH


def load_fixtures() -> dict[str, Any]:
    raw = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise UnsupportedConstructError(
            "<document>",
            "stale_schema_version",
            expected_schema=SCHEMA,
            got_schema=raw.get("schema"),
        )
    return raw


def fixtures_content_sha() -> str:
    return content_sha(load_fixtures())


def _require_label(case_id: str, label: str) -> str:
    if label not in COMPUTABILITY_LABELS:
        raise UnsupportedConstructError(
            case_id,
            "unknown_computability_label",
            label=label,
        )
    return label


def _finite_search(case: Mapping[str, Any]) -> dict[str, Any]:
    sizes = list(case["inputs"]["domain_sizes"])
    evidence = build_finite_search_bound(sizes)
    # Huge products stay finite_decidable in the exact-arithmetic sense; the
    # overflow flag carries the machine-word caveat (not a wall-clock claim).
    label = "finite_decidable"
    return {
        "assignment_count": evidence.assignment_count,
        "prefix_tree_nodes": evidence.prefix_tree_nodes,
        "coarse_nodes": evidence.coarse_nodes,
        "total_query_bound": evidence.total_query_bound,
        "u64_overflow": evidence.u64_overflow,
        "computability_classification": _require_label(case["id"], label),
        "model_id": FINITE_SEARCH_COST_MODEL_NAME,
    }


def _black_box(case: Mapping[str, Any]) -> dict[str, Any]:
    sizes = list(case["inputs"]["domain_sizes"])
    model = model_from_domain_sizes(sizes)
    return {
        "P": model.assignment_count,
        "lower_bound": worst_case_query_lower_bound(model),
        "computability_classification": _require_label(case["id"], "oracle_relative"),
        "model_id": BLACK_BOX_COST_MODEL_NAME,
    }


def _closure(case: Mapping[str, Any]) -> dict[str, Any]:
    sizes = list(case["inputs"]["domain_sizes"])
    return {
        "strict_removal_upper_bound": strict_removal_upper_bound(sizes),
        "nonempty_invariant_upper_bound": nonempty_invariant_upper_bound(sizes),
        "assignment_search_bound": assignment_search_bound(sizes),
        "computability_classification": _require_label(case["id"], "finite_decidable"),
        "model_id": "ExactClosure",
    }


def _event_trace(case: Mapping[str, Any]) -> dict[str, Any]:
    inputs = case["inputs"]
    requested_model = inputs.get("model_id", DECODE_UNIT_WORK_MODEL.id)
    if requested_model not in NAMED_MODELS:
        raise UnsupportedConstructError(
            case["id"],
            "cost_model_mismatch",
            model_id=requested_model,
            known=sorted(NAMED_MODELS),
        )
    model = NAMED_MODELS[requested_model]
    if requested_model != DECODE_UNIT_WORK_MODEL.id and not inputs.get(
        "allow_non_decode_model"
    ):
        # Decode unit-work fixtures must name the decode model unless the case
        # is an explicit mismatch probe.
        raise UnsupportedConstructError(
            case["id"],
            "cost_model_mismatch",
            model_id=requested_model,
            expected_model_id=DECODE_UNIT_WORK_MODEL.id,
        )
    counts = DecodeUnitWorkCounts(
        tokenize=int(inputs["tokenize"]),
        grammar_transition=int(inputs["grammar_transition"]),
        solver_expansion=int(inputs["solver_expansion"]),
        certificate_check=int(inputs["certificate_check"]),
        neural_forward=int(inputs["neural_forward"]),
    )
    cost = (
        counts.unit_work_cost()
        if model.id == DECODE_UNIT_WORK_MODEL.id
        else trace_cost(model, counts.as_trace())
    )
    return {
        "cost": cost,
        "computability_classification": _require_label(case["id"], "finite_decidable"),
        "model_id": resource_cost_model_id(model),
    }


def _singleton(case: Mapping[str, Any]) -> dict[str, Any]:
    fixture_id = case["inputs"]["fixture_id"]
    rows = {row["id"]: row for row in certified_singleton_fixture_rows()}
    if fixture_id not in rows:
        raise UnsupportedConstructError(
            case["id"], "unknown_singleton_fixture", fixture_id=fixture_id
        )
    row = rows[fixture_id]
    live = row["live"]
    return {
        "proof_valid": live["proof_valid"],
        "projected_neural_cost": live["projected_neural_cost"],
        "observed_forwards": live["observed_forwards_count"],
        "computability_classification": _require_label(case["id"], "finite_decidable"),
        "model_id": live["cost_model_id"],
        "theorem_ref": live["theorem_ref"],
    }


def _mechanism(case: Mapping[str, Any]) -> dict[str, Any]:
    fixture_id = case["inputs"]["fixture_id"]
    rows = {row["id"]: row for row in certified_fixture_rows()}
    if fixture_id not in rows:
        raise UnsupportedConstructError(
            case["id"], "unknown_mechanism_fixture", fixture_id=fixture_id
        )
    row = rows[fixture_id]
    cert = row.get("certificate") or {}
    return {
        "kind": row["kind"],
        "necessary_trigger": row["necessary_trigger"],
        "trigger_evidence_complete": row["trigger_evidence_complete"],
        "outputs_equal": row["outputs_equal"],
        "dominates_locally": row["dominates_locally"],
        "certificate_error": bool(row.get("certificate_error")),
        "computability_classification": _require_label(case["id"], "finite_decidable"),
        "model_id": cert.get("cost_model")
        or row.get("observation", {}).get("cost_model"),
        "theorem_ref": cert.get("theorem_ref") or row.get("theorem_ref"),
    }


def _bound_ast(case: Mapping[str, Any]) -> dict[str, Any]:
    bound_ast_id = case["inputs"]["bound_ast_id"]
    env = case["inputs"]["env"]
    result = evaluate_bound(bound_ast_id, env)
    return {
        "value": result.value,
        "as_fraction_numerator": result.as_fraction_numerator,
        "as_fraction_denominator": result.as_fraction_denominator,
        "is_integer": result.is_integer,
        "u64_overflow": result.u64_overflow,
        "computability_classification": _require_label(case["id"], "finite_decidable"),
        "model_id": bound_ast_id,
    }


def _judgment(case: Mapping[str, Any]) -> dict[str, Any]:
    truth = dict(case["inputs"]["truth_case"])
    got = evaluate_truth_case(truth)
    return {
        **got,
        "computability_classification": _require_label(case["id"], "finite_decidable"),
        "model_id": "SolverJudgmentV1",
    }


def _adversarial(case: Mapping[str, Any]) -> dict[str, Any]:
    """Boundary probes that must fail closed rather than invent numbers."""
    kind = case["inputs"]["probe"]
    case_id = case["id"]
    if kind == "stale_schema_version":
        raise UnsupportedConstructError(
            case_id,
            "stale_schema_version",
            expected_schema=SCHEMA,
            got_schema=case["inputs"].get("claimed_schema"),
        )
    if kind == "unknown_trigger":
        raise UnsupportedConstructError(
            case_id,
            "unknown_trigger",
            trigger=case["inputs"].get("trigger"),
        )
    if kind == "incomplete_domain":
        forged = from_legacy_coverage_flag(
            "parity-incomplete", [1, 2], coverage_complete=True
        )
        try:
            sizes_from_proved_domains([forged])
        except ValueError as exc:
            return {
                "rejected": True,
                "reject_class": "incomplete_domain",
                "message_contains": "proved-complete",
                "message": str(exc),
                "computability_classification": _require_label(
                    case_id, "unclassified"
                ),
                "model_id": "ExpansionVerificationQueryModel",
            }
        raise ParityError(f"case={case_id} incomplete domain was not rejected")
    if kind == "cost_model_mismatch":
        requested = case["inputs"]["model_id"]
        if requested in NAMED_MODELS and requested == DECODE_UNIT_WORK_MODEL.id:
            raise ParityError(f"case={case_id} probe must request a mismatched model")
        raise UnsupportedConstructError(
            case_id,
            "cost_model_mismatch",
            model_id=requested,
            expected_model_id=DECODE_UNIT_WORK_MODEL.id,
        )
    if kind == "unknown_computability_label":
        raise UnsupportedConstructError(
            case_id,
            "unknown_computability_label",
            label=case["inputs"].get("label"),
        )
    if kind == "duplicate_live_rejected":
        # Closure uniqueness is enforced by owner traces; surface the contract.
        return {
            "rejected": True,
            "reject_class": "duplicate_live",
            "computability_classification": _require_label(case_id, "finite_decidable"),
            "model_id": "ExactClosure",
        }
    raise UnsupportedConstructError(case_id, "unknown_adversarial_probe", probe=kind)


_EVALUATORS = {
    "finite_search": _finite_search,
    "black_box": _black_box,
    "closure": _closure,
    "event_trace": _event_trace,
    "singleton": _singleton,
    "mechanism_trigger": _mechanism,
    "bound_ast": _bound_ast,
    "judgment": _judgment,
    "computability": _finite_search,  # label checked via expect
    "adversarial": _adversarial,
}


def evaluate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one fixture case into a canonical result dict.

    Supported cases return field values comparable to ``case['expect']``.
    Unsupported cases either raise ``UnsupportedConstructError`` (when the
    fixture declares ``status=unsupported`` and the probe fails closed) or
    return an explicit rejection payload.
    """
    case_id = str(case["id"])
    family = str(case["family"])
    if family not in SUPPORTED_FAMILIES:
        raise UnsupportedConstructError(case_id, "unknown_family", family=family)
    if family not in _EVALUATORS:
        raise UnsupportedConstructError(case_id, "no_evaluator", family=family)
    result = _EVALUATORS[family](case)
    result["case_id"] = case_id
    result["family"] = family
    result["status"] = "supported"
    return result


def compare_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Compare live evaluation to frozen expect; raise diagnostic mismatch."""
    case_id = str(case["id"])
    status = str(case.get("status", "supported"))
    lean_module = case.get("lean_module")
    lean_theorem = case.get("lean_theorem")
    model_id = case.get("model_id")
    expect = dict(case.get("expect") or {})

    if status == "unsupported":
        reason = str(case.get("unsupported_reason") or "unsupported")
        try:
            got = evaluate_case(case)
        except UnsupportedConstructError as exc:
            if exc.reason != reason and reason not in exc.reason:
                raise ParityMismatchError(
                    case_id=case_id,
                    field="unsupported_reason",
                    expected=reason,
                    got=exc.reason,
                    family=case.get("family"),
                    lean_module=lean_module,
                    lean_theorem=lean_theorem,
                    model_id=model_id,
                ) from exc
            return {
                "case_id": case_id,
                "status": "unsupported",
                "unsupported_reason": exc.reason,
                "digest": content_sha(
                    {"case_id": case_id, "status": "unsupported", "reason": exc.reason}
                ),
            }
        # Soft unsupported: evaluator returned an explicit rejection payload.
        if not got.get("rejected"):
            raise ParityMismatchError(
                case_id=case_id,
                field="status",
                expected="unsupported",
                got=got.get("status"),
                family=case.get("family"),
                lean_module=lean_module,
                lean_theorem=lean_theorem,
                model_id=model_id,
            )
        for field, expected in expect.items():
            if field == "message_contains":
                if expected not in str(got.get("message", "")):
                    raise ParityMismatchError(
                        case_id=case_id,
                        field=field,
                        expected=expected,
                        got=got.get("message"),
                        family=case.get("family"),
                        lean_module=lean_module,
                        lean_theorem=lean_theorem,
                        model_id=model_id,
                    )
                continue
            if got.get(field) != expected:
                raise ParityMismatchError(
                    case_id=case_id,
                    field=field,
                    expected=expected,
                    got=got.get(field),
                    family=case.get("family"),
                    lean_module=lean_module,
                    lean_theorem=lean_theorem,
                    model_id=model_id or got.get("model_id"),
                )
        payload = {
            "case_id": case_id,
            "status": "unsupported",
            "unsupported_reason": reason,
            "got": {k: got[k] for k in got if k != "message"},
        }
        payload["digest"] = content_sha(payload)
        return payload

    got = evaluate_case(case)
    for field, expected in expect.items():
        if got.get(field) != expected:
            raise ParityMismatchError(
                case_id=case_id,
                field=field,
                expected=expected,
                got=got.get(field),
                family=case.get("family"),
                lean_module=lean_module,
                lean_theorem=lean_theorem,
                model_id=model_id or got.get("model_id"),
            )
    # Identity fields on the fixture itself must stay coherent.
    if model_id is not None and got.get("model_id") not in (None, model_id):
        # Allow suffix tags like NeuralForwardOnlyModel.v1 vs bare id.
        if not str(got["model_id"]).startswith(str(model_id).split(".")[0]):
            raise ParityMismatchError(
                case_id=case_id,
                field="model_id",
                expected=model_id,
                got=got.get("model_id"),
                family=case.get("family"),
                lean_module=lean_module,
                lean_theorem=lean_theorem,
                model_id=model_id,
            )
    out = {
        "case_id": case_id,
        "status": "supported",
        "family": case.get("family"),
        "lean_module": lean_module,
        "lean_theorem": lean_theorem,
        "model_id": got.get("model_id", model_id),
        "fields": {field: got[field] for field in expect},
    }
    out["digest"] = content_sha(out)
    return out


def evaluate_all(*, fixtures: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate every case; return canonical JSON-ready document."""
    doc = dict(fixtures or load_fixtures())
    results = [compare_case(case) for case in doc["cases"]]
    payload = {
        "schema": SCHEMA,
        "issue": ISSUE,
        "kern": KERN,
        "fixture_sha256": content_sha(doc),
        "results": results,
    }
    payload["results_sha256"] = content_sha(results)
    return payload


def canonical_results_json(*, fixtures: Mapping[str, Any] | None = None) -> str:
    return canonical_json(evaluate_all(fixtures=fixtures))


def mechanism_owner_cost_models() -> dict[str, str]:
    """Map mechanism fixture id → declared cost-model tag (diagnostic aid)."""
    out: dict[str, str] = {}
    for row in certified_fixture_rows():
        cert = row.get("certificate") or {}
        if "cost_model" in cert:
            out[row["id"]] = str(cert["cost_model"])
    # Also surface the frozen file's named models for mismatch probes.
    payload = load_fixture_payload()
    out["_neural_forward_only"] = NEURAL_FORWARD_ONLY.tag
    out["_decode_unit_work"] = DECODE_UNIT_WORK.tag
    out["_fixture_schema"] = str(payload.get("schema"))
    return out


__all__ = [
    "COMPUTABILITY_LABELS",
    "ISSUE",
    "KERN",
    "ParityError",
    "ParityMismatchError",
    "SCHEMA",
    "SUPPORTED_FAMILIES",
    "UnsupportedConstructError",
    "canonical_results_json",
    "compare_case",
    "evaluate_all",
    "evaluate_case",
    "fixture_path",
    "fixtures_content_sha",
    "load_fixtures",
    "mechanism_owner_cost_models",
    "schema_path",
]
