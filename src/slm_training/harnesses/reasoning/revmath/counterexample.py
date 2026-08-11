"""Computable/finite counterexample validation (HARN-06 / SLM-546).

Counterexample validators check the **exact** proposition and base assumptions
of the weakened theorem. A replayable finite model or computable trace is
required before ``refuted``. Proof-search failure, timeouts, and missing models
map to ``unknown`` — never impossibility / refutation (KERN-01).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskV1,
)

COUNTEREXAMPLE_META_SCHEMA = "revmath_counterexample_meta/v1"
COUNTEREXAMPLE_MODEL_SCHEMA = "revmath_counterexample_model/v1"
COUNTEREXAMPLE_REPORT_SCHEMA = "revmath_counterexample_report/v1"

ModelKind = Literal["finite_model", "computable_trace"]
HermeticCounterexampleScenario = Literal[
    "checked_refutation",
    "search_failed",
    "no_counterexample",
    "mismatched_proposition",
    "mismatched_assumptions",
    "timeout",
    "incomplete",
]
CounterexampleOutcomeName = Literal["witnessed", "refuted", "unknown", "invalid"]


def _sorted_ids(ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(ids)))


def _require_statement_digest(statement: str, digest: str, *, label: str) -> None:
    actual = content_sha(statement)
    if actual != digest:
        raise RevmathSchemaError(
            f"{label} statement_sha256 mismatch "
            f"(expected {digest!r}, got {actual!r})"
        )


@dataclass(frozen=True)
class CounterexampleModelV1:
    """Replayable finite model or computable trace (required for refutation)."""

    schema_version: str
    model_id: str
    model_kind: ModelKind
    # Opaque but deterministic payload string (fixture / serialized model).
    payload: str
    model_digest: str
    # Exact theorem the model is claimed to refute.
    target_statement_sha256: str
    target_assumption_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "payload": self.payload,
            "model_digest": self.model_digest,
            "target_statement_sha256": self.target_statement_sha256,
            "target_assumption_ids": list(self.target_assumption_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CounterexampleModelV1:
        kind = str(data.get("model_kind", ""))
        if kind not in ("finite_model", "computable_trace"):
            raise RevmathSchemaError(f"unknown model_kind {kind!r}")
        payload = str(data.get("payload") or "")
        digest = str(data.get("model_digest") or "")
        expected = content_sha(
            {
                "model_kind": kind,
                "payload": payload,
                "target_statement_sha256": str(data["target_statement_sha256"]),
                "target_assumption_ids": _sorted_ids(
                    data.get("target_assumption_ids") or ()
                ),
            }
        )
        if digest != expected:
            raise RevmathSchemaError(
                f"model_digest mismatch (expected {expected!r}, got {digest!r})"
            )
        return cls(
            schema_version=COUNTEREXAMPLE_MODEL_SCHEMA,
            model_id=str(data["model_id"]),
            model_kind=kind,  # type: ignore[arg-type]
            payload=payload,
            model_digest=digest,
            target_statement_sha256=str(data["target_statement_sha256"]),
            target_assumption_ids=_sorted_ids(data.get("target_assumption_ids") or ()),
        )


@dataclass(frozen=True)
class CounterexampleMetaV1:
    """Frozen counterexample sidecar for a weakened theorem."""

    schema_version: str
    meta_id: str
    # Original (stronger) environment before ablation.
    baseline_assumption_ids: tuple[str, ...]
    # Exact weakened theorem under test.
    weakened_statement: str
    weakened_statement_sha256: str
    weakened_assumption_ids: tuple[str, ...]
    removed_assumption_ids: tuple[str, ...]
    hermetic_scenario: HermeticCounterexampleScenario
    # Present only when a checked model/trace exists.
    model: CounterexampleModelV1 | None
    # Proof-search outcome for hermetic fixtures (never promotes to refuted alone).
    search_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meta_id": self.meta_id,
            "baseline_assumption_ids": list(self.baseline_assumption_ids),
            "weakened_statement": self.weakened_statement,
            "weakened_statement_sha256": self.weakened_statement_sha256,
            "weakened_assumption_ids": list(self.weakened_assumption_ids),
            "removed_assumption_ids": list(self.removed_assumption_ids),
            "hermetic_scenario": self.hermetic_scenario,
            "model": self.model.to_dict() if self.model is not None else None,
            "search_status": self.search_status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CounterexampleMetaV1:
        if data.get("schema_version") != COUNTEREXAMPLE_META_SCHEMA:
            raise RevmathSchemaError(
                f"expected schema {COUNTEREXAMPLE_META_SCHEMA!r}, "
                f"got {data.get('schema_version')!r}"
            )
        scenario = str(data.get("hermetic_scenario", ""))
        allowed = (
            "checked_refutation",
            "search_failed",
            "no_counterexample",
            "mismatched_proposition",
            "mismatched_assumptions",
            "timeout",
            "incomplete",
        )
        if scenario not in allowed:
            raise RevmathSchemaError(f"unknown hermetic_scenario {scenario!r}")
        statement = str(data["weakened_statement"])
        digest = str(data["weakened_statement_sha256"])
        _require_statement_digest(statement, digest, label="weakened")
        model_raw = data.get("model")
        model: CounterexampleModelV1 | None
        if model_raw is None:
            model = None
        elif isinstance(model_raw, Mapping):
            model = CounterexampleModelV1.from_dict(model_raw)
        else:
            raise RevmathSchemaError("model must be an object or null")
        search_status = str(data.get("search_status") or "unknown")
        return cls(
            schema_version=COUNTEREXAMPLE_META_SCHEMA,
            meta_id=str(data["meta_id"]),
            baseline_assumption_ids=_sorted_ids(
                data.get("baseline_assumption_ids") or ()
            ),
            weakened_statement=statement,
            weakened_statement_sha256=digest,
            weakened_assumption_ids=_sorted_ids(
                data.get("weakened_assumption_ids") or ()
            ),
            removed_assumption_ids=_sorted_ids(
                data.get("removed_assumption_ids") or ()
            ),
            hermetic_scenario=scenario,  # type: ignore[arg-type]
            model=model,
            search_status=search_status,
        )


@dataclass(frozen=True)
class CounterexampleCheckV1:
    """Independent check of a candidate model against the weakened theorem."""

    matches_proposition: bool
    matches_assumptions: bool
    model_present: bool
    model_replayable: bool
    independently_checked: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches_proposition": self.matches_proposition,
            "matches_assumptions": self.matches_assumptions,
            "model_present": self.model_present,
            "model_replayable": self.model_replayable,
            "independently_checked": self.independently_checked,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class CounterexampleReportV1:
    """Outcome for one counterexample obligation (fail-closed on search alone)."""

    schema_version: str
    task_id: str
    meta_id: str
    weakened_statement_sha256: str
    weakened_assumption_ids: tuple[str, ...]
    check: CounterexampleCheckV1
    outcome: CounterexampleOutcomeName
    judgment: Mapping[str, Any]
    search_status: str
    report_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "meta_id": self.meta_id,
            "weakened_statement_sha256": self.weakened_statement_sha256,
            "weakened_assumption_ids": list(self.weakened_assumption_ids),
            "check": self.check.to_dict(),
            "outcome": self.outcome,
            "judgment": dict(self.judgment),
            "search_status": self.search_status,
            "inferred_impossibility_from_search_failure": False,
            "report_digest": self.report_digest,
        }


def parse_counterexample_meta(data: Mapping[str, Any]) -> CounterexampleMetaV1:
    return CounterexampleMetaV1.from_dict(data)


def build_counterexample_model(
    *,
    model_id: str,
    model_kind: ModelKind,
    payload: str,
    target_statement_sha256: str,
    target_assumption_ids: Sequence[str],
) -> CounterexampleModelV1:
    """Construct a digest-bound model/trace (helper for fixtures/tests)."""

    assumps = _sorted_ids(target_assumption_ids)
    digest = content_sha(
        {
            "model_kind": model_kind,
            "payload": payload,
            "target_statement_sha256": target_statement_sha256,
            "target_assumption_ids": assumps,
        }
    )
    return CounterexampleModelV1(
        schema_version=COUNTEREXAMPLE_MODEL_SCHEMA,
        model_id=model_id,
        model_kind=model_kind,
        payload=payload,
        model_digest=digest,
        target_statement_sha256=target_statement_sha256,
        target_assumption_ids=assumps,
    )


def assert_meta_matches_task(task: RevmathTaskV1, meta: CounterexampleMetaV1) -> None:
    """Task must carry the exact weakened proposition and assumption set."""

    if task.proposition.statement_sha256 != meta.weakened_statement_sha256:
        raise RevmathSchemaError(
            "counterexample task proposition must match weakened theorem "
            f"(task={task.proposition.statement_sha256!r} "
            f"weakened={meta.weakened_statement_sha256!r})"
        )
    if task.proposition.statement != meta.weakened_statement:
        raise RevmathSchemaError(
            "task proposition.statement must equal weakened_statement"
        )
    allowed = _sorted_ids(task.base_theory.allowed_assumption_ids)
    if allowed != meta.weakened_assumption_ids:
        raise RevmathSchemaError(
            "task allowed_assumption_ids must equal weakened_assumption_ids "
            f"(task={allowed!r} meta={meta.weakened_assumption_ids!r})"
        )
    unknown_removed = set(meta.removed_assumption_ids) - set(
        meta.baseline_assumption_ids
    )
    if unknown_removed:
        raise RevmathSchemaError(
            f"removed assumptions not in baseline: {sorted(unknown_removed)}"
        )


def check_counterexample_against_theorem(
    meta: CounterexampleMetaV1,
) -> CounterexampleCheckV1:
    """Independently check model/trace against the exact weakened theorem.

    Search failure alone never yields a positive check — callers must map
    incomplete checks to ``unknown``, not ``refuted``.
    """

    reasons: list[str] = []
    model = meta.model
    model_present = model is not None
    if not model_present:
        reasons.append("no_replayable_model_or_trace")
        return CounterexampleCheckV1(
            matches_proposition=False,
            matches_assumptions=False,
            model_present=False,
            model_replayable=False,
            independently_checked=False,
            rejection_reasons=tuple(reasons),
        )

    assert model is not None
    matches_prop = model.target_statement_sha256 == meta.weakened_statement_sha256
    matches_assumps = (
        _sorted_ids(model.target_assumption_ids) == meta.weakened_assumption_ids
    )
    replayable = bool(model.payload.strip()) and len(model.model_digest) == 64
    if not matches_prop:
        reasons.append("model_targets_different_proposition")
    if not matches_assumps:
        reasons.append("model_targets_different_assumptions")
    if not replayable:
        reasons.append("model_not_replayable")
    if meta.hermetic_scenario in ("search_failed", "no_counterexample"):
        # Even if a stale model sneaks in, search-failure fixtures stay unchecked.
        reasons.append("search_failure_does_not_authorize_refutation")
        return CounterexampleCheckV1(
            matches_proposition=matches_prop,
            matches_assumptions=matches_assumps,
            model_present=True,
            model_replayable=replayable,
            independently_checked=False,
            rejection_reasons=_sorted_ids(reasons),
        )
    if meta.hermetic_scenario == "mismatched_proposition" and matches_prop:
        reasons.append("scenario_requires_proposition_mismatch")
    if meta.hermetic_scenario == "mismatched_assumptions" and matches_assumps:
        reasons.append("scenario_requires_assumption_mismatch")
    independently = matches_prop and matches_assumps and replayable and not reasons
    if meta.hermetic_scenario == "checked_refutation" and independently:
        return CounterexampleCheckV1(
            matches_proposition=True,
            matches_assumptions=True,
            model_present=True,
            model_replayable=True,
            independently_checked=True,
            rejection_reasons=(),
        )
    if independently and meta.hermetic_scenario not in (
        "timeout",
        "incomplete",
        "search_failed",
        "no_counterexample",
    ):
        return CounterexampleCheckV1(
            matches_proposition=True,
            matches_assumptions=True,
            model_present=True,
            model_replayable=True,
            independently_checked=True,
            rejection_reasons=(),
        )
    if not reasons:
        reasons.append("counterexample_not_independently_checked")
    return CounterexampleCheckV1(
        matches_proposition=matches_prop,
        matches_assumptions=matches_assumps,
        model_present=True,
        model_replayable=replayable,
        independently_checked=False,
        rejection_reasons=_sorted_ids(reasons),
    )


def _outcome_name(judgment: Mapping[str, Any]) -> CounterexampleOutcomeName:
    raw = str(judgment.get("outcome", "unknown"))
    if raw in ("witnessed", "refuted", "unknown", "invalid"):
        return raw  # type: ignore[return-value]
    return "unknown"


def evaluate_counterexample(
    task: RevmathTaskV1,
    meta: CounterexampleMetaV1,
    *,
    run_check: Callable[[RevmathTaskV1, CounterexampleMetaV1], Any],
) -> CounterexampleReportV1:
    """Run shared checker, then enforce independent model/trace requirements.

    Never promotes proof-search failure to ``refuted``.
    """

    assert_meta_matches_task(task, meta)
    if task.task_kind != "computable_finite_counterexample":
        raise RevmathSchemaError(
            "evaluate_counterexample requires "
            "task_kind=computable_finite_counterexample, "
            f"got {task.task_kind!r}"
        )
    check = check_counterexample_against_theorem(meta)
    record = run_check(task, meta)
    judgment = record.result.solver_judgment().to_dict()
    outcome = _outcome_name(judgment)

    # Fail closed: search failure / missing model cannot become refutation.
    if (
        meta.search_status in ("failed", "timeout", "incomplete", "no_result")
        and outcome == "refuted"
        and not check.independently_checked
    ):
        judgment = {
            "schema": judgment.get("schema", "solver_judgment/v1"),
            "outcome": "unknown",
            "payload": {
                "kind": "unknown_reason",
                "reason": "incomplete_coverage",
            },
            "checked": False,
        }
        outcome = "unknown"
    if outcome == "refuted" and not check.independently_checked:
        judgment = {
            "schema": judgment.get("schema", "solver_judgment/v1"),
            "outcome": "invalid" if check.rejection_reasons else "unknown",
            "payload": {
                "kind": (
                    "invalid_reason"
                    if check.rejection_reasons
                    else "unknown_reason"
                ),
                "reason": (
                    "payload_mismatch"
                    if check.rejection_reasons
                    else "incomplete_coverage"
                ),
            },
            "checked": bool(check.rejection_reasons),
        }
        outcome = "invalid" if check.rejection_reasons else "unknown"

    payload = {
        "schema_version": COUNTEREXAMPLE_REPORT_SCHEMA,
        "task_id": task.task_id,
        "meta_id": meta.meta_id,
        "weakened_statement_sha256": meta.weakened_statement_sha256,
        "weakened_assumption_ids": list(meta.weakened_assumption_ids),
        "check": check.to_dict(),
        "outcome": outcome,
        "judgment": judgment,
        "search_status": meta.search_status,
        "inferred_impossibility_from_search_failure": False,
    }
    return CounterexampleReportV1(
        schema_version=COUNTEREXAMPLE_REPORT_SCHEMA,
        task_id=task.task_id,
        meta_id=meta.meta_id,
        weakened_statement_sha256=meta.weakened_statement_sha256,
        weakened_assumption_ids=meta.weakened_assumption_ids,
        check=check,
        outcome=outcome,
        judgment=judgment,
        search_status=meta.search_status,
        report_digest=content_sha(payload),
    )


__all__ = [
    "COUNTEREXAMPLE_META_SCHEMA",
    "COUNTEREXAMPLE_MODEL_SCHEMA",
    "COUNTEREXAMPLE_REPORT_SCHEMA",
    "CounterexampleCheckV1",
    "CounterexampleMetaV1",
    "CounterexampleModelV1",
    "CounterexampleReportV1",
    "assert_meta_matches_task",
    "build_counterexample_model",
    "check_counterexample_against_theorem",
    "evaluate_counterexample",
    "parse_counterexample_meta",
]
