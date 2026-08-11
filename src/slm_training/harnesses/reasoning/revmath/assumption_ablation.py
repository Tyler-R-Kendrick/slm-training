"""Assumption-ablation generation + validation (HARN-04 / SLM-544).

Deterministically enumerates a finite ablation lattice from frozen metadata,
validates proofs under reduced environments, and audits dependencies for
hidden reintroduction of removed strength. Minimality claims are scoped only
to the explored candidate set — never global.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskV1,
)

ABLATION_META_SCHEMA = "revmath_assumption_ablation_meta/v1"
ABLATION_CANDIDATE_SCHEMA = "revmath_ablation_candidate/v1"
ABLATION_SEARCH_SCHEMA = "revmath_ablation_search_report/v1"
# Never claim minimality outside the finite lattice we actually searched.
MINIMALITY_CLAIM_SCOPE = "explored_finite_candidate_lattice"

HermeticScenario = Literal[
    "positive",
    "redundant",
    "necessary",
    "timeout",
    "hidden_import",
]

AblationOutcomeName = Literal["witnessed", "refuted", "unknown", "invalid"]


def _sorted_ids(ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(ids)))


def _removed_key(removed: Sequence[str]) -> str:
    # Prefer '+' over ',' so candidate_ids stay within RevmathTaskV1._ID.
    return "__".join(_sorted_ids(removed))


@dataclass(frozen=True)
class AblationMetaV1:
    """Frozen ablation metadata (sidecar; not part of RevmathTaskV1 identity)."""

    schema_version: str
    meta_id: str
    baseline_assumption_ids: tuple[str, ...]
    removable_assumption_ids: tuple[str, ...]
    # lemma_id -> assumption ids whose strength it carries / reintroduces
    import_edges: Mapping[str, tuple[str, ...]]
    strength_bearing_lemmas: tuple[str, ...]
    hermetic_scenario: HermeticScenario
    # Deterministic oracle for hermetic fixtures: removed_key -> status
    oracle_by_removed: Mapping[str, str]
    # Deterministic proof dependency lists for hermetic fixtures
    proof_deps_by_removed: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meta_id": self.meta_id,
            "baseline_assumption_ids": list(self.baseline_assumption_ids),
            "removable_assumption_ids": list(self.removable_assumption_ids),
            "import_edges": {
                lemma: list(assumps) for lemma, assumps in sorted(self.import_edges.items())
            },
            "strength_bearing_lemmas": list(self.strength_bearing_lemmas),
            "hermetic_scenario": self.hermetic_scenario,
            "oracle_by_removed": dict(sorted(self.oracle_by_removed.items())),
            "proof_deps_by_removed": {
                key: list(deps)
                for key, deps in sorted(self.proof_deps_by_removed.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AblationMetaV1:
        if data.get("schema_version") != ABLATION_META_SCHEMA:
            raise RevmathSchemaError(
                f"expected schema {ABLATION_META_SCHEMA!r}, "
                f"got {data.get('schema_version')!r}"
            )
        edges_raw = data.get("import_edges") or {}
        if not isinstance(edges_raw, Mapping):
            raise RevmathSchemaError("import_edges must be an object")
        oracle_raw = data.get("oracle_by_removed") or {}
        deps_raw = data.get("proof_deps_by_removed") or {}
        if not isinstance(oracle_raw, Mapping) or not isinstance(deps_raw, Mapping):
            raise RevmathSchemaError(
                "oracle_by_removed and proof_deps_by_removed must be objects"
            )
        scenario = str(data.get("hermetic_scenario", ""))
        if scenario not in (
            "positive",
            "redundant",
            "necessary",
            "timeout",
            "hidden_import",
        ):
            raise RevmathSchemaError(f"unknown hermetic_scenario {scenario!r}")
        return cls(
            schema_version=ABLATION_META_SCHEMA,
            meta_id=str(data["meta_id"]),
            baseline_assumption_ids=_sorted_ids(data.get("baseline_assumption_ids") or ()),
            removable_assumption_ids=_sorted_ids(
                data.get("removable_assumption_ids") or ()
            ),
            import_edges={
                str(lemma): _sorted_ids(assumps)
                for lemma, assumps in edges_raw.items()
            },
            strength_bearing_lemmas=_sorted_ids(
                data.get("strength_bearing_lemmas") or ()
            ),
            hermetic_scenario=scenario,  # type: ignore[arg-type]
            oracle_by_removed={str(k): str(v) for k, v in oracle_raw.items()},
            proof_deps_by_removed={
                str(k): _sorted_ids(v if isinstance(v, Sequence) else ())
                for k, v in deps_raw.items()
            },
        )


@dataclass(frozen=True)
class AblationCandidateV1:
    """One finite lattice node: exact proposition + reduced assumption set."""

    schema_version: str
    candidate_id: str
    statement_sha256: str
    remaining_assumption_ids: tuple[str, ...]
    removed_assumption_ids: tuple[str, ...]
    candidate_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "statement_sha256": self.statement_sha256,
            "remaining_assumption_ids": list(self.remaining_assumption_ids),
            "removed_assumption_ids": list(self.removed_assumption_ids),
            "candidate_digest": self.candidate_digest,
        }


@dataclass(frozen=True)
class AblationCandidateRecordV1:
    """Outcome for one explored candidate (judgment + audit)."""

    candidate: AblationCandidateV1
    outcome: AblationOutcomeName
    judgment: Mapping[str, Any]
    proof_dependency_ids: tuple[str, ...]
    hidden_reintroduction: tuple[str, ...]
    lattice_minimal_among_explored_successes: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "outcome": self.outcome,
            "judgment": dict(self.judgment),
            "proof_dependency_ids": list(self.proof_dependency_ids),
            "hidden_reintroduction": list(self.hidden_reintroduction),
            "lattice_minimal_among_explored_successes": (
                self.lattice_minimal_among_explored_successes
            ),
            "minimality_claim_scope": MINIMALITY_CLAIM_SCOPE,
        }


@dataclass(frozen=True)
class AblationSearchReportV1:
    """Finite-lattice search report — minimality scoped to explored set only."""

    schema_version: str
    task_id: str
    meta_id: str
    statement_sha256: str
    explored_candidate_count: int
    records: tuple[AblationCandidateRecordV1, ...]
    minimality_claim_scope: str
    report_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "meta_id": self.meta_id,
            "statement_sha256": self.statement_sha256,
            "explored_candidate_count": self.explored_candidate_count,
            "records": [record.to_dict() for record in self.records],
            "minimality_claim_scope": self.minimality_claim_scope,
            "report_digest": self.report_digest,
        }


def parse_ablation_meta(data: Mapping[str, Any]) -> AblationMetaV1:
    return AblationMetaV1.from_dict(data)


def assert_meta_matches_task(task: RevmathTaskV1, meta: AblationMetaV1) -> None:
    baseline = _sorted_ids(task.base_theory.allowed_assumption_ids)
    if baseline != meta.baseline_assumption_ids:
        raise RevmathSchemaError(
            "ablation meta baseline_assumption_ids must equal "
            "task.base_theory.allowed_assumption_ids "
            f"(task={baseline!r} meta={meta.baseline_assumption_ids!r})"
        )
    unknown = set(meta.removable_assumption_ids) - set(meta.baseline_assumption_ids)
    if unknown:
        raise RevmathSchemaError(
            f"removable assumptions not in baseline: {sorted(unknown)}"
        )
    for lemma, assumps in meta.import_edges.items():
        spill = set(assumps) - set(meta.baseline_assumption_ids)
        if spill:
            raise RevmathSchemaError(
                f"import_edges[{lemma!r}] references unknown assumptions "
                f"{sorted(spill)}"
            )
    missing_strength = set(meta.strength_bearing_lemmas) - set(meta.import_edges)
    if missing_strength:
        raise RevmathSchemaError(
            "strength_bearing_lemmas missing import_edges: "
            f"{sorted(missing_strength)}"
        )


def generate_ablation_candidates(
    task: RevmathTaskV1,
    meta: AblationMetaV1,
) -> tuple[AblationCandidateV1, ...]:
    """Enumerate the finite powerset lattice over removable assumptions.

    Includes the empty removal (baseline control). Order is deterministic:
    ascending removed-count, then lexicographic removed ids.
    """

    assert_meta_matches_task(task, meta)
    removable = meta.removable_assumption_ids
    baseline = meta.baseline_assumption_ids
    statement_sha = task.proposition.statement_sha256
    candidates: list[AblationCandidateV1] = []
    for r in range(len(removable) + 1):
        for combo in itertools.combinations(removable, r):
            removed = _sorted_ids(combo)
            remaining = _sorted_ids(
                aid for aid in baseline if aid not in set(removed)
            )
            body = {
                "statement_sha256": statement_sha,
                "remaining_assumption_ids": list(remaining),
                "removed_assumption_ids": list(removed),
                "task_kind": "assumption_ablation",
                "meta_id": meta.meta_id,
            }
            digest = content_sha(body)
            cand_id = (
                f"{task.task_id}.cand.rm_{_removed_key(removed) or 'none'}"
            )
            candidates.append(
                AblationCandidateV1(
                    schema_version=ABLATION_CANDIDATE_SCHEMA,
                    candidate_id=cand_id,
                    statement_sha256=statement_sha,
                    remaining_assumption_ids=remaining,
                    removed_assumption_ids=removed,
                    candidate_digest=digest,
                )
            )
    return tuple(candidates)


def materialize_candidate_task(
    task: RevmathTaskV1,
    candidate: AblationCandidateV1,
) -> RevmathTaskV1:
    """Clone task with reduced allowed assumptions; proposition stays exact."""

    if candidate.statement_sha256 != task.proposition.statement_sha256:
        raise RevmathSchemaError(
            "candidate statement_sha256 must match task proposition "
            "(exact proposition required)"
        )
    return task.model_copy(
        update={
            "task_id": candidate.candidate_id,
            "task_kind": "assumption_ablation",
            "base_theory": task.base_theory.model_copy(
                update={
                    "allowed_assumption_ids": candidate.remaining_assumption_ids
                }
            ),
        }
    )


def audit_hidden_reintroduction(
    *,
    removed_assumption_ids: Sequence[str],
    proof_dependency_ids: Sequence[str],
    import_edges: Mapping[str, Sequence[str]],
    strength_bearing_lemmas: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Detect lemmas that reintroduce removed assumption strength.

    A dependency is a hidden reintroduction when it is (or is listed as) a
    strength-bearing lemma whose import_edges intersect the removed set.
    Removed assumption ids appearing directly as proof dependencies also count.
    """

    removed = set(removed_assumption_ids)
    if not removed:
        return ()
    strength = set(strength_bearing_lemmas or import_edges.keys())
    hits: list[str] = []
    for dep in proof_dependency_ids:
        if dep in removed:
            hits.append(dep)
            continue
        if dep not in strength and dep not in import_edges:
            continue
        carried = set(import_edges.get(dep, ()))
        if carried & removed:
            hits.append(dep)
    return _sorted_ids(hits)


def lattice_minimal_successes(
    successes: Sequence[AblationCandidateV1],
) -> frozenset[str]:
    """Return candidate_ids that are minimal among explored successes only.

    Minimal means no other explored success has a proper subset of remaining
    assumptions (equivalently: a proper superset of removed assumptions).
    """

    remaining_sets = {
        cand.candidate_id: set(cand.remaining_assumption_ids) for cand in successes
    }
    minimal: set[str] = set()
    for cand_id, remaining in remaining_sets.items():
        if any(
            other_id != cand_id and other < remaining
            for other_id, other in remaining_sets.items()
        ):
            continue
        minimal.add(cand_id)
    return frozenset(minimal)


def _outcome_name(judgment: Mapping[str, Any]) -> AblationOutcomeName:
    raw = str(judgment.get("outcome", "unknown"))
    if raw in ("witnessed", "refuted", "unknown", "invalid"):
        return raw  # type: ignore[return-value]
    return "unknown"



def proof_deps_from_capture(capture: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract proof_dependency_ids from a runner capture dict."""

    deps_raw = capture.get("proof_dependency_ids")
    if deps_raw is not None:
        if isinstance(deps_raw, str):
            return _sorted_ids(deps_raw.split(",")) if deps_raw else ()
        if isinstance(deps_raw, Sequence):
            return _sorted_ids(deps_raw)
    stdout = str(capture.get("stdout") or "")
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "proof_dependency_ids" in payload:
            raw = payload.get("proof_dependency_ids") or ()
            if isinstance(raw, str):
                return _sorted_ids(raw.split(",")) if raw else ()
            return _sorted_ids(raw)
    return ()

def evaluate_ablation_lattice(
    task: RevmathTaskV1,
    meta: AblationMetaV1,
    *,
    run_candidate: Callable[[RevmathTaskV1, AblationCandidateV1], Any],
) -> AblationSearchReportV1:
    """Generate candidates, run each via ``run_candidate``, audit, score lattice.

    ``run_candidate`` must reuse the shared revmath runner (no second
    orchestrator). It returns an object with ``.result.solver_judgment()`` and
    optional ``.capture`` carrying ``proof_dependency_ids``.
    """

    assert_meta_matches_task(task, meta)
    if task.task_kind != "assumption_ablation":
        raise RevmathSchemaError(
            f"evaluate_ablation_lattice requires task_kind=assumption_ablation, "
            f"got {task.task_kind!r}"
        )
    candidates = generate_ablation_candidates(task, meta)
    raw_records: list[tuple[AblationCandidateV1, dict[str, Any], tuple[str, ...], tuple[str, ...]]] = []
    for candidate in candidates:
        record = run_candidate(task, candidate)
        judgment = record.result.solver_judgment().to_dict()
        capture = getattr(record, "capture", {}) or {}
        deps = proof_deps_from_capture(capture)
        hidden = audit_hidden_reintroduction(
            removed_assumption_ids=candidate.removed_assumption_ids,
            proof_dependency_ids=deps,
            import_edges=meta.import_edges,
            strength_bearing_lemmas=meta.strength_bearing_lemmas,
        )
        if hidden and judgment.get("outcome") == JudgmentOutcome.WITNESSED.value:
            # Fail closed: never certify a witnessed proof that reintroduces
            # removed strength through stronger library lemmas.
            judgment = {
                "schema": judgment.get("schema", "solver_judgment/v1"),
                "outcome": "invalid",
                "payload": {
                    "kind": "invalid_reason",
                    "reason": "payload_mismatch",
                },
                "checked": True,
            }
        raw_records.append((candidate, judgment, deps, hidden))

    success_cands = tuple(
        cand
        for cand, judgment, _deps, hidden in raw_records
        if _outcome_name(judgment) == "witnessed" and not hidden
    )
    minimal_ids = lattice_minimal_successes(success_cands)

    records: list[AblationCandidateRecordV1] = []
    for cand, judgment, deps, hidden in raw_records:
        outcome = _outcome_name(judgment)
        is_minimal: bool | None
        if outcome == "witnessed" and not hidden:
            is_minimal = cand.candidate_id in minimal_ids
        else:
            is_minimal = None
        records.append(
            AblationCandidateRecordV1(
                candidate=cand,
                outcome=outcome,
                judgment=judgment,
                proof_dependency_ids=deps,
                hidden_reintroduction=hidden,
                lattice_minimal_among_explored_successes=is_minimal,
            )
        )

    payload = {
        "schema_version": ABLATION_SEARCH_SCHEMA,
        "task_id": task.task_id,
        "meta_id": meta.meta_id,
        "statement_sha256": task.proposition.statement_sha256,
        "explored_candidate_count": len(records),
        "records": [r.to_dict() for r in records],
        "minimality_claim_scope": MINIMALITY_CLAIM_SCOPE,
    }
    return AblationSearchReportV1(
        schema_version=ABLATION_SEARCH_SCHEMA,
        task_id=task.task_id,
        meta_id=meta.meta_id,
        statement_sha256=task.proposition.statement_sha256,
        explored_candidate_count=len(records),
        records=tuple(records),
        minimality_claim_scope=MINIMALITY_CLAIM_SCOPE,
        report_digest=content_sha(payload),
    )


__all__ = [
    "ABLATION_CANDIDATE_SCHEMA",
    "ABLATION_META_SCHEMA",
    "ABLATION_SEARCH_SCHEMA",
    "MINIMALITY_CLAIM_SCOPE",
    "AblationCandidateRecordV1",
    "AblationCandidateV1",
    "AblationMetaV1",
    "AblationSearchReportV1",
    "assert_meta_matches_task",
    "audit_hidden_reintroduction",
    "evaluate_ablation_lattice",
    "generate_ablation_candidates",
    "lattice_minimal_successes",
    "materialize_candidate_task",
    "parse_ablation_meta",
    "proof_deps_from_capture",
]
