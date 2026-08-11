"""Goal-action evidence and domain-adequacy classification (PGS-C03 / SLM-501).

Private implementation module; public symbols are re-exported from
``goal_support.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slm_training.data.progspec.goal_constraints import canonical_json_bytes
from slm_training.dsl.solver.goal_support import (
    GOAL_SUPPORT_IMPLEMENTATION_VERSION,
    GoalProfileAuthorityTier,
    GoalSupportProvider,
    GoalSupportResultV1,
    replay_goal_support_result,
)
from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleId, SupportVerdict
from slm_training.dsl.solver.support import EnumerativeSupportOracle, SupportQuery

GOAL_ACTION_EVIDENCE_SCHEMA_VERSION = "goal_action_evidence/v1"
GOAL_DOMAIN_ADEQUACY_REPORT_SCHEMA_VERSION = "goal_domain_adequacy_report/v1"
DOMAIN_ADEQUACY_CLASSIFIER_ID = "goal_support_domain_adequacy/v1"
DOMAIN_ADEQUACY_CLASSIFIER_VERSION = "v1"

_HARD_AUTHORITIES: frozenset[str] = frozenset({"compiler-hard", "verifier-hard"})

GoalActionPartition = Literal["supported", "unsupported", "unknown", "unobserved"]

DomainAdequacyClassification = Literal[
    "domain_adequate_selected_supported",
    "selection_regret",
    "selection_unresolved",
    "domain_inadequate_under_bounds",
    "coverage_unknown",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GoalActionWorkCountersV1(_StrictModel):
    nodes: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    depth: int = Field(default=0, ge=0)
    backtracks: int = Field(default=0, ge=0)
    verifier_calls: int = Field(default=0, ge=0)
    terminal_count: int = Field(default=0, ge=0)


class GoalActionEvidenceV1(_StrictModel):
    """Digest-bound per-action goal-support evidence (no raw terminal content)."""

    schema_version: str = Field(
        default=GOAL_ACTION_EVIDENCE_SCHEMA_VERSION,
        pattern=r"^goal_action_evidence/v1$",
    )
    state_fingerprint: str = Field(min_length=1)
    legal_set_fingerprint: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    profile_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    hole_id: dict[str, Any]
    action_id: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    legal: bool = True
    partition: GoalActionPartition
    base_certificate_digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")
    goal_sidecar_digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")
    authority_tier: GoalProfileAuthorityTier
    replay_ok: bool = False
    work_counters: GoalActionWorkCountersV1 = Field(
        default_factory=GoalActionWorkCountersV1
    )
    obstruction_core_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$|^$"
    )
    stop_reason: str | None = None
    evidence_digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def _check_integrity(self) -> "GoalActionEvidenceV1":
        if not self.legal:
            raise ValueError("grammar legality is independent; legal must remain True")
        if self.partition == "unobserved":
            if self.base_certificate_digest or self.goal_sidecar_digest:
                raise ValueError("unobserved actions must not carry certificate digests")
        else:
            if not self.base_certificate_digest or not self.goal_sidecar_digest:
                raise ValueError("observed actions require certificate and sidecar digests")
        if self.evidence_digest:
            expected = self.compute_digest()
            if self.evidence_digest != expected:
                raise ValueError("evidence_digest does not match canonical payload")
        return self

    def _canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("evidence_digest", None)
        return payload

    def compute_digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self._canonical_payload())
        ).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_digest:
            payload.pop("evidence_digest", None)
        elif not payload.get("evidence_digest"):
            payload["evidence_digest"] = self.compute_digest()
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoalActionEvidenceV1":
        if str(value.get("schema_version")) != GOAL_ACTION_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported GoalActionEvidenceV1 version")
        if not value.get("evidence_digest"):
            raise ValueError("persisted GoalActionEvidenceV1 requires evidence_digest")
        return cls.model_validate(value)


class GoalDomainActionPartitionsV1(_StrictModel):
    supported: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    unobserved: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_disjoint_cover(self) -> "GoalDomainActionPartitionsV1":
        buckets = (
            self.supported,
            self.unsupported,
            self.unknown,
            self.unobserved,
        )
        seen: set[str] = set()
        for bucket in buckets:
            if len(bucket) != len(set(bucket)):
                raise ValueError("partition bucket contains duplicate action ids")
            overlap = seen & set(bucket)
            if overlap:
                raise ValueError(f"partition buckets overlap on {sorted(overlap)}")
            seen.update(bucket)
        for name, bucket in (
            ("supported", self.supported),
            ("unsupported", self.unsupported),
            ("unknown", self.unknown),
            ("unobserved", self.unobserved),
        ):
            if bucket != tuple(sorted(bucket)):
                raise ValueError(f"{name} partition must be canonically sorted")
        return self


class AggregateObstructionSummaryV1(_StrictModel):
    obstruction_core_digests: tuple[str, ...]
    core_atoms_union: tuple[str, ...]

    @model_validator(mode="after")
    def _check_sorted(self) -> "AggregateObstructionSummaryV1":
        if self.obstruction_core_digests != tuple(sorted(self.obstruction_core_digests)):
            raise ValueError("obstruction_core_digests must be sorted")
        if self.core_atoms_union != tuple(sorted(self.core_atoms_union)):
            raise ValueError("core_atoms_union must be sorted")
        if len(self.obstruction_core_digests) != len(set(self.obstruction_core_digests)):
            raise ValueError("obstruction_core_digests must be unique")
        return self


class GoalDomainAdequacyStatsV1(_StrictModel):
    action_count: int = Field(ge=0)
    queried_action_count: int = Field(ge=0)
    terminal_count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    verifier_calls: int = Field(ge=0)
    expanded_nodes: int = Field(ge=0)
    backtracks: int = Field(ge=0)
    stop_reasons: tuple[str, ...] = ()


class GoalDomainAdequacyReportV1(_StrictModel):
    """State-level domain-adequacy diagnosis over one exact legal set."""

    schema_version: str = Field(
        default=GOAL_DOMAIN_ADEQUACY_REPORT_SCHEMA_VERSION,
        pattern=r"^goal_domain_adequacy_report/v1$",
    )
    classification: DomainAdequacyClassification
    state_fingerprint: str = Field(min_length=1)
    legal_set_fingerprint: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    selected_action_id: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    profile_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    problem_id: str = Field(min_length=1)
    pack_id: str = Field(min_length=1)
    constraint_version: str = Field(min_length=1)
    bounds_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    hole_id: dict[str, Any]
    partitions: GoalDomainActionPartitionsV1
    evidence_digests: tuple[str, ...]
    stats: GoalDomainAdequacyStatsV1
    obstruction_summary: AggregateObstructionSummaryV1 | None = None
    classifier_id: str = Field(default=DOMAIN_ADEQUACY_CLASSIFIER_ID, min_length=1)
    classifier_version: str = Field(
        default=DOMAIN_ADEQUACY_CLASSIFIER_VERSION, min_length=1
    )
    implementation_version: str = Field(
        default=GOAL_SUPPORT_IMPLEMENTATION_VERSION, min_length=1
    )
    exact_action_cap: int = Field(ge=0)
    cap_applied: bool
    domain_coverage: Literal["complete", "partial", "none"] = "complete"
    report_digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def _check_integrity(self) -> "GoalDomainAdequacyReportV1":
        if self.evidence_digests != tuple(sorted(self.evidence_digests)):
            raise ValueError("evidence_digests must be sorted")
        if self.stats.stop_reasons != tuple(sorted(set(self.stats.stop_reasons))):
            raise ValueError("stop_reasons must be sorted and unique")
        all_ids = (
            *self.partitions.supported,
            *self.partitions.unsupported,
            *self.partitions.unknown,
            *self.partitions.unobserved,
        )
        if len(all_ids) != self.stats.action_count:
            raise ValueError("action_count must match partition cover size")
        if self.report_digest:
            expected = self.compute_digest()
            if self.report_digest != expected:
                raise ValueError("report_digest does not match canonical payload")
        if self.classification == "domain_inadequate_under_bounds":
            if self.partitions.supported or self.partitions.unknown or self.partitions.unobserved:
                raise ValueError(
                    "domain_inadequate_under_bounds forbids supported/unknown/unobserved"
                )
            if self.cap_applied:
                raise ValueError(
                    "domain_inadequate_under_bounds forbids cap-applied subsets"
                )
            if self.domain_coverage != "complete":
                raise ValueError(
                    "domain_inadequate_under_bounds requires complete domain coverage"
                )
        return self

    def _canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("report_digest", None)
        return payload

    def compute_digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self._canonical_payload())
        ).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_digest:
            payload.pop("report_digest", None)
        elif not payload.get("report_digest"):
            payload["report_digest"] = self.compute_digest()
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoalDomainAdequacyReportV1":
        if str(value.get("schema_version")) != GOAL_DOMAIN_ADEQUACY_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported GoalDomainAdequacyReportV1 version")
        return cls.model_validate(value)


def action_id_from_value(value: DomainValue) -> str:
    """Stable opaque identity for one compiler-legal domain value."""
    return hashlib.sha256(canonical_json_bytes(value.to_dict())).hexdigest()


def legal_set_fingerprint(
    *,
    state_fingerprint: str,
    hole_id: HoleId,
    action_ids: tuple[str, ...],
) -> str:
    payload = {
        "state_fingerprint": state_fingerprint,
        "hole_id": hole_id.to_dict(),
        "action_ids": sorted(action_ids),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def bounds_digest_from_state(state: FiniteDomainState) -> str:
    return hashlib.sha256(
        canonical_json_bytes(state.bounds.to_dict())
    ).hexdigest()


def domain_adequacy_cap_policy_table() -> dict[str, Any]:
    """Document exact action-cap subset selection (deterministic, never inferred)."""
    return {
        "within_cap": "query every legal action through bounded VSS goal support",
        "above_cap": {
            "selection": "first exact_action_cap action_ids in lexicographic sort order",
            "omitted_partition": "unobserved",
            "inference_forbidden": True,
        },
        "greedy_continuation_forbidden": True,
        "grammar_legality_independent": True,
    }


def domain_adequacy_classification_table() -> dict[str, Any]:
    """Exact domain-adequacy classification truth table."""
    return {
        "domain_adequate_selected_supported": {
            "when": "supported nonempty and selected ∈ supported",
        },
        "selection_regret": {
            "when": "supported nonempty and selected ∈ unsupported",
        },
        "selection_unresolved": {
            "when": "supported nonempty and selected ∈ unknown ∪ unobserved",
        },
        "domain_inadequate_under_bounds": {
            "when": (
                "supported empty, unknown empty, unobserved empty, "
                "every legal action replay-valid UNSUPPORTED under hard profile"
            ),
            "forbidden_when": [
                "any unknown",
                "any unobserved",
                "any stale/unreplayed evidence",
                "cap omitted any action",
                "non-hard profile",
            ],
        },
        "coverage_unknown": {"when": "all other cases"},
    }


def _select_query_action_ids(
    action_ids: tuple[str, ...],
    *,
    exact_action_cap: int,
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    ordered = tuple(sorted(action_ids))
    if len(ordered) <= exact_action_cap:
        return ordered, (), False
    return ordered[:exact_action_cap], ordered[exact_action_cap:], True


def _assign_partition(
    *,
    observed: bool,
    verdict: SupportVerdict | None,
    replay_ok: bool,
    hard_profile: bool,
) -> GoalActionPartition:
    if not observed:
        return "unobserved"
    if verdict is SupportVerdict.SUPPORTED and replay_ok:
        return "supported"
    if (
        verdict is SupportVerdict.UNSUPPORTED
        and replay_ok
        and hard_profile
    ):
        return "unsupported"
    return "unknown"


def _classify_domain_adequacy(
    partitions: GoalDomainActionPartitionsV1,
    selected_action_id: str,
    *,
    all_unsupported_replay_valid: bool,
    hard_profile: bool,
    cap_applied: bool,
    domain_complete: bool,
    obstruction_summary: AggregateObstructionSummaryV1 | None,
) -> DomainAdequacyClassification:
    if partitions.supported:
        if selected_action_id in partitions.supported:
            return "domain_adequate_selected_supported"
        if selected_action_id in partitions.unsupported:
            return "selection_regret"
        if (
            selected_action_id in partitions.unknown
            or selected_action_id in partitions.unobserved
        ):
            return "selection_unresolved"

    if (
        not partitions.supported
        and not partitions.unknown
        and not partitions.unobserved
        and partitions.unsupported
        and all_unsupported_replay_valid
        and hard_profile
        and not cap_applied
        and domain_complete
    ):
        return "domain_inadequate_under_bounds"
    return "coverage_unknown"


def _terminal_records(verifier: object) -> tuple[Any, ...]:
    trace = getattr(verifier, "last_trace", None)
    if trace is None:
        return ()
    records = getattr(trace, "records", ())
    return tuple(records)


def _aggregate_obstruction_summary(
    evidence: tuple[GoalActionEvidenceV1, ...],
    *,
    classification: DomainAdequacyClassification,
    hard_profile: bool,
    core_atoms_union: tuple[str, ...],
) -> AggregateObstructionSummaryV1 | None:
    if classification != "domain_inadequate_under_bounds":
        return None
    if not hard_profile:
        return None
    digests: list[str] = []
    for item in evidence:
        if item.partition != "unsupported" or not item.replay_ok:
            return None
        if item.obstruction_core_digest is None:
            return None
        digests.append(item.obstruction_core_digest)
    if not digests:
        return None
    return AggregateObstructionSummaryV1(
        obstruction_core_digests=tuple(sorted(digests)),
        core_atoms_union=core_atoms_union,
    )


def analyze_goal_domain(
    *,
    state: FiniteDomainState,
    hole_id: HoleId,
    selected_action: DomainValue,
    provider: GoalSupportProvider,
    exact_action_cap: int,
) -> tuple[GoalDomainAdequacyReportV1, tuple[GoalActionEvidenceV1, ...]]:
    """Query bounded goal support over one legal set and classify adequacy."""
    if exact_action_cap < 0:
        raise ValueError("exact_action_cap must be non-negative")

    provider.validate_identities(state)
    profile = provider.profile
    profile_digest = provider.profile_digest
    hard_profile = profile.authority_tier in _HARD_AUTHORITIES

    domain = state.domain(hole_id)
    domain_coverage = str(dict(domain.metadata).get("coverage", "complete"))
    if domain_coverage not in {"complete", "partial", "none"}:
        raise ValueError("domain coverage must be complete, partial, or none")
    value_by_id = {action_id_from_value(value): value for value in domain.values}
    action_ids = tuple(sorted(value_by_id))
    if not action_ids:
        raise ValueError("legal action set must be non-empty")

    selected_action_id = action_id_from_value(selected_action)
    if selected_action_id not in value_by_id:
        raise ValueError("selected_action is not in the legal set")

    legal_fp = legal_set_fingerprint(
        state_fingerprint=state.fingerprint,
        hole_id=hole_id,
        action_ids=action_ids,
    )
    query_ids, unobserved_ids, cap_applied = _select_query_action_ids(
        action_ids,
        exact_action_cap=exact_action_cap,
    )

    evidence_rows: list[GoalActionEvidenceV1] = []
    supported: list[str] = []
    unsupported: list[str] = []
    unknown: list[str] = []
    unobserved: list[str] = list(unobserved_ids)

    total_terminals = 0
    total_verifier_calls = 0
    total_nodes = 0
    total_backtracks = 0
    stop_reasons: set[str] = set()
    all_unsupported_replay_valid = True
    obstruction_core_atoms: set[str] = set()

    for action_id in query_ids:
        value = value_by_id[action_id]
        query = SupportQuery(
            state_fingerprint=state.fingerprint,
            hole_id=hole_id,
            candidate=value,
        )
        result, sidecar = provider.check_with_sidecar(state, query)
        replay = replay_goal_support_result(
            result.certificate,
            sidecar,
            state=state,
            provider=provider,
        )
        replay_ok = replay.ok
        partition = _assign_partition(
            observed=True,
            verdict=result.verdict,
            replay_ok=replay_ok,
            hard_profile=hard_profile,
        )
        if partition == "supported":
            supported.append(action_id)
        elif partition == "unsupported":
            unsupported.append(action_id)
            if not replay_ok or result.verdict is not SupportVerdict.UNSUPPORTED:
                all_unsupported_replay_valid = False
        else:
            unknown.append(action_id)
            all_unsupported_replay_valid = False

        terminal_count = len(sidecar.terminal_evidence_digests)
        total_terminals += terminal_count
        action_verifier_calls = (
            result.counters.verifier_calls + replay.counters.verifier_calls
        )
        action_nodes = result.counters.nodes + replay.counters.nodes
        action_tokens = result.counters.tokens + replay.counters.tokens
        action_backtracks = result.counters.backtracks + replay.counters.backtracks
        action_depth = max(result.counters.depth, replay.counters.depth)
        if result.certificate.stop_reason:
            stop_reasons.add(result.certificate.stop_reason)

        if (
            partition == "unsupported"
            and replay_ok
            and sidecar.obstruction_core_digest
        ):
            from slm_training.dsl.solver.goal_support_obstruction import (
                compute_domain_obstruction_core,
            )

            replay_verifier = provider._fresh_verifier()
            core_result = EnumerativeSupportOracle(
                provider.expander, replay_verifier
            ).check(
                state, query
            )
            action_verifier_calls += core_result.counters.verifier_calls
            action_nodes += core_result.counters.nodes
            action_tokens += core_result.counters.tokens
            action_backtracks += core_result.counters.backtracks
            action_depth = max(action_depth, core_result.counters.depth)
            core = compute_domain_obstruction_core(
                certificate=result.certificate,
                terminal_records=_terminal_records(replay_verifier),
                profile=profile,
                state=state,
                replay_ok=True,
            )
            if core is not None:
                obstruction_core_atoms.update(core.core_atoms)

        total_verifier_calls += action_verifier_calls
        total_nodes += action_nodes
        total_backtracks += action_backtracks
        sidecar_bound = GoalSupportResultV1.from_dict(
            sidecar.to_dict(include_digest=True)
        )
        row = GoalActionEvidenceV1(
            state_fingerprint=state.fingerprint,
            legal_set_fingerprint=legal_fp,
            profile_digest=profile_digest,
            hole_id=hole_id.to_dict(),
            action_id=action_id,
            partition=partition,
            base_certificate_digest=result.certificate.digest,
            goal_sidecar_digest=sidecar_bound.digest or sidecar_bound.compute_digest(),
            authority_tier=profile.authority_tier,
            replay_ok=replay_ok,
            work_counters=GoalActionWorkCountersV1(
                nodes=action_nodes,
                tokens=action_tokens,
                depth=action_depth,
                backtracks=action_backtracks,
                verifier_calls=action_verifier_calls,
                terminal_count=terminal_count,
            ),
            obstruction_core_digest=(
                sidecar.obstruction_core_digest
                if partition == "unsupported" and replay_ok
                else None
            ),
            stop_reason=result.certificate.stop_reason,
        )
        evidence_rows.append(
            GoalActionEvidenceV1.from_dict(row.to_dict(include_digest=True))
        )

    for action_id in unobserved_ids:
        row = GoalActionEvidenceV1(
            state_fingerprint=state.fingerprint,
            legal_set_fingerprint=legal_fp,
            profile_digest=profile_digest,
            hole_id=hole_id.to_dict(),
            action_id=action_id,
            partition="unobserved",
            authority_tier=profile.authority_tier,
        )
        evidence_rows.append(
            GoalActionEvidenceV1.from_dict(row.to_dict(include_digest=True))
        )

    partitions = GoalDomainActionPartitionsV1(
        supported=tuple(sorted(supported)),
        unsupported=tuple(sorted(unsupported)),
        unknown=tuple(sorted(unknown)),
        unobserved=tuple(sorted(unobserved)),
    )
    evidence_tuple = tuple(
        sorted(evidence_rows, key=lambda item: item.action_id)
    )
    candidate_summary = _aggregate_obstruction_summary(
        evidence_tuple,
        classification="domain_inadequate_under_bounds",
        hard_profile=hard_profile,
        core_atoms_union=tuple(sorted(obstruction_core_atoms)),
    )
    classification = _classify_domain_adequacy(
        partitions,
        selected_action_id,
        all_unsupported_replay_valid=all_unsupported_replay_valid,
        hard_profile=hard_profile,
        cap_applied=cap_applied,
        domain_complete=domain_coverage == "complete",
        obstruction_summary=candidate_summary,
    )
    obstruction_summary = (
        candidate_summary
        if classification == "domain_inadequate_under_bounds"
        else None
    )
    evidence_digests = tuple(
        sorted(item.evidence_digest or item.compute_digest() for item in evidence_tuple)
    )

    report = GoalDomainAdequacyReportV1(
        classification=classification,
        state_fingerprint=state.fingerprint,
        legal_set_fingerprint=legal_fp,
        selected_action_id=selected_action_id,
        profile_digest=profile_digest,
        problem_id=state.problem_id,
        pack_id=state.pack_id,
        constraint_version=state.constraint_version,
        bounds_digest=bounds_digest_from_state(state),
        hole_id=hole_id.to_dict(),
        partitions=partitions,
        evidence_digests=evidence_digests,
        stats=GoalDomainAdequacyStatsV1(
            action_count=len(action_ids),
            queried_action_count=len(query_ids),
            terminal_count=total_terminals,
            query_count=len(query_ids),
            verifier_calls=total_verifier_calls,
            expanded_nodes=total_nodes,
            backtracks=total_backtracks,
            stop_reasons=tuple(sorted(stop_reasons)),
        ),
        obstruction_summary=obstruction_summary,
        exact_action_cap=exact_action_cap,
        cap_applied=cap_applied,
        domain_coverage=domain_coverage,
    )
    bound = GoalDomainAdequacyReportV1.from_dict(report.to_dict(include_digest=True))
    return bound, evidence_tuple
