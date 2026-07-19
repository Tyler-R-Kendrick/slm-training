"""EFS2-04 wiring: a cached cheap-to-expensive verifier cascade.

This module provides the typed stage/result/cache scaffolding for ordering
repository-owned verifier checks by cost and pruning value. It is intentionally
eval-only wiring: it does not replace the production verifier stack in decode
paths, and it does not claim a ship-grade run.

The cascade reuses the existing G0-G12 gate implementations from
``slm_training.data.verify.stack`` as cheap-to-medium stages, and exposes a
small protocol for wrapping arbitrary expensive verifiers (runtime checks,
SMT/solver calls, AgentV-like audits) as later stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Literal

from slm_training.data.verify.stack import (
    GATE_NAMES,
    Gate,
    GateResult,
    GateStatus,
    _canonical,
    _dataflow,
    _grammar,
    _lexical,
    _reference_graph,
    _schema,
)
from slm_training.lineage.records import content_sha


class Verdict(str, Enum):
    """Result lattice for one verifier stage."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class VerifierStageSpec:
    """Static contract for one cascade stage.

    The spec is serializable and content-hashable; it carries everything a
    cache key needs to decide whether a previous result is valid for the
    current stage version, grammar/schema contract, and completeness class.
    """

    stage_id: str
    version: str
    name: str
    input_kind: str = "source"
    dependencies: tuple[str, ...] = ()
    contract_hash: str = ""
    sound_fail: bool = False
    cache_policy: Literal["exact", "summary", "none"] = "exact"
    reason_schema: str = "string"
    skip_stages_on_fail: tuple[str, ...] = ()
    cost_hint: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "version": self.version,
            "name": self.name,
            "input_kind": self.input_kind,
            "dependencies": list(self.dependencies),
            "contract_hash": self.contract_hash,
            "sound_fail": self.sound_fail,
            "cache_policy": self.cache_policy,
            "reason_schema": self.reason_schema,
            "skip_stages_on_fail": list(self.skip_stages_on_fail),
            "cost_hint": self.cost_hint,
        }


@dataclass(frozen=True)
class VerifierResultV1:
    """Outcome of running one stage on one candidate."""

    stage_id: str
    status: Verdict
    sound: bool = False
    reason: str | None = None
    certificate: dict[str, Any] | None = None
    cost: float = 0.0
    cached: bool = False
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "stage_id": self.stage_id,
            "status": self.status.value,
            "sound": self.sound,
            "cost": self.cost,
            "cached": self.cached,
            "skipped": self.skipped,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        if self.certificate is not None:
            result["certificate"] = self.certificate
        return result


StageEvaluator = Callable[[str, dict[str, Any] | None], VerifierResultV1]


@dataclass(frozen=True)
class VerifierStage:
    """A stage spec bound to its evaluator."""

    spec: VerifierStageSpec
    evaluate: StageEvaluator


@dataclass(frozen=True)
class VerifierCascadeResult:
    """Complete cascade run for one candidate."""

    candidate_id: str
    source: str
    results: tuple[VerifierResultV1, ...]
    pruned: bool
    prune_stage_id: str | None
    total_cost: float
    cache_hits: int
    cache_misses: int
    final_status: Verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "results": [r.to_dict() for r in self.results],
            "pruned": self.pruned,
            "prune_stage_id": self.prune_stage_id,
            "total_cost": self.total_cost,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "final_status": self.final_status.value,
        }


@dataclass
class VerifierCache:
    """Content-addressed in-run cache with optional persistence path."""

    entries: dict[str, VerifierResultV1] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def key(
        self,
        source: str,
        spec: VerifierStageSpec,
        context: dict[str, Any] | None,
        pack_version: str,
        environment: dict[str, Any] | None,
    ) -> str:
        payload = {
            "source": source,
            "stage_id": spec.stage_id,
            "stage_version": spec.version,
            "contract_hash": spec.contract_hash,
            "pack_version": pack_version,
            "context": _sorted_context(context),
            "environment": _sorted_context(environment),
        }
        return content_sha(payload)

    def get(self, key: str) -> VerifierResultV1 | None:
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        self.misses += 1
        return None

    def put(self, key: str, result: VerifierResultV1) -> None:
        self.entries[key] = result


def _sorted_context(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {k: value[k] for k in sorted(value)}


class VerifierCascade:
    """Run a ordered list of verifier stages with caching and short-circuiting.

    Rules:

    * Only a ``sound_fail=True`` stage returning ``Verdict.FAIL`` prunes the
      candidate and may skip downstream stages.
    * ``UNKNOWN``, ``ERROR``, and ``NOT_APPLICABLE`` are not rejection.
    * ``ERROR`` results are never cached as proof.
    * Cache keys include stage version, contract hash, pack version, and a
      sorted subset of the evaluation context.
    """

    def __init__(
        self,
        stages: list[VerifierStage],
        *,
        cache: VerifierCache | None = None,
        pack_version: str = "openui-lark/0.2.x",
        environment: dict[str, Any] | None = None,
    ) -> None:
        self.stages = stages
        self.cache = cache or VerifierCache()
        self.pack_version = pack_version
        self.environment = environment or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_version": self.pack_version,
            "environment": self.environment,
            "stages": [stage.spec.to_dict() for stage in self.stages],
        }

    def _run_stage(
        self,
        stage: VerifierStage,
        source: str,
        context: dict[str, Any] | None,
    ) -> VerifierResultV1:
        if stage.spec.cache_policy == "none":
            return self._evaluate(stage, source, context)

        key = self.cache.key(
            source, stage.spec, context, self.pack_version, self.environment
        )
        cached = self.cache.get(key)
        if cached is not None:
            # A cache hit pays only lookup cost, not the original verifier cost.
            return replace(cached, cached=True, cost=0.0)

        result = self._evaluate(stage, source, context)
        if result.status is not Verdict.ERROR:
            self.cache.put(key, result)
        return result

    def _evaluate(
        self,
        stage: VerifierStage,
        source: str,
        context: dict[str, Any] | None,
    ) -> VerifierResultV1:
        try:
            result = stage.evaluate(source, context)
        except Exception as exc:  # noqa: BLE001 - cascade must not crash the caller
            result = VerifierResultV1(
                stage_id=stage.spec.stage_id,
                status=Verdict.ERROR,
                sound=False,
                reason=f"stage raised: {exc}",
            )
        return replace(result, cost=result.cost + stage.spec.cost_hint)

    def evaluate(
        self,
        candidate_id: str,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> VerifierCascadeResult:
        """Cheap-to-expensive cascade with sound-fail pruning."""
        results: list[VerifierResultV1] = []
        skipped: set[str] = set()
        pruned = False
        prune_stage_id: str | None = None
        total_cost = 0.0
        final_status = Verdict.PASS

        for stage in self.stages:
            if stage.spec.stage_id in skipped:
                results.append(
                    VerifierResultV1(
                        stage_id=stage.spec.stage_id,
                        status=Verdict.NOT_APPLICABLE,
                        sound=False,
                        skipped=True,
                    )
                )
                continue

            result = self._run_stage(stage, source, context)
            results.append(result)
            total_cost += result.cost

            if result.status is Verdict.FAIL and stage.spec.sound_fail:
                pruned = True
                prune_stage_id = stage.spec.stage_id
                skipped.update(stage.spec.skip_stages_on_fail)
                final_status = Verdict.FAIL
            elif not pruned and result.status not in (
                Verdict.PASS,
                Verdict.NOT_APPLICABLE,
            ):
                final_status = result.status

        return VerifierCascadeResult(
            candidate_id=candidate_id,
            source=source,
            results=tuple(results),
            pruned=pruned,
            prune_stage_id=prune_stage_id,
            total_cost=total_cost,
            cache_hits=self.cache.hits,
            cache_misses=self.cache.misses,
            final_status=final_status,
        )

    def evaluate_flat(
        self,
        candidate_id: str,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> VerifierCascadeResult:
        """Run every stage without sound-fail short-circuiting.

        This is the authoritative flat-stack baseline used to compare cascade
        pruning. Downstream skip lists are still applied for stages that report
        themselves as ``NOT_APPLICABLE`` via the evaluator, but sound failures do
        not halt execution.
        """
        results: list[VerifierResultV1] = []
        pruned = False
        prune_stage_id: str | None = None
        total_cost = 0.0

        for stage in self.stages:
            result = self._run_stage(stage, source, context)
            results.append(result)
            total_cost += result.cost
            if (
                not pruned
                and result.status is Verdict.FAIL
                and stage.spec.sound_fail
            ):
                pruned = True
                prune_stage_id = stage.spec.stage_id

        final_status = Verdict.PASS
        for result in results:
            if result.status not in (Verdict.PASS, Verdict.NOT_APPLICABLE):
                final_status = result.status
                break

        return VerifierCascadeResult(
            candidate_id=candidate_id,
            source=source,
            results=tuple(results),
            pruned=pruned,
            prune_stage_id=prune_stage_id,
            total_cost=total_cost,
            cache_hits=self.cache.hits,
            cache_misses=self.cache.misses,
            final_status=final_status,
        )


def _gate_status_to_verdict(status: GateStatus) -> Verdict:
    if status is GateStatus.PASS:
        return Verdict.PASS
    if status is GateStatus.FAIL:
        return Verdict.FAIL
    return Verdict.NOT_APPLICABLE


def make_gate_stage(
    gate: Gate,
    evaluator: Callable[[str], GateResult],
    *,
    version: str = "1",
    contract_hash: str = "",
    sound_fail: bool = True,
    skip_stages_on_fail: tuple[str, ...] = (),
    cost_hint: float = 1.0,
) -> VerifierStage:
    """Wrap an existing G0-G12 gate evaluator as a cascade stage."""
    spec = VerifierStageSpec(
        stage_id=gate.value,
        version=version,
        name=GATE_NAMES[gate],
        contract_hash=contract_hash,
        sound_fail=sound_fail,
        cache_policy="exact",
        reason_schema="gate_result",
        skip_stages_on_fail=skip_stages_on_fail,
        cost_hint=cost_hint,
    )

    def evaluate(source: str, _context: dict[str, Any] | None) -> VerifierResultV1:
        gate_result = evaluator(source)
        status = _gate_status_to_verdict(gate_result.status)
        return VerifierResultV1(
            stage_id=spec.stage_id,
            status=status,
            sound=sound_fail and status is Verdict.FAIL,
            reason=gate_result.detail,
        )

    return VerifierStage(spec, evaluate)


def default_openui_cascade(
    *,
    cache: VerifierCache | None = None,
    pack_version: str = "openui-lark/0.2.x",
) -> VerifierCascade:
    """Return a conservative cascade using the existing gate stack.

    Ordering follows the EFS2-04 recommendation: lexical, grammar, schema,
    references (scope/binding), dataflow, canonicalization. Each earlier stage
    skips the later stages on a sound failure.
    """
    return VerifierCascade(
        [
            make_gate_stage(
                Gate.LEXICAL,
                _lexical,
                skip_stages_on_fail=(
                    Gate.GRAMMAR.value,
                    Gate.SCHEMA.value,
                    Gate.REFERENCES.value,
                    Gate.DATAFLOW.value,
                    Gate.CANONICAL.value,
                ),
            ),
            make_gate_stage(
                Gate.GRAMMAR,
                _grammar,
                skip_stages_on_fail=(
                    Gate.SCHEMA.value,
                    Gate.REFERENCES.value,
                    Gate.DATAFLOW.value,
                    Gate.CANONICAL.value,
                ),
            ),
            make_gate_stage(
                Gate.SCHEMA,
                _schema,
                skip_stages_on_fail=(
                    Gate.REFERENCES.value,
                    Gate.DATAFLOW.value,
                    Gate.CANONICAL.value,
                ),
            ),
            make_gate_stage(
                Gate.REFERENCES,
                _reference_graph,
                skip_stages_on_fail=(Gate.DATAFLOW.value, Gate.CANONICAL.value),
            ),
            make_gate_stage(
                Gate.DATAFLOW,
                _dataflow,
                skip_stages_on_fail=(Gate.CANONICAL.value,),
            ),
            make_gate_stage(Gate.CANONICAL, _canonical),
        ],
        cache=cache,
        pack_version=pack_version,
    )
