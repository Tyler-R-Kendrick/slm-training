"""Decode-time forest pruning via certificate-checked exact closure (VSS1-03).

`solver_prune` adapts an authoritative :class:`CompletionForest` to a
:class:`FiniteDomainState`, runs exact closure, and returns the forest pruned to
the **certificate-checked live subset** — a subset of the original paths that
preserves full path identity/forced suffixes and the original forest order.

PGS-E01 (SLM-504) adds default-off ``off`` / ``diagnostic`` / ``certified``
goal-support modes on the same seam: diagnostic runs ``exact_goal_closure`` for
evidence only (forest identity preserved); certified delegates to
``exact_goal_closure`` via :func:`goal_support_certified_prune`.

Honesty invariants (owned by ``docs/design/verified-scope-solver.md``):

* it only prunes when ``forest.coverage == "complete"`` — closure is authoritative
  only over an exhaustive candidate set; a ``partial``/``none`` forest is returned
  unchanged;
* ``UNKNOWN`` candidates are always kept (``keep_and_rank``); only replay-valid
  ``UNSUPPORTED`` candidates are dropped, so a later soft ranker (logits) can never
  reintroduce a removed candidate — it is simply absent from the pruned forest;
* a certified bottom yields an **empty** pruned forest, letting the existing decode
  dead-end/rollback path handle it rather than returning an unverified fallback.

This module is Torch-free and is **not** invoked by default decode — the caller
gates it behind a disabled-by-default flag.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest
from slm_training.dsl.solver.adapters import completion_forest_state
from slm_training.dsl.solver.closure import (
    ClosureResult,
    SupportProvider,
    exact_closure,
)
from slm_training.dsl.solver.state import SolverBounds, SupportVerdict
from slm_training.dsl.solver.support import SupportCertificate
from slm_training.models.decode_stats import GoalSupportTelemetrySink

UNKNOWN_POLICIES = ("keep_and_rank",)
GOAL_SUPPORT_MODES = ("off", "diagnostic", "certified")


def normalize_goal_support_mode(mode: str | None) -> str:
    """Return a canonical tri-state mode; ``None``/empty means ``off``."""
    if mode is None:
        return "off"
    normalized = str(mode).strip().lower()
    if normalized in ("", "none", "false", "0"):
        return "off"
    if normalized not in GOAL_SUPPORT_MODES:
        raise ValueError(
            f"unsupported goal_support_mode {mode!r}; expected one of {GOAL_SUPPORT_MODES}"
        )
    return normalized


def validate_goal_support_config(config: Any, *, context: str = "config") -> None:
    """Fail closed on illegal goal-support mode/profile combinations."""
    mode = normalize_goal_support_mode(getattr(config, "goal_support_mode", "off"))
    if mode == "off":
        return
    profile_mode = getattr(config, "goal_support_profile_mode", None)
    if profile_mode is not None:
        normalized_profile = str(profile_mode).strip().lower()
        if mode == "certified" and normalized_profile != "production_exact":
            raise ValueError(
                f"{context}: goal_support_mode=certified requires "
                "goal_support_profile_mode=production_exact"
            )
        if normalized_profile in {"evaluation_oracle", "advisory_diagnostic"}:
            raise ValueError(
                f"{context}: goal_support_profile_mode={profile_mode!r} is forbidden "
                f"for goal_support_mode={mode!r}"
            )
    compiler_mode = str(getattr(config, "compiler_decode_mode", "off") or "off").lower()
    if mode == "certified" and compiler_mode == "off":
        raise ValueError(
            f"{context}: goal_support_mode=certified requires "
            "compiler_decode_mode != off"
        )


@dataclass(frozen=True)
class GoalSupportDecodeBinding:
    """Structured decode inputs required to construct a :class:`GoalSupportProvider`."""

    ready: bool
    disable_reason: str | None = None
    request: GenerationRequest | None = None
    constraint_set: Any | None = None
    profile: Any | None = None


@dataclass(frozen=True)
class GoalSupportDiagnosticObservation:
    """Observational goal-support evidence; never mutates the completion forest."""

    observed: bool
    reason: str | None = None
    closure_result: ClosureResult | None = None


class InstrumentedGoalSupportProvider:
    """Thin telemetry wrapper over :class:`GoalSupportProvider` for decode paths."""

    def __init__(self, inner: SupportProvider, sink: GoalSupportTelemetrySink) -> None:
        self._inner = inner
        self._sink = sink

    @property
    def backend_version(self) -> str:
        return self._inner.backend_version

    def validate_identities(self, state) -> None:
        self._inner.validate_identities(state)

    def check(self, state, query):
        from slm_training.dsl.solver.goal_support import GoalSupportProvider

        if not isinstance(self._inner, GoalSupportProvider):
            return self._inner.check(state, query)
        result, sidecar = self._inner.check_with_sidecar(state, query)
        self._sink.backtracks += int(result.counters.backtracks)
        if sidecar.obstruction_core_digest:
            self._sink.obstruction_cores += 1
        replay = self._inner.replay(result.certificate, state=state)
        if not replay.ok:
            self._sink.replay_failures += 1
        elif result.verdict is SupportVerdict.UNKNOWN:
            # Replay failure or budget UNKNOWN — never count as unsupported/pruned.
            pass
        return result

    def replay(self, certificate, *, state):
        replay = self._inner.replay(certificate, state=state)
        if not replay.ok:
            self._sink.replay_failures += 1
        return replay


def instrument_goal_support_provider(
    provider: SupportProvider,
    sink: GoalSupportTelemetrySink,
) -> SupportProvider:
    """Wrap a goal-support provider for request-local telemetry when enabled."""
    from slm_training.dsl.solver.goal_support import GoalSupportProvider

    if isinstance(provider, GoalSupportProvider):
        return InstrumentedGoalSupportProvider(provider, sink)
    return provider


def build_goal_support_decode_binding(
    request: GenerationRequest | None,
    *,
    pack_id: str = "openui",
) -> GoalSupportDecodeBinding:
    """Compile request/profile inputs or return an explicit disable reason."""
    if request is None:
        return GoalSupportDecodeBinding(
            ready=False, disable_reason="missing_generation_request"
        )
    try:
        from slm_training.data.progspec.prompt_requirements import (
            PromptSemanticRequirementsV1,
        )
        from slm_training.data.progspec.synthesis_problem import (
            PackIdentityV1,
            VerifiedSynthesisProblemV1,
        )
        from slm_training.data.semantic_plan.requirements_compile import (
            compile_goal_constraints,
        )
        from slm_training.dsl.pack import get_pack
        from slm_training.dsl.solver.goal_support import (
            GoalVerifierProfileV1,
            validate_profile_against_constraint_set,
        )

        pack = get_pack(pack_id)
        problem = VerifiedSynthesisProblemV1(
            problem_id="decode",
            pack_identity=PackIdentityV1(
                pack_id=pack.pack_id,
                contract_version=getattr(pack, "contract_version", 5),
            ),
            requirements=PromptSemanticRequirementsV1(facts=()),
        )
        constraint_set = compile_goal_constraints(problem, request, pack)
        profile = GoalVerifierProfileV1(
            profile_id="decode/production_exact",
            mode="production_exact",
            problem_digest=constraint_set.problem_digest,
            constraint_set_digest=constraint_set.digest,
            pack_identity=problem.pack_identity,
            pack_identity_digest=constraint_set.pack_identity_digest,
            required_constraint_ids=constraint_set.hard_constraint_ids,
            required_gates=(),
            required_evaluators=(),
            metric_identities=(),
            authority_tier="compiler-hard",
        )
        profile = GoalVerifierProfileV1.from_dict(
            profile.model_copy(update={"digest": profile.compute_digest()}).to_dict()
        )
        validate_profile_against_constraint_set(profile, constraint_set)
        return GoalSupportDecodeBinding(
            ready=True,
            disable_reason=None,
            request=request,
            constraint_set=constraint_set,
            profile=profile,
        )
    except Exception as exc:  # noqa: BLE001 — surface construction failure honestly
        return GoalSupportDecodeBinding(
            ready=False,
            disable_reason=f"{type(exc).__name__}:{exc}",
        )


def _value_key(value) -> tuple[str, tuple[int, ...]]:
    payload = value.payload
    return (
        str(payload.get("kind", "")),
        tuple(int(token) for token in payload.get("token_ids", ())),
    )


def _prune_with_closure(
    forest: CompletionForest,
    prefix_ids,
    provider: SupportProvider,
    *,
    pack_id: str,
    constraint_version: str,
    bounds: SolverBounds,
    unknown_policy: str = "keep_and_rank",
    state=None,
    cache: dict | None = None,
    certificate_store: dict[str, SupportCertificate] | None = None,
    closure_fn: Callable[..., ClosureResult] = exact_closure,
) -> tuple[CompletionForest, ClosureResult | None]:
    """Return ``(pruned_forest, closure_result)`` for one decode decision.

    A forest path is removed **only** when it was in the closure's input domain and
    got certified-removed (``removed = original_domain − survivors``); a candidate
    the oracle never considered is always kept. ``state`` may be a pre-built
    projection (e.g. the provider's own root state) so fingerprints line up with
    the oracle; when ``None`` it is projected from ``forest``. ``closure_result`` is
    ``None`` when no closure ran (non-``complete`` coverage or empty forest).
    """
    if unknown_policy not in UNKNOWN_POLICIES:
        raise ValueError(f"unsupported solver_unknown_policy: {unknown_policy!r}")
    if forest.coverage != "complete" or not forest.paths:
        return forest, None

    if state is None:
        state = completion_forest_state(
            prefix_ids=tuple(int(token) for token in prefix_ids),
            forest=forest,
            pack_id=pack_id,
            constraint_version=constraint_version,
            bounds=bounds,
        )
    result = closure_fn(state, provider, cache=cache, certificate_store=certificate_store)
    original = state.holes[0].values if state.holes else ()
    survivors = result.state.holes[0].values if result.state.holes else ()
    # Only paths the closure actually certified-removed are dropped.
    removed_keys = {_value_key(v) for v in original} - {_value_key(v) for v in survivors}
    if not removed_keys:
        return forest, result  # nothing certified-removed -> keep identity
    ordered = tuple(
        path
        for path in forest.paths
        if (path.kind, tuple(path.token_ids)) not in removed_keys
    )
    if len(ordered) == len(forest.paths):
        return forest, result
    return CompletionForest(ordered, forest.coverage, forest.terminals), result


def solver_prune(
    forest: CompletionForest,
    prefix_ids,
    provider: SupportProvider,
    *,
    pack_id: str,
    constraint_version: str,
    bounds: SolverBounds,
    unknown_policy: str = "keep_and_rank",
    state=None,
    cache: dict | None = None,
    certificate_store: dict[str, SupportCertificate] | None = None,
) -> tuple[CompletionForest, ClosureResult | None]:
    """Return ``(pruned_forest, closure_result)`` for one decode decision."""
    return _prune_with_closure(
        forest,
        prefix_ids,
        provider,
        pack_id=pack_id,
        constraint_version=constraint_version,
        bounds=bounds,
        unknown_policy=unknown_policy,
        state=state,
        cache=cache,
        certificate_store=certificate_store,
        closure_fn=exact_closure,
    )


def goal_support_certified_prune(
    forest: CompletionForest,
    prefix_ids,
    provider: SupportProvider,
    *,
    pack_id: str,
    constraint_version: str,
    bounds: SolverBounds,
    unknown_policy: str = "keep_and_rank",
    state=None,
    cache: dict | None = None,
    certificate_store: dict[str, SupportCertificate] | None = None,
) -> tuple[CompletionForest, ClosureResult | None]:
    """Certified goal-support pruning via ``exact_goal_closure`` authority guard."""
    from slm_training.dsl.solver.goal_support import (
        GoalSupportProvider,
        exact_goal_closure,
    )

    if not isinstance(provider, GoalSupportProvider):
        raise ValueError("goal_support_certified_prune requires GoalSupportProvider")
    return _prune_with_closure(
        forest,
        prefix_ids,
        provider,
        pack_id=pack_id,
        constraint_version=constraint_version,
        bounds=bounds,
        unknown_policy=unknown_policy,
        state=state,
        cache=cache,
        certificate_store=certificate_store,
        closure_fn=exact_goal_closure,
    )


def goal_support_diagnostic_observe(
    forest: CompletionForest,
    prefix_ids,
    provider: SupportProvider,
    *,
    pack_id: str,
    constraint_version: str,
    bounds: SolverBounds,
    state=None,
    cache: dict | None = None,
    certificate_store: dict[str, SupportCertificate] | None = None,
) -> tuple[CompletionForest, GoalSupportDiagnosticObservation]:
    """Run goal-support closure for counters/evidence; return the original forest."""
    from slm_training.dsl.solver.goal_support import (
        GoalSupportProvider,
        exact_goal_closure,
    )

    if not isinstance(provider, GoalSupportProvider):
        raise ValueError("goal_support_diagnostic_observe requires GoalSupportProvider")
    if forest.coverage != "complete" or not forest.paths:
        return forest, GoalSupportDiagnosticObservation(
            observed=False, reason="incomplete_forest"
        )
    if state is None:
        state = completion_forest_state(
            prefix_ids=tuple(int(token) for token in prefix_ids),
            forest=forest,
            pack_id=pack_id,
            constraint_version=constraint_version,
            bounds=bounds,
        )
    try:
        result = exact_goal_closure(
            state,
            provider,
            cache=cache,
            certificate_store=certificate_store,
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic reports honestly, never prunes
        return forest, GoalSupportDiagnosticObservation(
            observed=False,
            reason=f"closure_failed:{type(exc).__name__}:{exc}",
        )
    return forest, GoalSupportDiagnosticObservation(
        observed=True,
        reason=None,
        closure_result=result,
    )
