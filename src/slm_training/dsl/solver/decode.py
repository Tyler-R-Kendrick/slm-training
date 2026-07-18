"""Decode-time forest pruning via certificate-checked exact closure (VSS1-03).

`solver_prune` adapts an authoritative :class:`CompletionForest` to a
:class:`FiniteDomainState`, runs exact closure, and returns the forest pruned to
the **certificate-checked live subset** — a subset of the original paths that
preserves full path identity/forced suffixes and the original forest order.

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

from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest
from slm_training.dsl.solver.adapters import completion_forest_state
from slm_training.dsl.solver.closure import ClosureResult, SupportProvider, exact_closure
from slm_training.dsl.solver.state import SolverBounds
from slm_training.dsl.solver.support import SupportCertificate

UNKNOWN_POLICIES = ("keep_and_rank",)


def _value_key(value) -> tuple[str, tuple[int, ...]]:
    payload = value.payload
    return (
        str(payload.get("kind", "")),
        tuple(int(token) for token in payload.get("token_ids", ())),
    )


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
    result = exact_closure(
        state, provider, cache=cache, certificate_store=certificate_store
    )
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
