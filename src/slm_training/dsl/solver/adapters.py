"""Narrow Torch-free projections into the finite-domain solver state."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol, TypeVar

from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
    SupportVerdict,
)


TopologyNodeT = TypeVar("TopologyNodeT")


class TopologyDomainAdapter(Protocol[TopologyNodeT]):
    """Future model-independent seam for bounded topology-domain projections."""

    def domain_for(
        self, node: TopologyNodeT, *, bounds: SolverBounds
    ) -> HoleDomain:
        """Project one bounded topology node without model scores."""
        ...


def completion_forest_state(
    *,
    prefix_ids: tuple[int, ...] | list[int],
    forest: CompletionForest,
    pack_id: str,
    constraint_version: str,
    bounds: SolverBounds,
) -> FiniteDomainState:
    """Project the current next compiler decision, not a globally solved program."""
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in prefix_ids
    ):
        raise ValueError("completion-forest prefix_ids must be non-negative integers")
    if forest.coverage not in {"complete", "partial", "none"}:
        raise ValueError("completion-forest coverage must be complete, partial, or none")
    for path in forest.paths:
        if not isinstance(path.kind, str) or not path.kind:
            raise ValueError("completion-forest path kind must be non-empty text")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in path.token_ids
        ):
            raise ValueError(
                "completion-forest path token_ids must be non-negative integers"
            )
    prefix = tuple(prefix_ids)
    payload = json.dumps(prefix, separators=(",", ":"))
    prefix_sha = hashlib.sha256(payload.encode()).hexdigest()
    hole_id = HoleId(
        namespace="completion_forest",
        path=(len(prefix), prefix_sha),
        kind="next_semantic_decision",
    )
    values = tuple(
        DomainValue.create(
            "completion_path",
            {"kind": path.kind, "token_ids": list(path.token_ids)},
        )
        for path in forest.paths
    )
    domain = HoleDomain(
        hole_id=hole_id,
        values=values,
        metadata=(
            ("coverage", forest.coverage),
            ("support_verdict", SupportVerdict.UNKNOWN.value),
        ),
    )
    return FiniteDomainState(
        problem_id=f"completion-forest:{prefix_sha}",
        pack_id=pack_id,
        constraint_version=constraint_version,
        bounds=bounds,
        holes=(domain,),
    )
