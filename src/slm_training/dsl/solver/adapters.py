"""Adapters projecting existing compiler artifacts into solver states.

See ``docs/design/verified-scope-solver.md``. This module is model-independent:
it imports only the compiler projection (:class:`CompletionForest`) and never
``torch`` or any model-inference path. The projection here is the default
reference substrate for the support layer; it does *not* itself compute
``SUPPORTED`` / ``UNSUPPORTED`` / ``UNKNOWN`` verdicts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
)

__all__ = ["CompletionForestProjection", "completion_forest_state"]

# Namespace for the single "next semantic decision" hole a completion forest
# projects to. A later VSS issue may introduce a "topology_node" namespace.
_NAMESPACE = "completion_forest"
_TOKEN_PATH_TAG = "token_path"


def _prefix_fingerprint(prefix_ids: Sequence[int]) -> str:
    """Stable 16-hex digest of the decode prefix, keying the projected hole."""
    ids = [int(token) for token in prefix_ids]
    raw = json.dumps(ids, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def completion_forest_state(
    *,
    prefix_ids: Sequence[int],
    forest: CompletionForest,
    pack_id: str,
    constraint_version: str,
    bounds: SolverBounds,
) -> FiniteDomainState:
    """Project a compiler :class:`CompletionForest` into a one-hole state.

    The forest's next semantic decision becomes a single stable hole, keyed by the
    prefix length and prefix fingerprint. Each candidate carries the *full*
    ``CompletionPath.token_ids`` plus ``kind`` (not only the first token) so
    grammar-forced suffixes stay distinguishable. The forest ``coverage`` guarantee
    is carried verbatim in the hole metadata.

    An empty forest projects to a bottom state (the hole domain is empty). A
    singleton path projects to a structurally solved state **for that projection
    only** — it is not a globally verified program and asserts no ``SUPPORTED``
    verdict. This adapter is not invoked by the default decode path, so existing
    compiler/model behaviour is unchanged.
    """
    if not isinstance(forest, CompletionForest):
        raise ValueError(f"completion_forest_state requires a CompletionForest, got {type(forest).__name__}")
    prefix = tuple(int(token) for token in prefix_ids)
    fingerprint = _prefix_fingerprint(prefix)
    hole_id = HoleId(namespace=_NAMESPACE, path=(len(prefix), fingerprint), kind="next_action")
    values = tuple(
        DomainValue(
            tag=_TOKEN_PATH_TAG,
            token_ids=tuple(int(token) for token in path.token_ids),
            kind=path.kind,
        )
        for path in forest.paths
    )
    hole = HoleDomain(hole_id=hole_id, values=values, metadata=(("coverage", str(forest.coverage)),))
    return FiniteDomainState(
        problem_id=f"{_NAMESPACE}:{fingerprint}",
        pack_id=str(pack_id),
        constraint_version=str(constraint_version),
        bounds=bounds,
        holes=(hole,),
    )


@dataclass(frozen=True)
class CompletionForestProjection:
    """Reference :class:`FiniteDomainProjection` over a compiler forest.

    Holds the projection inputs and defers to :func:`completion_forest_state`.
    Provided as the concrete implementation of the projection seam; topology-node
    projections (a later VSS issue) implement the same ``finite_domain_state()``.
    """

    prefix_ids: tuple[int, ...]
    forest: CompletionForest
    pack_id: str
    constraint_version: str
    bounds: SolverBounds

    def finite_domain_state(self) -> FiniteDomainState:
        return completion_forest_state(
            prefix_ids=self.prefix_ids,
            forest=self.forest,
            pack_id=self.pack_id,
            constraint_version=self.constraint_version,
            bounds=self.bounds,
        )
