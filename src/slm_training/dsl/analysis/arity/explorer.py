"""Bounded domain exploration for arity analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slm_training.dsl.analysis.arity.canonical import canonicalize_ast
from slm_training.dsl.analysis.arity.report import ContinuationSummary
from slm_training.dsl.analysis.arity.types import AnalysisBounds, StateSignature

if TYPE_CHECKING:
    from slm_training.dsl.grammar.backends.types import GrammarBackend


_FRAME_VERSION = "cap0-02-v1"


def _state_signature(root: object) -> StateSignature:
    atoms = canonicalize_ast(root)
    return StateSignature(
        version=_FRAME_VERSION,
        generation_order="depth-first-left-to-right",
        atoms=atoms,
    )


def _bounded_sources(
    backend: GrammarBackend,
    bounds: AnalysisBounds,
    seed_sources: list[str] | None = None,
) -> list[str]:
    """Return sources to analyze within the declared bounds.

    When a pack corpus generator is available, use it to produce a bounded set
    of programs; otherwise fall back to the caller-supplied sources.
    """
    sources = list(seed_sources or [])
    # For very small bounds, prefer the provided seeds.
    if sources:
        return sources[: bounds.max_ast_nodes]

    # Attempt to use a pack generator if the DSL has one.
    try:
        from slm_training.dsl.pack import get_pack

        pack = get_pack(backend.dsl_id)
        generator = getattr(pack, "corpus_generator", None)
        if generator is not None:
            records = generator(count=bounds.max_ast_nodes, seed=0)
            for record in records:
                src = getattr(record, "openui", None) or getattr(record, "source", None)
                if isinstance(src, str):
                    sources.append(src)
    except Exception:
        # Pack may not exist for arbitrary DSLs; continue with whatever seeds we have.
        pass

    return sources


def explore(
    backend: GrammarBackend,
    bounds: AnalysisBounds,
    seed_sources: list[str] | None = None,
) -> list[ContinuationSummary]:
    """Explore a bounded domain and return continuation summaries.

    This initial implementation materializes complete-program signatures. A
    future iteration can extend the BFS to partial prefixes using the grammar's
    incremental engine.
    """
    sources = _bounded_sources(backend, bounds, seed_sources)
    summaries: list[ContinuationSummary] = []
    for source in sources:
        program = backend.parse(source)
        root = program.root
        if root is None:
            continue
        sig = _state_signature(root)
        summaries.append(
            ContinuationSummary(
                state_signature=sig,
                next_actions=(),
                terminal=True,
                complete=True,
            )
        )
    return summaries
