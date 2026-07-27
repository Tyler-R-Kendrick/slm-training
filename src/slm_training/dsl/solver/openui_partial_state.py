"""OpenUI wiring for PartialStateClassV1 (DSH1-07 / SLM-359).

Bridges two already-canonical OpenUI authorities into the parser-independent
:func:`slm_training.dsl.solver.partial_state.classify_partial_state` contract
without inventing a new parser or grammar authority:

* :func:`slm_training.dsl.parser.validate_output` — the live fragment parser
  behind ``OutputKind`` (``document``/``statement``/``expression``/
  ``typed_node``/``lexical``), already used by scope extraction
  (``data/scope_extract.py``) and opaque-region splicing
  (``dsl/opaque_regions.py``).
* :func:`slm_training.dsl.grammar.fastpath.compiler_draft.build_completion_forest`
  — the compiler-drafted next-action forest already used for constrained
  decode.

Torch-free; not invoked by decode by default. ``classify_openui_source``
handles serialized text (COMPLETE/FRAGMENT/INVALID); ``classify_openui_prefix``
handles an in-progress token prefix (PREFIX/INVALID). Neither function
performs destructive pruning: an ``INVALID`` verdict here is diagnostic, not a
license to discard a candidate elsewhere.
"""

from __future__ import annotations

from typing import Any, Sequence

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    build_completion_forest,
    gold_compiler_decision_positions,
    literal_frame_is_open,
    unresolved_binder_reference_pieces,
)
from slm_training.dsl.parser import ParseError, validate_output
from slm_training.dsl.schema import OUTPUT_KINDS, OutputKind
from slm_training.dsl.solver.partial_state import (
    PartialStateClassV1,
    classify_partial_state,
)

# Fragment kinds tried (in order) after "document" when classifying source
# text; "document" is tried separately/first by classify_openui_source.
_FRAGMENT_KIND_ORDER: tuple[OutputKind, ...] = (
    "statement",
    "expression",
    "typed_node",
    "lexical",
)


def classify_openui_source(
    source: str,
    *,
    document_first: bool = True,
    fragment_kinds: Sequence[OutputKind] = _FRAGMENT_KIND_ORDER,
    fragment_category: str | None = None,
) -> PartialStateClassV1:
    """Classify a serialized OpenUI span as COMPLETE, a named FRAGMENT, or INVALID.

    Tries ``"document"`` first (unless ``document_first`` is ``False``), then
    each kind in ``fragment_kinds`` in order, through the live
    ``slm_training.dsl.parser.validate_output`` fragment parser — the same
    parser opaque-region splicing already trusts. Returns ``INVALID`` when
    every attempted kind is rejected. This function never builds a
    ``CompletionForest`` and therefore never returns ``PREFIX``; use
    :func:`classify_openui_prefix` for an in-progress decode token prefix.
    """
    attempts: list[OutputKind] = []
    if document_first:
        attempts.append("document")
    attempts.extend(kind for kind in fragment_kinds if kind in OUTPUT_KINDS)

    for kind in attempts:
        try:
            validate_output(source, kind, fragment_category)
        except ParseError:
            continue
        return classify_partial_state(
            completed_kind=kind,
            forest_has_paths=False,
            forest_coverage="none",
        )
    return classify_partial_state(
        completed_kind=None,
        forest_has_paths=False,
        forest_coverage="none",
    )


def classify_openui_prefix(
    tokenizer: Any,
    prefix_ids: Sequence[int],
    *,
    max_path_tokens: int = 8,
) -> PartialStateClassV1:
    """Classify an in-progress decode token prefix via the compiler forest.

    Never returns ``COMPLETE``/``FRAGMENT`` — those require serialized source
    text (:func:`classify_openui_source`); a full/named-fragment prefix should
    be re-classified there once decode stops. Raises when the prefix ends
    inside an open literal frame (``LIT_STR``/``LIT_NUM`` without a closing
    ``LIT_END``): per the DSH1-07 stop rule, no cut may split a lexical token
    or opaque marker, so that position is refused outright rather than
    silently classified.
    """
    ids = [int(token_id) for token_id in prefix_ids]
    cut_is_lexical_boundary = not literal_frame_is_open(tokenizer, ids)
    forest = build_completion_forest(tokenizer, ids, max_path_tokens=max_path_tokens)
    decision_positions = set(
        gold_compiler_decision_positions(
            tokenizer, ids, max_path_tokens=max_path_tokens
        )
    )
    singleton_positions = tuple(sorted(set(range(len(ids))) - decision_positions))
    required_roles = unresolved_binder_reference_pieces(tokenizer, ids)
    min_completion_cost = (
        min(len(path.token_ids) for path in forest.paths) if forest.paths else None
    )
    return classify_partial_state(
        completed_kind=None,
        forest_has_paths=bool(forest.paths),
        forest_coverage=forest.coverage,
        frontier=forest.terminals,
        open_nonterminals=forest.terminals,
        required_roles=required_roles,
        singleton_positions=singleton_positions,
        min_completion_cost=min_completion_cost,
        cut_is_lexical_boundary=cut_is_lexical_boundary,
    )


__all__ = ["classify_openui_source", "classify_openui_prefix"]
