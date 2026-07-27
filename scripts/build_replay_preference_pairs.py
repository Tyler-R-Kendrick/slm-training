#!/usr/bin/env python3
"""Build a first replay-preference-pairs corpus and report it honestly.

SLM-418 (DSH5-10)'s extraction (7/7 patterns) and rendering
(``slm_training.harnesses.preference.replay_pairs``, sixth slice) both
exist, but no real captured multi-turn ``ConversationTraceV1`` corpus does
-- confirmed while scoping this slice: `resources/data/` has no persisted
``conversation_trace/v1`` records anywhere, and no harness ingests one (see
``docs/design/dsh5-10-replay-preference-rows.md``). This script does not
invent one. It builds a single small, deterministic, self-contained
conversation exercising several of the seven named patterns (edit-then-
undo, undo-then-redo, checkout-another-state) with the same toy
zero-argument cycling operator this repo's own DSH5-10 test suite uses, so
the pipeline (extract -> render -> write pairs.jsonl) can be run and
measured end to end for real. This is ``fixture_or_scratch`` corpus, the
same honesty class as the smoke-loop recipes in
``docs/design/autotrain-loop-ledger-20260725.md`` -- not a claim about real
usage patterns, model quality, or held-out benefit.

One of the three rows this trace produces (``undo_then_redo``) never
renders: for any deterministic, zero-argument operator, ``redo`` and
"reapply the same operator at the same input state" are, by definition,
the identical text -- no cycle-length trick avoids this; only a genuinely
non-deterministic or argument-bearing operator would. The renderer's own
dedup guard (``render_replay_preference_pair`` in
``slm_training.harnesses.preference.replay_pairs``) correctly declines
rather than emitting a self-contradictory pair -- see this script's own
``pairs_dropped`` in its printed report, and
``test_undo_then_redo_row_declines_a_degenerate_collapse`` in
``tests/test_harnesses/preference/test_replay_pairs.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    append_operator_turn,
    branch_fingerprint,
    build_reference_table,
    checkout_conversation_state,
    create_conversation_trace,
    extract_replay_preference_rows,
    redo_conversation,
    undo_conversation,
)
from slm_training.dsl.operators.conversation import ConversationTraceV1
from slm_training.dsl.pack import DslPack, get_pack
from slm_training.harnesses.preference import write_pairs
from slm_training.harnesses.preference.replay_pairs import render_replay_preference_pairs

_SOURCE = 'root = TextContent(":hero.title")'
_OPERATOR_ID = "openui.demo_cycle_text"
_REQUEST_ID = "replay-pairs-demo"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="scripts.build_replay_preference_pairs",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id=_REQUEST_ID,
    )


def _table(state: OperatorStateV1, branch: str, *, seed: int):
    return build_reference_table(
        request_id=_REQUEST_ID,
        state_digest=state.state_digest,
        branch_digest=branch,
        descriptors=(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha("demo-value"),
                value_type="openui.string",
            ),
        ),
        seed=seed,
    )


def _execute(state: OperatorStateV1, _arguments) -> OperatorMutationV1:
    # A 4-state cycle, not 3: long enough that the checkout-back-to-root
    # target in build_demo_trace() below doesn't coincidentally equal what
    # one more deterministic reapplication of this same operator would
    # produce at the checkout decision state (which would make the
    # rendered pair degenerate -- see the module's own
    # undo_then_redo note for why that's unavoidable for a *different*
    # reason on that pattern, but avoidable here by construction).
    replacements = (
        (":hero.title", ":hero.body"),
        (":hero.body", ":hero.caption"),
        (":hero.caption", ":hero.footer"),
        (":hero.footer", ":hero.title"),
    )
    for before, after in replacements:
        if before in state.source:
            return OperatorMutationV1(
                source=state.source.replace(before, after),
                effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
            )
    raise ValueError("demo.no_transition")


def build_demo_trace() -> tuple[DslPack, OperatorLibraryV1, ConversationTraceV1]:
    """A single scratch trace exercising three of the seven named patterns.

    root -> edit -> undo (edit_then_undo) -> redo (undo_then_redo) -> edit
    -> checkout(root) (checkout_another_state). Deterministic (fixed seeds,
    no randomness); see the module docstring for why this is not real
    captured conversation data.
    """
    base_pack = get_pack("openui")
    root_state = OperatorStateV1.from_source(base_pack, _SOURCE)
    declaration = AstOperatorV1(
        operator_id=_OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, _execute),))
    pack = replace(base_pack, operator_library=library)
    branch = branch_fingerprint(root_state.state_digest, _sha("demo-branch"))
    root_table = _table(root_state, branch, seed=1)
    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=root_table,
        provenance=_provenance(root_state),
    )

    def apply_once(current: ConversationTraceV1, *, seed: int) -> ConversationTraceV1:
        result = library.apply(
            pack, current.current.state, _OPERATOR_ID, (), _provenance(current.current.state)
        )
        assert result.succeeded and result.state is not None
        return append_operator_turn(
            current,
            pack=pack,
            library=library,
            application=result.application,
            output_reference_table=_table(result.state, current.current.branch_digest, seed=seed),
        )

    edited = apply_once(trace, seed=2)
    original_child_id = edited.current_state_id
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))
    redone = redo_conversation(
        undone, target_state_id=original_child_id, provenance=_provenance(undone.current.state)
    )
    edited_again = apply_once(redone, seed=3)
    checked_out = checkout_conversation_state(
        edited_again,
        target_state_id=edited_again.root_state_id,
        provenance=_provenance(edited_again.current.state),
    )
    return pack, library, checked_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/data/preference/replay_demo_pairs.jsonl")
    )
    args = parser.parse_args(argv)

    pack, library, trace = build_demo_trace()
    report = extract_replay_preference_rows(
        trace, pack=pack, library=library, provenance_for=_provenance
    )
    pairs = render_replay_preference_pairs(
        report.rows,
        resolve_node=trace.node,
        pack=pack,
        library=library,
        provenance_for=_provenance,
    )
    n = write_pairs(args.out, pairs)
    print(
        json.dumps(
            {
                "rows": len(report.rows),
                "counts_by_relation": report.counts_by_relation,
                "pairs_rendered": n,
                "pairs_dropped": len(report.rows) - n,
                "out": str(args.out),
                "corpus_kind": "fixture_or_scratch",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
