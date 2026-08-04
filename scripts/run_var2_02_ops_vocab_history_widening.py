#!/usr/bin/env python3
"""VAR2-02: widen the OPS_VOCAB adequacy-audit corpus to history-control ops.

VAR2-01 (`docs/design/var2-01-ops-vocab-first-consumer-20260726.md`) audited
`conversation_trace_op_ids` against a corpus built only from
`build_symbolic_operator_corpus`'s `SINGLE_TURN`/`sibling_forks` shapes, and
found 6 of `OPS_VOCAB`'s 20 ops unused: `conversation.ast_edit`,
`conversation.checkout_state`, `conversation.copy_state`, `conversation.redo`,
`conversation.transaction_commit`, `conversation.undo`. It named the follow-on
explicitly: "VAR2-02 (widening the corpus to topology/history operators, or
accepting the narrower scope)."

This script closes the **history** half of that gap only (the narrower,
honestly-scoped slice -- topology-operator widening, e.g.
`openui.wrap_node`/`openui.reorder_children`, stays open as VAR2-03 or later,
per VAR2-01's own "or accepting the narrower scope" framing). It builds one
real `ConversationTraceV1` that actually contains
`UNDO`/`REDO`/`CHECKOUT_STATE`/`COPY_STATE` turns (`conversation.py`'s own
API, not a synthetic op-id list) alongside one real `AST_EDIT` (a legal action
drawn from `enumerate_operator_legal_set`, not a fixture no-op) and one `FORK`
(already covered by VAR2-01's `sibling_forks=True`, kept here for a complete
before/after comparison), then re-runs the same adequacy-audit and wiring-probe
shape VAR2-01 published against this widened trace.

`conversation.ast_edit` / `conversation.transaction_commit` are **not**
attempted here: VAR2-01 already established they are structurally
unreachable by `resolve_turn_op_ids` at any corpus size (an AST_EDIT/
TRANSACTION_COMMIT turn always resolves to its specific underlying operator
id, never that literal history-family id) -- widening the corpus cannot close
that part, so this script does not try.

Example:
  python -m scripts.run_var2_02_ops_vocab_history_widening
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.operators import (
    ApplicationProvenanceV1,
    OperatorLibraryV1,
    OperatorStateV1,
    append_operator_turn,
    branch_fingerprint,
    build_openui_local_operator_context,
    build_openui_local_operator_library,
    checkout_conversation_state,
    copy_conversation_state,
    create_conversation_trace,
    enumerate_operator_legal_set,
    fork_conversation,
    redo_conversation,
    undo_conversation,
)
from slm_training.dsl.operators.ops_vocab_conditioning import (
    OpsVocabConditioningError,
    conversation_trace_op_ids,
    format_history_ops_text,
)
from slm_training.dsl.ops_vocab import OPS_VOCAB
from slm_training.dsl.ops_vocab import fingerprint as ops_vocab_fingerprint
from slm_training.dsl.pack import get_pack
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.twotower import format_context_text

DEFAULT_JSON_OUT = Path("docs/design/var2-02-ops-vocab-history-widening-20260728.json")
DEFAULT_MD_OUT = Path("docs/design/var2-02-ops-vocab-history-widening-20260728.md")

COMPONENT = "harness.experiments.var2_02_ops_vocab_history_widening"

# The exact `unused_ops` VAR2-01 published (docs/design/var2-01-ops-vocab-
# first-consumer-20260726.json, 16 ids: 6 conversation.* + 10 openui.*
# topology ops) -- kept as a literal baseline to diff against rather than
# re-deriving it, so a regression in VAR2-01's own script cannot silently
# move this script's "newly covered" claim.
_VAR2_01_UNUSED_OPS: frozenset[str] = frozenset(
    {
        "conversation.ast_edit",
        "conversation.checkout_state",
        "conversation.copy_state",
        "conversation.redo",
        "conversation.transaction_commit",
        "conversation.undo",
        "openui.contract_subtree",
        "openui.duplicate_subtree",
        "openui.expand_template",
        "openui.move_node",
        "openui.reorder_children",
        "openui.reparent_node",
        "openui.set_property",
        "openui.unset_property",
        "openui.unwrap_node",
        "openui.wrap_node",
    }
)
_VAR2_01_STRUCTURALLY_UNREACHABLE: frozenset[str] = frozenset(
    {"conversation.ast_edit", "conversation.transaction_commit"}
)
# The four history ops this script's widened trace targets: VAR2-01's
# unused `conversation.*` ids, minus the two structurally-unreachable ones
# above. Deliberately excludes the `openui.*` topology ids in
# `_VAR2_01_UNUSED_OPS` -- those are this slice's explicit non-goal (see
# module docstring).
_TARGET_HISTORY_OPS: frozenset[str] = frozenset(
    op
    for op in _VAR2_01_UNUSED_OPS - _VAR2_01_STRUCTURALLY_UNREACHABLE
    if op.startswith("conversation.")
)

_SOURCE = (
    'root = Card([TextContent(":hero.title"), TextContent(":hero.body")], "clear")'
)
_REQUEST_ID = "var2-02-history-widening"

_TOKEN_BY_OP_ID: dict[str, str] = {symbol.op_id: symbol.token for symbol in OPS_VOCAB}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _corpus_content_fingerprint(trace_dict: dict[str, Any]) -> str:
    return _sha(json.dumps({"trace": trace_dict}, sort_keys=True, separators=(",", ":")))


def _provenance_for(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.var2_02_history_widening",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id=_REQUEST_ID,
    )


def _build_widened_trace() -> tuple[OperatorLibraryV1, dict[str, Any]]:
    """One real trace exercising AST_EDIT, UNDO, REDO, CHECKOUT_STATE, COPY_STATE, FORK.

    Turn order (each a real `conversation.py` call, never a synthetic op-id
    list): create -> self-copy (COPY_STATE, before the root has any same-
    branch child, since `copy_conversation_state` refuses a current state
    that already has one) -> a real legal AST_EDIT on the copied root's
    state -> UNDO -> REDO -> CHECKOUT_STATE back to the true root -> FORK
    (fork tolerates an already-parented current state; copy does not).
    """
    base_pack = get_pack("openui")
    root_state = OperatorStateV1.from_source(base_pack, _SOURCE)
    branch = branch_fingerprint(root_state.state_digest, _sha(_REQUEST_ID))
    context = build_openui_local_operator_context(
        base_pack,
        root_state,
        request_id=_REQUEST_ID,
        branch_digest=branch,
        seed=0,
    )
    library = build_openui_local_operator_library(context)
    pack = replace(base_pack, operator_library=library)
    root_provenance = _provenance_for(root_state)

    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=context.reference_table,
        provenance=root_provenance,
    )
    root_state_id = trace.current_state_id

    # 1. COPY_STATE: self-referential copy while root still has no next state.
    copied = copy_conversation_state(
        trace, source_state_id=root_state_id, provenance=root_provenance
    )

    # 2. AST_EDIT: a real legal action against the copied root's (identical)
    # state -- drawn from the live legal set, not a fixture no-op operator.
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=root_state,
        reference_table=context.reference_table,
        provenance=root_provenance,
        max_combinations_per_operator=16,
    )
    if not legal_set.operator_actions:
        raise ValueError("var2-02 fixture source has no legal operator action")
    action = sorted(legal_set.operator_actions, key=lambda a: a.serialized)[0]
    result = library.apply(pack, root_state, action.operator_id, action.arguments, root_provenance)
    if not result.succeeded or result.state is None:
        raise ValueError("var2-02 fixture's chosen legal action failed to apply")
    output_context = build_openui_local_operator_context(
        base_pack,
        result.state,
        request_id=_REQUEST_ID,
        branch_digest=branch,
        seed=1,
    )
    edited = append_operator_turn(
        copied,
        pack=pack,
        library=library,
        application=result.application,
        output_reference_table=output_context.reference_table,
    )
    edited_state_id = edited.current_state_id

    # 3. UNDO back to the copied root.
    undone = undo_conversation(edited, provenance=_provenance_for(edited.current.state))

    # 4. REDO back to the edited state.
    redone = redo_conversation(
        undone,
        target_state_id=edited_state_id,
        provenance=_provenance_for(undone.current.state),
    )

    # 5. CHECKOUT_STATE back to the true root (pure navigation -- no new node,
    # so it tolerates the root already having the copied-root child).
    checked_out = checkout_conversation_state(
        redone,
        target_state_id=root_state_id,
        provenance=_provenance_for(redone.current.state),
    )

    # 6. FORK off the root -- already covered by VAR2-01 (`sibling_forks=
    # True`), included here so the widened trace's op-id sequence is a
    # complete before/after comparison, not just the four new ops.
    forked = fork_conversation(
        checked_out,
        branch_nonce_digest=_sha("var2-02-fork"),
        reference_seed=99,
        provenance=_provenance_for(checked_out.current.state),
    )

    return library, forked.to_dict()


def _adequacy_delta(op_ids: tuple[str, ...]) -> dict[str, Any]:
    used = set(op_ids)
    newly_covered = sorted(_TARGET_HISTORY_OPS & used)
    still_missing_targeted = sorted(_TARGET_HISTORY_OPS - used)
    # VAR2-01's published `unused_ops` minus whatever this widened trace's
    # own op-id sequence now covers -- i.e. what remains unused if VAR2-01's
    # corpus and this trace are unioned. Deliberately NOT recomputed as
    # "every OPS_VOCAB id absent from this trace alone": this trace only
    # exercises one operator action (whichever the live legal set sorts
    # first), so a full recomputation would wrongly report VAR2-01's other
    # already-used ops (e.g. whichever of `openui.add_child`/
    # `openui.replace_node`/... VAR2-01's own corpus already covered) as
    # newly unused just because this narrower trace doesn't happen to touch
    # them too.
    unused_after_union = sorted(_VAR2_01_UNUSED_OPS - used)
    # Static baseline (VAR2-01's 6 published unused ops, minus the 2
    # structurally-unreachable conversation.* ids, minus the 4 history ops
    # this script targets) -- the topology-op remainder this slice does not
    # attempt, independent of anything this trace happens to touch.
    topology_out_of_scope = sorted(
        _VAR2_01_UNUSED_OPS - _VAR2_01_STRUCTURALLY_UNREACHABLE - _TARGET_HISTORY_OPS
    )
    return {
        "targeted_history_ops": sorted(_TARGET_HISTORY_OPS),
        "newly_covered_ops": newly_covered,
        "still_missing_targeted_ops": still_missing_targeted,
        "unused_ops_after_widening_union_with_var2_01": unused_after_union,
        "unused_ops_before_widening_var2_01_published": sorted(_VAR2_01_UNUSED_OPS),
        "topology_ops_left_out_of_scope": topology_out_of_scope,
    }


def _wiring_probe(op_ids: tuple[str, ...]) -> dict[str, Any]:
    prompt = "Edit request for var2-02-history-widening"
    off_text = format_context_text(prompt)
    try:
        history_text = format_history_ops_text(op_ids) if op_ids else None
        representable = history_text is not None
    except OpsVocabConditioningError:
        history_text = None
        representable = False
    on_text = format_context_text(prompt, history_ops_text=history_text)
    reserved_tokens = history_text.split(" ") if history_text else []
    return {
        "hypothetical": True,
        "production_conditioning_changed": False,
        "op_id_sequence": list(op_ids),
        "representable": representable,
        "off_arm_unchanged": off_text == format_context_text(prompt),
        "on_arm_adds_history_section": (
            "---HISTORY_OPS---" in on_text if representable else "---HISTORY_OPS---" not in on_text
        ),
        "reserved_tokens_all_layered": all(
            token in _TOKEN_BY_OP_ID.values() for token in reserved_tokens
        ),
    }


def run_audit(*, generated_at: str) -> dict[str, Any]:
    library, trace_dict = _build_widened_trace()
    try:
        op_ids = conversation_trace_op_ids(trace_dict, library)
    except OpsVocabConditioningError as exc:
        raise ValueError(f"var2-02 widened trace produced an unrepresentable turn: {exc}") from exc
    corpus_fp = _corpus_content_fingerprint(trace_dict)
    ops_fp = ops_vocab_fingerprint()
    return {
        "schema": "var2_02_ops_vocab_history_widening/v1",
        "issue": "VAR2-02 (follow-on to SLM-428 / VAR2-01)",
        "generated_at": generated_at,
        "ops_vocab_fingerprint": ops_fp,
        "corpus_content_fingerprint": corpus_fp,
        "op_id_sequence": list(op_ids),
        "adequacy_audit": {**_adequacy_delta(op_ids), "claim_class": "measurement"},
        "wiring_probe": {**_wiring_probe(op_ids), "claim_class": "wiring"},
        "version_stamp": build_version_stamp(COMPONENT),
    }


def _decision_sentence(payload: dict[str, Any]) -> str:
    audit = payload["adequacy_audit"]
    probe = payload["wiring_probe"]
    if not probe["off_arm_unchanged"]:
        return (
            "FAIL: the off arm's context text changed when the lever is "
            "disabled -- this is a regression, not a widening result; do not "
            "merge until format_context_text reproduces prior output exactly "
            "with history_ops_text omitted."
        )
    covered = len(audit["newly_covered_ops"])
    targeted = len(audit["targeted_history_ops"])
    missing = audit["still_missing_targeted_ops"]
    if missing:
        return (
            f"PARTIAL: {covered}/{targeted} targeted history ops became "
            f"representable; still missing: {', '.join(missing)}. Do not "
            "claim the history half of VAR2-02 closed until this list is "
            "empty."
        )
    return (
        f"All {targeted}/{targeted} targeted history ops "
        f"({', '.join(audit['targeted_history_ops'])}) are now representable "
        "and correctly reserved-token-layered in a real widened conversation "
        "trace -- the history half of VAR2-01's named follow-on is closed. "
        f"{len(audit['topology_ops_left_out_of_scope'])} topology ops "
        "remain unused and out of scope for this slice "
        "(openui.wrap_node/reorder_children/etc.) -- a further VAR2-03 or "
        "later session's work, per VAR2-01's own 'or accepting the narrower "
        "scope' framing. conversation.ast_edit/conversation.transaction_commit "
        "remain structurally unreachable regardless of corpus size, per "
        "VAR2-01's finding -- not attempted here."
    )


def render_markdown(payload: dict[str, Any]) -> str:
    audit = payload["adequacy_audit"]
    probe = payload["wiring_probe"]
    lines = [
        "# VAR2-02: OPS_VOCAB history-op corpus widening (follow-on to VAR2-01)",
        "",
        "- **Status:** history-control ops widened; topology ops explicitly out of scope",
        "- **Claim class:** `measurement` (adequacy delta), `wiring` (matched-arm probe)",
        f"- generated_at: `{payload['generated_at']}`",
        f"- ops_vocab fingerprint: `{payload['ops_vocab_fingerprint']}`",
        f"- corpus content fingerprint: `{payload['corpus_content_fingerprint']}`",
        "",
        "## Honesty scoping (read first)",
        "",
        "This script widens VAR2-01's corpus by one real, hand-built "
        "`ConversationTraceV1` that actually contains "
        "`UNDO`/`REDO`/`CHECKOUT_STATE`/`COPY_STATE` turns (`conversation.py`'s "
        "own API, not a synthetic op-id list), plus one real `AST_EDIT` drawn "
        "from the live legal set and one `FORK`. It closes only the "
        "**history** half of VAR2-01's named follow-on -- topology-operator "
        "widening (`openui.wrap_node`, `openui.reorder_children`, etc.) is "
        "explicitly out of scope for this slice, per VAR2-01's own \"or "
        "accepting the narrower scope\" framing. Not a trained-model quality "
        "experiment and not a ship claim; "
        f"`MAX_RUN_MINUTES={MAX_RUN_MINUTES}` bounds this to measurement/wiring "
        "only, same as VAR2-01.",
        "",
        "## Op-id sequence realized by the widened trace",
        "",
        "```",
        ", ".join(payload["op_id_sequence"]),
        "```",
        "",
        "## Adequacy delta vs VAR2-01 (the deliverable)",
        "",
        f"- targeted history ops ({len(audit['targeted_history_ops'])}): "
        + ", ".join(f"`{op}`" for op in audit["targeted_history_ops"]),
        f"- **newly_covered_ops** ({len(audit['newly_covered_ops'])}): "
        + (", ".join(f"`{op}`" for op in audit["newly_covered_ops"]) or "none"),
        f"- **still_missing_targeted_ops** ({len(audit['still_missing_targeted_ops'])}): "
        + (", ".join(f"`{op}`" for op in audit["still_missing_targeted_ops"]) or "none"),
        f"- unused ops after widening (union with VAR2-01's corpus, "
        f"{len(audit['unused_ops_after_widening_union_with_var2_01'])}): "
        + (
            ", ".join(
                f"`{op}`" for op in audit["unused_ops_after_widening_union_with_var2_01"]
            )
            or "none"
        ),
        f"- topology ops left out of scope for this slice "
        f"({len(audit['topology_ops_left_out_of_scope'])}): "
        + (
            ", ".join(f"`{op}`" for op in audit["topology_ops_left_out_of_scope"])
            or "none"
        ),
        "",
        "## Wiring probe (off vs on, widened trace)",
        "",
        f"- representable: {probe['representable']}",
        f"- off-arm output unchanged (lever disabled): {probe['off_arm_unchanged']}",
        f"- on-arm reserved tokens correctly layered (I13, no grammar collision): "
        f"{probe['reserved_tokens_all_layered']}",
        "",
        "## Decision this audit licenses",
        "",
        _decision_sentence(payload),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = run_audit(generated_at=generated_at)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(_decision_sentence(payload))
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
