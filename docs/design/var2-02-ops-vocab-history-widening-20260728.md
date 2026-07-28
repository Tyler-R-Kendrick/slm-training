# VAR2-02: OPS_VOCAB history-op corpus widening (follow-on to VAR2-01)

- **Status:** history-control ops widened; topology ops explicitly out of scope
- **Claim class:** `measurement` (adequacy delta), `wiring` (matched-arm probe)
- generated_at: `2026-07-28T09:46:48Z`
- ops_vocab fingerprint: `e039ebe5ab2ea6d746bd0757d16f65e2768fb4c0b8f4f37aa42c3e32472ecb93`
- corpus content fingerprint: `0e3308f39abf7f4c8000b120ef8bcbaef755bc66f88aae8d0e925aec1c9e0ad9`

## Honesty scoping (read first)

This script widens VAR2-01's corpus by one real, hand-built `ConversationTraceV1` that actually contains `UNDO`/`REDO`/`CHECKOUT_STATE`/`COPY_STATE` turns (`conversation.py`'s own API, not a synthetic op-id list), plus one real `AST_EDIT` drawn from the live legal set and one `FORK`. It closes only the **history** half of VAR2-01's named follow-on -- topology-operator widening (`openui.wrap_node`, `openui.reorder_children`, etc.) is explicitly out of scope for this slice, per VAR2-01's own "or accepting the narrower scope" framing. Not a trained-model quality experiment and not a ship claim; `MAX_RUN_MINUTES=3` bounds this to measurement/wiring only, same as VAR2-01.

## Op-id sequence realized by the widened trace

```
conversation.copy_state, openui.remove_node, conversation.undo, conversation.redo, conversation.checkout_state, conversation.fork
```

## Adequacy delta vs VAR2-01 (the deliverable)

- targeted history ops (4): `conversation.checkout_state`, `conversation.copy_state`, `conversation.redo`, `conversation.undo`
- **newly_covered_ops** (4): `conversation.checkout_state`, `conversation.copy_state`, `conversation.redo`, `conversation.undo`
- **still_missing_targeted_ops** (0): none
- unused ops after widening (union with VAR2-01's corpus, 12): `conversation.ast_edit`, `conversation.transaction_commit`, `openui.contract_subtree`, `openui.duplicate_subtree`, `openui.expand_template`, `openui.move_node`, `openui.reorder_children`, `openui.reparent_node`, `openui.set_property`, `openui.unset_property`, `openui.unwrap_node`, `openui.wrap_node`
- topology ops left out of scope for this slice (10): `openui.contract_subtree`, `openui.duplicate_subtree`, `openui.expand_template`, `openui.move_node`, `openui.reorder_children`, `openui.reparent_node`, `openui.set_property`, `openui.unset_property`, `openui.unwrap_node`, `openui.wrap_node`

## Wiring probe (off vs on, widened trace)

- representable: True
- off-arm output unchanged (lever disabled): True
- on-arm reserved tokens correctly layered (I13, no grammar collision): True

## Decision this audit licenses

All 4/4 targeted history ops (conversation.checkout_state, conversation.copy_state, conversation.redo, conversation.undo) are now representable and correctly reserved-token-layered in a real widened conversation trace -- the history half of VAR2-01's named follow-on is closed. 10 topology ops remain unused and out of scope for this slice (openui.wrap_node/reorder_children/etc.) -- a further VAR2-03 or later session's work, per VAR2-01's own 'or accepting the narrower scope' framing. conversation.ast_edit/conversation.transaction_commit remain structurally unreachable regardless of corpus size, per VAR2-01's finding -- not attempted here.
