# AbstractPlanV1 — reserved abstract-plan token codebook (AP-016 / SLM-302)

`slm_training.dsl.abstract_plan.AbstractPlanV1` is a versioned, collision-free
contract that reserves `<beginabstract>`/`<endabstract>` delimiters plus a
fixed-size pool of abstract codebook tokens (`<ABS_0>` … `<ABS_{M-1}>`)
inside the pre-allocated `abstract_plan` logical namespace
(`slm_training.dsl.openui_tokens.TOKEN_ID_NAMESPACE_RANGES["abstract_plan"]`).

Defaults: `M` (`slot_count`) = 64, `m_max` (`max_slot_count`) = 128, `T`
(`rounds`) = 3 — all configurable per instance, bounded by the reserved
namespace capacity.

This is serialization/config only. No production behavior changes:

* `DSLNativeTokenizer.build(..., abstract_plan_slots=0)` (default) and
  `ChoiceTokenizer.build(..., abstract_plan_slots=0)` (default) emit the
  exact same vocabulary, ids, and `vocab_size` as before this change —
  verified by `tests/test_dsl/test_abstract_plan.py::test_feature_off_parity_*`.
* Passing `abstract_plan_slots > 0` appends the delimiter pair and slot rows
  **after** every existing token (append-only), so no prior token id moves.
  This is an opt-in path for a later, separately versioned experiment; it is
  not wired into any default config or generation path.
* Base-variant slots carry no assigned meaning (`AbstractPlanV1.role_metadata`
  is empty, `is_interpretable` is `False`) — token identity alone must not
  leak semantics ahead of a causal-use experiment.

## Collision freedom

Reserved token ids are logical ids inside the `abstract_plan` namespace
(`0x4000`–`0x7000`), disjoint from the `control`, `openui`, and
`model_native` namespaces (see `docs/design/tokenizer-grammar-invariants.md`
and `tests/test_dsl/test_tokenizer_grammar_invariants.py::test_logical_token_namespaces_are_disjoint`).
`AbstractPlanV1.assert_no_collisions(existing_tokens)` fails closed if any
reserved token text already exists in a vocabulary.

## Checkpoint / embedding migration

`slm_training.dsl.abstract_plan.resize_embedding_preserving_rows` grows an
`nn.Embedding` table by initializing only the new rows (matched to the
existing table's mean/std); `verify_embedding_resize_preserved_old_rows`
fails closed unless every pre-existing row is bit-for-bit unchanged after a
resize. Neither `DSL_TOKENIZER_VERSION` nor `CHOICE_TOKENIZER_VERSION` is
bumped by this change, since the default (disabled) vocabulary layout is
unchanged; a future change that enables the block by default must bump both
and publish an explicit checkpoint migration.

## Reproduction

```bash
pytest -q tests/test_dsl/test_abstract_plan.py tests/test_dsl/test_tokenizer_grammar_invariants.py
python -m scripts.verify_tokenizer_grammar_invariants
```
