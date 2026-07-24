# Tokenizer–grammar invariants

`python -m scripts.verify_tokenizer_grammar_invariants` is the CI certificate
for the fixed OpenUI output tokenizers. It checks the immutable
`e763_symbol_only_eval_r2_20260722` manifest (19 programs across smoke,
held-out, adversarial, OOD, and RICO-held suites). This is a deterministic
codec check, not a model evaluation or a ship-gate claim.

The certificate fails closed when either native or choice tokenizer has a
duplicate/non-contiguous ID, inverse-map disagreement, NFC-normalized token
collision, moved control-token boundary, unknown-token leakage, or a changed
round-trip ID sequence. It also compares grammar-legal actions at every
reachable prefix before and after each round-trip.

## Compatibility registry

`src/slm_training/resources/tokenizer_layout_registry.json` pins the version,
vocabulary size, SHA-256 ID layout, and control IDs for the `dsl_native` and
`choice_codec` checkpoint formats. Intentional layout changes require a
tokenizer-version bump and a documented migration before updating this file.

Codec-local embedding IDs intentionally remain zero-based and overlap across
the legacy, native, and choice codecs; existing checkpoints rely on that.
`slm_training.dsl.openui_tokens` therefore provides versioned *logical*
namespaces (`control`, `openui`, `abstract_plan`, `model_native`) for any
cross-codec consumer rather than renumbering stored embedding rows.
