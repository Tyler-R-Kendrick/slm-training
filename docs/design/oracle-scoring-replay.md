# SLM-260 / VSD0-01 — Oracle scoring replay

A fixture-only harness that feeds exact-gold outputs and controlled semantic
variants into the same production scoring path used by
``evaluate_model``, quality matrices, ship gates, and AgentV.

## Purpose

Production eval code lives in
``src/slm_training/harnesses/model_build/eval_runner`` and
``src/slm_training/evals/meaningful_program``.  When we change a metric,
gate threshold, or decode policy, we need a fast, deterministic way to confirm
that the judge still agrees with human-authored expectations on small oracle
inputs.  This harness provides those oracle inputs and records the result.

It is **not** a model-evaluation or ship claim.  It is a regression guard for
the scoring machinery itself.

## Variant taxonomy

Every fixture record (card, slider, switch, tabs, callout, image_block)
emits one row per variant kind in ``VARIANT_KINDS``:

| Kind | Expected | What it tests |
| --- | --- | --- |
| ``exact_gold`` | True | Prediction equals the gold OpenUI. |
| ``canonical_roundtrip`` | True | Prediction is ``validate(gold).serialized``. |
| ``alpha_renamed_equivalent`` | True | Non-root assignment identifiers are renamed consistently; semantics unchanged. |
| ``egraph_equivalent`` | True | Independent non-root top-level statements are reversed; semantics unchanged. |
| ``unbound_reference`` | False | Introduces a reference to an undefined identifier; binding correctness must reject it. |
| ``wrong_component_or_property_role`` | False | Replaces the requested component with another parser-valid component; prompt component missing. |
| ``wrong_placeholder_identity`` | False | Replaces an inventory placeholder with an out-of-inventory slot; unexpected identity. |
| ``prompt_contract_omission`` | False | Omits one required placeholder usage while remaining syntax/schema valid. |
| ``prompt_incompatible_but_valid`` | False | Swaps the requested component type for another parser-valid type. |
| ``duplicate_or_filler_gaming`` | False | Adds duplicate subtrees or spammed placeholders; anti-gaming must reject. |
| ``unreachable_or_dead_content`` | False | Adds an unused assignment referencing an existing slot; unreachable binding. |

## Production scoring reuse

``score_prediction`` calls exactly the helpers used by
``eval_runner._score_one`` for document records:

- ``meaningful_program_v1``
- ``binding_aware_meaningful_v2``
- ``_raw_syntax_valid``
- ``_placeholder_fidelity`` / ``_placeholder_fidelity_normalized`` / ``_placeholder_validity``
- ``_tree_match``
- ``structural_similarity``
- ``component_type_recall``
- ``_contract_precision`` / ``_contract_recall``
- ``_reward_for_prediction``

The returned detail dict contains the same keys as the production ``details``
entry (plugin-only topology evidence fields are excluded because no plugin is
involved).

## How to run

```bash
# Built-in fixture records
python -m scripts.audit_gold_scoring --output outputs/oracle_scoring_replay.json

# With a custom JSONL record set
python -m scripts.audit_gold_scoring \
  --records outputs/data/adversarial/my_records.jsonl \
  --suite my_suite \
  --output outputs/my_suite_replay.json \
  --run-dir outputs/runs/my_suite_replay
```

The script writes the full replay manifest JSON and prints a small status blob.
Runtime is a few seconds; no torch or GPU is required.

## Honest caveats

- **Fixture-only evidence.** These records are hand-authored canaries, not a
  representative eval sample.  They cannot support a ship readiness claim.
- **No model, no decode policy.** The manifest does not exercise grammar
  constraints, slot-contract constrained decode, or model-specific topology
  telemetry; it only checks the scoring layer.

## Latest fixture run

A trimmed manifest from the built-in fixture records lives at
``docs/design/iter-slm260-oracle-scoring-replay-20260721.json``.
