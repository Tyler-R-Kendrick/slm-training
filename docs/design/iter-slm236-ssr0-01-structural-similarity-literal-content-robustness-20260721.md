# SLM-236 (SSR0-01): structural-similarity literal-content robustness probe (slm236-structural-similarity-literal-content-robustness-20260721)

**Matrix set:** `slm236_structural_similarity_literal_content_robustness`
**Version:** `ssr0-01-v1`
**Status:** fixture
**Claim class:** wiring
**Gate hash:** `c0686b131952f987...`
**Disposition:** gap_confirmed — Across all 3 structural shapes, every content-only wording edit (3/3 rows) left structural_similarity at exactly 1.0, while every literal edit that added ordinary punctuation or a component-shaped substring (15/15 rows) lowered the score below 1.0 despite identical DSL structure -- confirming structural_similarity is not literal-content-invariant, and that this propagates into score_openui's composite RL reward term with parse validity and placeholder fidelity held constant.

## Hypothesis

structural_similarity (eval_runner.py) computes its component-multiset and bracket/paren "depth" proxies over the raw source string, including the contents of ordinary text-content string literals (only strip_style_literals is applied first, which strips style-prop literals, not text-content literals). Because of this, two OpenUI documents with byte-identical DSL structure that differ only in the literal text of one leaf string argument can receive different structural_similarity scores: ordinary punctuation (parentheses, square brackets) in the literal text perturbs the depth proxy, and a literal substring shaped like "Word(" is misread by the component regex as an extra component occurrence, both independent of any real structural difference. This propagates into the RL reward contract (score_openui's composite term) without a corresponding change in parse validity or placeholder fidelity.

## Falsifier

For every same-shape, content-only-edited pair (identical DSL structure, one leaf literal changed), structural_similarity returns 1.0 regardless of the literal text's punctuation or wording; or the component regex/depth proxies are shown not to read literal-string content in the real implementation.

## Honest caveats

- Fixture/wiring evidence only: no checkpoint, GPU run, RL training step, or ship-gate claim is made or implied.
- All documents are synthetic but real, grammar-valid OpenUI (each is round-tripped through the real slm_training.dsl.parser.validate before being scored) -- this is not a reimplementation of structural_similarity, _component_multiset, or score_openui, only real calls to them.
- The variant set (7 literal-content edits x 3 structural shapes) is small and hand-authored; it demonstrates the mechanism exists and is reproducible, not its exact prevalence across a real corpus of generated predictions.
- The downstream reward probe uses an empty slot_inventory so the placeholder_fidelity term of score_openui stays constant across variants -- this isolates the structural_similarity term's contribution to composite reward, it does not characterize a full RL training run.
- This harness does not change structural_similarity, _component_multiset, score_openui, or any eval/RL default. It documents a concrete scoring-mechanics gap as a candidate for a future, separately reviewed hardening change (e.g. scoring only parsed AST component/call nodes instead of raw text) -- never implemented here.

## Per-row results (structural_similarity)

| shape | variant | category | score | divergent | spurious components |
| --- | --- | --- | --- | --- | --- |
| single_leaf_card | neutral | baseline | 1.0000 | False | — |
| single_leaf_card | plain_alt_wording | content_only | 1.0000 | False | — |
| single_leaf_card | parens_benign | benign_punctuation | 0.9500 | True | — |
| single_leaf_card | brackets_benign | benign_punctuation | 0.9500 | True | — |
| single_leaf_card | fake_component_single | adversarial_regex | 0.8100 | True | Details |
| single_leaf_card | fake_component_multi | adversarial_regex | 0.6667 | True | Email, Support |
| single_leaf_card | mixed_adversarial | adversarial_mixed | 0.7100 | True | Policy |
| nested_two_leaf_card | neutral | baseline | 1.0000 | False | — |
| nested_two_leaf_card | plain_alt_wording | content_only | 1.0000 | False | — |
| nested_two_leaf_card | parens_benign | benign_punctuation | 0.9500 | True | — |
| nested_two_leaf_card | brackets_benign | benign_punctuation | 0.9500 | True | — |
| nested_two_leaf_card | fake_component_single | adversarial_regex | 0.8100 | True | Details |
| nested_two_leaf_card | fake_component_multi | adversarial_regex | 0.6667 | True | Email, Support |
| nested_two_leaf_card | mixed_adversarial | adversarial_mixed | 0.7100 | True | Policy |
| button_row | neutral | baseline | 1.0000 | False | — |
| button_row | plain_alt_wording | content_only | 1.0000 | False | — |
| button_row | parens_benign | benign_punctuation | 0.9000 | True | — |
| button_row | brackets_benign | benign_punctuation | 0.9000 | True | — |
| button_row | fake_component_single | adversarial_regex | 0.6667 | True | Details |
| button_row | fake_component_multi | adversarial_regex | 0.4500 | True | Email, Support |
| button_row | mixed_adversarial | adversarial_mixed | 0.4667 | True | Policy |

## Downstream RL reward probe (`score_openui`, shape=`single_leaf_card`)

| variant | category | composite | structural_similarity | parse | placeholder_fidelity |
| --- | --- | --- | --- | --- | --- |
| neutral | baseline | 0.8500 | 1.0000 | 1.0000 | 0.5000 |
| plain_alt_wording | content_only | 0.8500 | 1.0000 | 1.0000 | 0.5000 |
| parens_benign | benign_punctuation | 0.8375 | 0.9500 | 1.0000 | 0.5000 |
| brackets_benign | benign_punctuation | 0.8375 | 0.9500 | 1.0000 | 0.5000 |
| fake_component_single | adversarial_regex | 0.8025 | 0.8100 | 1.0000 | 0.5000 |
| fake_component_multi | adversarial_regex | 0.7667 | 0.6667 | 1.0000 | 0.5000 |
| mixed_adversarial | adversarial_mixed | 0.7775 | 0.7100 | 1.0000 | 0.5000 |

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `structural_similarity`, `_component_multiset`, `score_openui`, or any eval/RL default, does not train a model, and makes no ship or gate claim. It documents a concrete scoring-mechanics gap in the literal-content robustness of a structural metric shared by the eval scoreboard, RL reward contract, and preference counterfactual re-scoring, as a candidate for a future, separately reviewed hardening change (never implemented here).

## Reproducibility

```bash
python -m scripts.run_slm236_structural_similarity_literal_content_robustness --mode plan-only
python -m scripts.run_slm236_structural_similarity_literal_content_robustness --mode fixture
```
