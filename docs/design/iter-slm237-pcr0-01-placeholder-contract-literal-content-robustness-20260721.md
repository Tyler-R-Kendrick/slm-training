# SLM-237 (PCR0-01): placeholder-contract literal-content robustness probe (slm237-placeholder-contract-literal-content-robustness-20260721)

**Matrix set:** `slm237_placeholder_contract_literal_content_robustness`
**Version:** `pcr0-01-v1`
**Status:** fixture
**Claim class:** wiring
**Gate hash:** `d145e2f5308fdb3f...`
**Disposition:** gap_confirmed — Across all 3 structural shapes: content-only body wording edits left all three contract metrics at 1.0 (robust); an unrelated placeholder-shaped body mention lowered _contract_precision below 1.0 while _contract_recall and _placeholder_fidelity stayed at 1.0 (asymmetric precision contamination); and a genuine contract violation (real placeholder omitted) scored a perfect 1.0 on all three metrics whenever the body happened to mention the placeholder token in prose, versus a correctly-penalized 0.0 on an otherwise-identical violation with no mention -- confirming extract_placeholders is not literal-content-invariant and can award full false credit for an unfilled contract slot.

## Hypothesis

extract_placeholders (dsl/placeholders.py) matches its placeholder regex against the raw source string with no awareness of string-literal boundaries, so a colon-prefixed dotted token inside an ordinary text-content literal is indistinguishable from a real ``:placeholder`` slot reference. This has two consequences shared by eval_runner's _contract_precision / _contract_recall / _placeholder_fidelity and openui_rl's score_openui: (1) an unrelated placeholder-shaped mention in a free literal lowers _contract_precision even when the real contract is fully and correctly filled, while _contract_recall / _placeholder_fidelity stay at 1.0 because they only test the gold set; and (2) when a prediction *omits* its one real contract placeholder (a genuine violation) but a free literal happens to mention that exact placeholder token in prose, _contract_precision / _contract_recall / _placeholder_fidelity and score_openui's placeholder_fidelity term all score it as if the contract were perfectly satisfied -- identical to a correct prediction and strictly better than an otherwise-identical violation that does not happen to mention the token.

## Falsifier

Either: an unrelated placeholder-shaped literal mention leaves _contract_precision at 1.0 when the real contract is otherwise fully and correctly filled; or a prediction that omits its one real contract placeholder scores below 1.0 on _contract_precision / _contract_recall / _placeholder_fidelity / score_openui's placeholder_fidelity term even when a free literal mentions the exact placeholder token in prose (i.e. the omission is always correctly penalized regardless of incidental text).

## Honest caveats

- Fixture/wiring evidence only: no checkpoint, GPU run, RL training step, or ship-gate claim is made or implied.
- All documents are synthetic but real, grammar-valid OpenUI (each is round-tripped through the real slm_training.dsl.parser.validate before being scored) -- this is not a reimplementation of extract_placeholders, _contract_precision, _contract_recall, _placeholder_fidelity, or score_openui, only real calls to them.
- The variant set (5 literal-content edits x 3 structural shapes) is small and hand-authored; it demonstrates the mechanism exists and is reproducible, not its exact prevalence across a real corpus of generated predictions.
- The downstream reward probe uses a single-placeholder slot_inventory. Composite-reward divergence between the mentioned/unmentioned contract-violation rows is NOT isolated to the placeholder_fidelity term alone: grammar_score (harnesses/preference/__init__.py) also calls extract_placeholders directly and returns 0.0 (failing the 'parse' term) whenever the whole serialized document has no placeholder-shaped token anywhere, so the reward-probe rows show both the parse term and the placeholder_fidelity term flipping together -- a second, independent instance of the same raw-text-scan mechanism, not a clean single-term isolation. It does not characterize a full RL training run.
- This harness does not change extract_placeholders, _contract_precision, _contract_recall, _placeholder_fidelity, score_openui, or any eval/RL default. It documents a concrete scoring-mechanics gap as a candidate for a future, separately reviewed hardening change (e.g. scoring only parsed AST placeholder-slot nodes instead of raw text) -- never implemented here.

## Per-row results (contract metrics)

| shape | variant | category | violated | pred placeholders | precision | recall | fidelity | false credit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_leaf_card | neutral | baseline | False | :hero.title | 1.0000 | 1.0000 | 1.0000 | False |
| single_leaf_card | plain_alt_wording | content_only | False | :hero.title | 1.0000 | 1.0000 | 1.0000 | False |
| single_leaf_card | spurious_unrelated | spurious_unrelated | False | :hero.title, :support.email | 0.5000 | 1.0000 | 1.0000 | False |
| single_leaf_card | violation_mentioned | contract_violation_mentioned | True | :hero.title | 1.0000 | 1.0000 | 1.0000 | True |
| single_leaf_card | violation_unmentioned | contract_violation_unmentioned | True | — | 0.0000 | 0.0000 | 0.0000 | False |
| nested_two_leaf_card | neutral | baseline | False | :hero.title | 1.0000 | 1.0000 | 1.0000 | False |
| nested_two_leaf_card | plain_alt_wording | content_only | False | :hero.title | 1.0000 | 1.0000 | 1.0000 | False |
| nested_two_leaf_card | spurious_unrelated | spurious_unrelated | False | :hero.title, :support.email | 0.5000 | 1.0000 | 1.0000 | False |
| nested_two_leaf_card | violation_mentioned | contract_violation_mentioned | True | :hero.title | 1.0000 | 1.0000 | 1.0000 | True |
| nested_two_leaf_card | violation_unmentioned | contract_violation_unmentioned | True | — | 0.0000 | 0.0000 | 0.0000 | False |
| button_row | neutral | baseline | False | :hero.title | 1.0000 | 1.0000 | 1.0000 | False |
| button_row | plain_alt_wording | content_only | False | :hero.title | 1.0000 | 1.0000 | 1.0000 | False |
| button_row | spurious_unrelated | spurious_unrelated | False | :hero.title, :support.email | 0.5000 | 1.0000 | 1.0000 | False |
| button_row | violation_mentioned | contract_violation_mentioned | True | :hero.title | 1.0000 | 1.0000 | 1.0000 | True |
| button_row | violation_unmentioned | contract_violation_unmentioned | True | — | 0.0000 | 0.0000 | 0.0000 | False |

## Downstream RL reward probe (`score_openui`, shape=`single_leaf_card`)

| variant | category | violated | composite | placeholder_fidelity | structural_similarity | parse |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | baseline | False | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| plain_alt_wording | content_only | False | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| spurious_unrelated | spurious_unrelated | False | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| violation_mentioned | contract_violation_mentioned | True | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| violation_unmentioned | contract_violation_unmentioned | True | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `extract_placeholders`, `_contract_precision`, `_contract_recall`, `_placeholder_fidelity`, `score_openui`, or any eval/RL default, does not train a model, and makes no ship or gate claim. It documents a concrete scoring-mechanics gap in the literal-content robustness of the placeholder-contract metric family shared by the eval scoreboard and the RL reward contract, as a candidate for a future, separately reviewed hardening change (never implemented here).

## Reproducibility

```bash
python -m scripts.run_slm237_placeholder_contract_literal_content_robustness --mode plan-only
python -m scripts.run_slm237_placeholder_contract_literal_content_robustness --mode fixture
```
