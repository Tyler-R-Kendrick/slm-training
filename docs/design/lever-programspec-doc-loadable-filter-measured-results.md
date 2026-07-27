# Programspec-document loadable filter train lever — NOT SHIP

**Honesty:** `fixture_or_scratch` / smoke `n=3`. **Not a ship claim.**

## Hypothesis
Raw programspec builds fail `train_model` load contracts. Filtering to records that
pass **symbol-only v2 + role-safe + harness meta** unlocks non-fixture training
and may lift `meaningful_program_rate` vs the fixture quality champ
(`lever_exposure12_v1`) under the frozen micro-recipe.

## Workaround (not a gate weaken)
1. Build: `lever_programspec_doc_v1` (programspec, target_kinds=document, cap=12).
2. Filter offline with the same checks `load_train_records` + `assert_role_safe_output` use.
3. Result: **466/587** kept
   (rejected: {'placeholder_non_content': 114, 'non_canonical_harness': 7}).

## Recipe
Champion micro-recipe: s16 · lr=1e-3 · bs=2 · structural_bias=1.5 · seed=47 ·
ASAP · decode_timeout=30 · scratch/cpu · grammar constrained.

## Results

| arm | corpus_n | last_loss | parse | meaningful | reward | empty | p50 | max_lat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wf_smoke_v2 | 103 | 7.452 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 1469.53 | 1580.01 |
| exposure12 (quality champ) | 107 | 7.189 | 1.0 | 0.6666666666666666 | 0.8523333333333333 | 0 | 30011.11 | 277005.45 |
| **psdoc_loadable_v1** | 466 | 11.691 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 22781.1 | 23409.35 |

## Decision
**REJECT** — loadable programspec-doc does not beat exposure12 (meanful 0.6666666666666666→0.3333333333333333, parse 1.0→1.0, empty 0→0)

### Synthesis / harness note
Filter is an interim **workaround**. Durable fix: synthesizers emit only
train-loadable targets (canonical Harness serialization; no placeholders in
non-content props) so raw builds do not need post-hoc filtering.

## Next lever
If reject/partial: decode RNG determinism on quality champ, or improve synthesizers.
If accept: multi-seed confirm psdoc_loadable + optional ltr_max=64 latency re-tune.

Captured: 2026-07-27T17:37:40.051154+00:00
