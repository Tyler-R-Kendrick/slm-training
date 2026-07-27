# Exposure-cap lever: admit abstraction_ladder (cap 6→12) — NOT SHIP

**Honesty:** `fixture_or_scratch` / smoke suite `n=3`. **Not a ship claim.**

## Hypothesis

Strict profile default `max_records_per_parent=6` zeroed **abstraction_ladder** yield
(all L1–L5 rejected for `train_hero_01` under exposure). Raising the cap to **12**
admits the ladder (5 records) and may lift `meaningful_program_rate` under the
frozen champion micro-recipe (s16 · lr=1e-3 · bs=2 · sb=1.5 · seed=47 · ASAP · t30).

This is a **data exposure lever**, not a decontamination/quality-score gate change.

## Build evidence (synthesis-feedback)

| version | cap | n | abstraction_ladder admitted | yield |
| --- | ---: | ---: | ---: | ---: |
| lever_fixture_v1 | 6 (strict default) | 101 | 0 | 0.0 |
| lever_exposure12_v1 | 12 | 107 | 5 (L1–L5) | 1.0 |

Dedup/decontamination still active (fuzzy/semantic/ngram rejects retained).

## Train/eval results (champion recipe, smoke n=3)

| arm | n | last_loss | parse | meaningful | reward | empty | lat_p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wf_smoke_v2 (champ train) | 103 | 7.452 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 1469.53 |
| lever_fixture_v1 cap6 | 101 | 12.850 | 1.0 | 0.0 | 0.48100000000000004 | 0 | 30016.3 |
| lever_exposure12_v1 cap12 | 107 | 7.189 | 1.0 | 0.6666666666666666 | 0.8523333333333333 | 0 | 30011.11 |

## Decision

**ACCEPT** — exposure12 (ladder admitted) lifted meaningful without parse regression vs wf_smoke_v2

### Synthesis note
Admitting ladder via exposure is **successful as a data-build outcome** (yield 0→1).
Whether that improves model metrics is a separate scoreboard question — see decision.

### Recipe implication

Treat **`lever_exposure12_v1` + champion hparams** as the **quality micro-champion**
on smoke n=3 (meaningful **0.33→0.67**, reward **0.765→0.852**, parse stays 1.0).
Latency p50 ~30s is a **regression** vs the prior latency champ (~1.5s on `wf_smoke_v2`);
do not claim a free lunch — decode/latency levers may need re-tuning on this corpus.

## Next lever
If metrics still flat: non-fixture sources (`programspec` / `language_contract` / limited
RICO) that pass symbol-only contracts; or longer train wall with matched eval domain.

Captured: 2026-07-27T16:28:06.513589+00:00
