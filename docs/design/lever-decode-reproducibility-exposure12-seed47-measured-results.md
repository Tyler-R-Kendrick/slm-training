# Decode reproducibility on exposure12 quality champ — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.** Measurement lever (determinism).

## Hypothesis
Re-evaluating the **same** checkpoint with **identical** flags yields identical
`meaningful_program_rate` / hero decode outcomes (decode is deterministic).

## Setup
- checkpoint: `exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt`
- flags: grammar_constrained · fixed_asap · decode_timeout=30 · eval_limit=3 · suite=smoke
- repeats: original eval + 3 fresh re-evals (no training)

## Results

| rep | parse | meaningful | empty | hero_outcome | hero_meaningful | hero_lat_ms | p50 | max |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| orig | 1.0 | 0.6666666666666666 | 0 | model_valid | True | 277005 | 30011.11 | 277005 |
| rep1 | 1.0 | 0.3333333333333333 | 0 | fallback_output | False | 30005 | 30003.65 | 30005 |
| rep2 | 0.6666666666666666 | 0.0 | 1 | fallback_output | False | 30008 | 30007.76 | 30008 |
| rep3 | 1.0 | 0.3333333333333333 | 0 | fallback_output | False | 30006 | 30005.24 | 30006 |

### Summary
- meaningful range: **0.0 – 0.6666666666666666** (unique values: 3)
- hero outcomes: ['model_valid', 'fallback_output', 'fallback_output', 'fallback_output']
- hero latency: 30005 – 277005 ms

## Decision
**NON DETERMINISTIC** — meanful 0.0–0.6666666666666666 across 4 re-evals; hero outcomes {'model_valid', 'fallback_output'}; hero lat 30005–277005ms

### Implication
Quality single-run claims (including prior meanful=0.67) need **multi-rep medians**.
Next lever: find/seed decode RNG (`evaluate_model` seed if available) or enforce
deterministic ranking; do not promote decode knobs on single-run noise.

Captured: 2026-07-27T17:42:53.873262+00:00
