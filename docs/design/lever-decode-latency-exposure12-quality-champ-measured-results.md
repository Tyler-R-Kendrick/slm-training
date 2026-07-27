# Decode-latency recovery on exposure12 quality champ — NOT SHIP

**Honesty:** `fixture_or_scratch` / smoke `n=3`. **Not a ship claim.**

## Hypothesis

On the frozen quality checkpoint (`lever_exposure12_v1` · seed47 · sb=1.5 ·
s16/lr1e-3/bs2), eval-only decode knobs reduce latency **without** dropping
`meaningful_program_rate` below **0.67**.

## Baseline diagnosis

| example | outcome | meaningful | latency_ms |
| --- | --- | --- | ---: |
| smoke_hero_01 | model_valid | True | 277005.45 |
| smoke_button_01 | fallback_output | False | 30011.11 |
| smoke_callout_01 | fallback_output | True | 30009.16 |

Hero alone is **~277s** (`model_valid`); two others ~30s with `fallback_output`.
p50 ~30s understates the wall (sum ≈ 337s).

## Arms (same checkpoint, ASAP default unless noted)

| arm | flags | parse | meaningful | reward | empty | p50 | p95 | max | sum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_asap_t30_default | `defaults (ltr_max=256, attempts=3, gen=8, asap)` | 1.0 | 0.6666666666666666 | 0.8523333333333333 | 0 | 30011.11 | 277005.45 | 277005 | 337026 |
| ltr_max_128 | `--grammar-ltr-max-tokens 128` | 0.6666666666666666 | 0.3333333333333333 | 0.6326666666666666 | 1 | 75709.54 | 131383.81 | 131384 | 237112 |
| ltr_max_64 | `--grammar-ltr-max-tokens 64` | 1.0 | 0.6666666666666666 | 0.8543333333333333 | 0 | 30078.89 | 89173.25 | 89173 | 149255 |
| max_attempts_1 | `--max-attempts 1` | 1.0 | 0.3333333333333333 | 0.7693333333333333 | 0 | 30010.45 | 144384.72 | 144385 | 204401 |
| gen_steps_4 | `--gen-steps 4` | 1.0 | 0.6666666666666666 | 0.915 | 0 | 187622.22 | 236281.09 | 236281 | 453909 |
| fixed_ltr | `--constraint-debt-routing-mode fixed_ltr` | 0.6666666666666666 | 0.3333333333333333 | 0.5336666666666666 | 1 | 30005.53 | 215369.01 | 215369 | 275380 |
| combo_ltr128_att1 | `--grammar-ltr-max-tokens 128 --max-attempts 1` | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 30005.76 | 30006.94 | 30007 | 90015 |

## Decision

**ACCEPT** — ltr_max_64 holds quality (parse=1.0, meanful=0.6666666666666666) and cuts latency max 277005→89173ms p50 30011.11→30078.89 sum 337026→149255

Update quality-serving decode recipe with the accepted arm; keep constrained decode.

## Next lever (evidence-backed)

Multi-seed confirm accepted decode recipe on exposure12; then non-fixture data.

Captured: 2026-07-27T17:11:12.535782+00:00

