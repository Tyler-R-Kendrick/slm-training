# Fresh fixture train build vs wf_smoke_v2 — NOT SHIP

**Honesty:** `fixture_or_scratch` / smoke suite `n=3`. **Not a ship claim.**

## Hypothesis

A **fresh strict fixture rebuild** (`lever_fixture_v1`, current synthesizers + gates)
lifts `meaningful_program_rate` vs the older `wf_smoke_v2` train snapshot under the
champion micro-recipe (s16 · lr=1e-3 · bs=2 · sb=1.5 · seed=47 · ASAP · t30).

## Build (synthesis-feedback)

- source=fixture, profile=strict, version=`lever_fixture_v1`
- admitted **101**, rejected **19**
- warnings: eval n-gram overlap flagged 3 (gate held)
- family yields: human_curated≈0.95, layout_augment=1.0, prompt_paraphrase≈0.79,
  frontier_described=1.0, **abstraction_ladder=0.0** (exposure cap `max_records_per_parent`)
- No gate weakening.

## Results

| arm | records | last_loss | parse | meaningful | reward | empty | lat_p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wf_smoke_v2 | 103 | 7.452 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 1469.53 |
| lever_fixture_v1 | 101 | 12.850 | 1.0 | 0.0 | 0.48100000000000004 | 0 | 30016.3 |

## Decision

**REJECT** — fresh fixture rebuild does not lift meaningful/parse vs wf_smoke_v2 at micro recipe

Fresh fixture rebuild is nearly the same size/domain as `wf_smoke_v2` and does not
move the meaningful-program ceiling on smoke n=3. Next quality leverage is **non-fixture
sources** (RICO/programspec with contract-valid synthesizers) or harness work on
`abstraction_ladder` exposure (yield 0), not more micro hyperparameter sweeps.

Captured: 2026-07-27T16:18:48.030712+00:00
