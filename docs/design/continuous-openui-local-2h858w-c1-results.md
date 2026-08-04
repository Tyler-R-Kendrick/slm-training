# Continuous autotrain: 2026-08-04 (session 2h858w) cycle 1 — exact tie, non-positive

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `5ba8e430` (`main` tip at cycle start; branch
`claude/great-dirac-2h858w`, scheduled autotrain routine)

**Verdict:** `bounds` again ties its size-matched `control` exactly on every
smoke quality metric, reproducing the same non-positive result multiple
independent same-loop sessions observed on 2026-08-03. The only difference is
p50 latency, which alone does not qualify as a metric win. Fixture screening
only — not a ship or promotion claim.

> Note on campaign-id collisions: `outputs/autoresearch/` is gitignored and
> local to each ephemeral session container, so the driver's deterministic
> `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1` id is this
> session's own independent cycle-1 measurement; the `2h858w` branch suffix
> disambiguates the filename from same-day runs by other sessions.

| Arm | Params | Seed | structural_similarity | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 100001 | .0575 | .63333 | 3758.99 |
| bounds | 1,608,962 | 100001 | .0575 | .63333 | 3610.88 |

Both arms parse all 3 smoke documents, use the same 1,608,962 trainable
parameters, and tie exactly on every quality metric:
`meaningful_program_rate`, `ast_beq_rate`, `canonical_beq_rate`, and
`reward_score` are all **0 on both arms**. `bounds` decodes ~3.9% faster
(3610.88ms vs 3758.99ms p50), but this is a pure latency delta with no
accompanying quality or held/mpr signal, so per the SDLC quality-aware
tradeoff rule (`_classify_metric_tradeoff`) it is **not** a positive result
on its own.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), every quality
threshold (`meaningful_program_rate`, `structural_similarity`,
`component_type_recall`, `ast_beq_rate`, `canonical_beq_rate`,
`reward_score`), and `held_out`/`adversarial`/`ood`/`rico_held` were not run.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n` + null primary-metric delta,
`smoke.structural_similarity` control=candidate=`.0575`, improvement=`0`).
Per `sdlc` autotrain-iteration-delivery, no stacked PR layer is opened for
this cycle — local commit and docs only.

## Next priorities (ranked by the driver)

1. Test the distinct size-matched `component-plan` quality hypothesis next
   instead of re-running the now-exhausted `bounds` arm (confidence 0.90).
2. Keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).
3. Rotate thrash recommendation across the lever bank rather than
   bounds-only (confidence 0.65).
4. Soft ship-gate fails on fixture `n` never stop the continuous loop
   (confidence 0.80).

Machine evidence:
[`continuous-openui-local-2h858w-c1-results.json`](continuous-openui-local-2h858w-c1-results.json).
