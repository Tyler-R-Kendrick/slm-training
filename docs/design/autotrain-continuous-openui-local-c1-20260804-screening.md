# Autotrain continuous-openui-local c1 (2026-08-04): bounds-lever screening

**Verdict:** fixture screening only, ship gates rejected. Not a positive
result — no stacked PR for this cycle.

Fresh continuous loop `continuous-openui-local`, cycle 1, started clean from
merged `origin/main` `eba6db3` (no divergence: upstream and integration
commits match). Ran the driver's default matched pair — `control` (both
grammar levers off) vs `bounds` (bounds lever) — against the published
`wf_smoke_v2` fixture train set and the `smoke` eval suite only (no
`held_out` / `adversarial` / `ood` / `rico_held` suites configured):

```bash
python -m scripts.run_autotrain_continuous \
  --loop-id continuous-openui-local --supervised --max-cycles 1 \
  --train-version wf_smoke_v2 --steps 20
```

Both arms trained the same 1,608,960-parameter `twotower` model (size-matched,
`EG_params` unchanged) and produced structurally identical decode quality:

| Arm | latency p50 (ms) | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 |
| --- | --- | --- | --- | --- | --- |
| control | 3299.53 | 1.0 | 0.0 | 0.0575 | 0.6333 |
| bounds (candidate) | 3854.54 | 1.0 | 0.0 | 0.0575 | 0.6333 |

`smoke.structural_similarity` (primary metric) improvement = 0.0 — the bounds
lever produced no measurable change on this fixture pass, so this is a
matched null, not a win.

Honest ship gates (`AgentEvals`) reject both arms on the same 11 failures —
evidence-volume (`insufficient_n actual=3 need>=20`, and the `held_out` /
`adversarial` / `ood` / `rico_held` suites are simply not wired for this
cycle) plus quality thresholds (`meaningful_program_rate`,
`structural_similarity`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, `reward_score` all below their published bars). This is
the expected fixture-scale diagnostic outcome, not a claim of readiness —
see `honest-ship-eval`.

SDLC Phase A classification: **NON_POSITIVE**
(`fixture_insufficient_n_alone`, `primary_metric_null_or_worse`). Per
`sdlc` autotrain-iteration-delivery, non-positive cycles stay local-commit +
docs only — **no stacked PR is opened for this cycle**.

Checkpoints were written for both arms
(`runs/c20260804-continuous-openui-local-8c0b60dd-c1-{control,bounds}/checkpoints/last.pt`)
but are fixture/scratch-scale screening artifacts, not promotion candidates —
no `MODEL_CARD.md` roster entry is warranted (no climb/ship claim attaches to
either checkpoint).

Next-run priority from the driver: the bounds-lever arm is now exhausted for
this campaign lineage; the ranked successor is the size-matched
`component-plan` quality hypothesis
(`c20260804-continuous-openui-local-8c0b60dd-c1-component-plan`), keeping the
matched control fixed as baseline every cycle.

Machine evidence:
[`autotrain-continuous-openui-local-c1-20260804-screening.json`](autotrain-continuous-openui-local-c1-20260804-screening.json).
