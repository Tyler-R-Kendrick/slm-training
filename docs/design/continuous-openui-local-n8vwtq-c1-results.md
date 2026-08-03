# Continuous autotrain: 2026-08-03 cycle 1, session n8vwtq (non-positive)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Base commit:** `089b1649` (current `main` tip, already carrying merged
[#1369](https://github.com/Tyler-R-Kendrick/slm-training/pull/1369),
[#1376](https://github.com/Tyler-R-Kendrick/slm-training/pull/1376), and
[#1378](https://github.com/Tyler-R-Kendrick/slm-training/pull/1378))

| Arm | Loss | parse | MPR | structural_similarity | binder F1 | fidelity | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 22.6219 | 1.0 | 0.0 | .05750 | .6333 | .5278 | 1685.14 |
| bounds | 22.6219 | 1.0 | 0.0 | .05750 | .6333 | .5278 | 1786.34 |

**Verdict: non-positive.** Byte-identical checkpoints and training loss to the
`j48f8u` session's cycle 1 (control sha256 `d2f2dc4b...c557e44b`, bounds sha256
`eb81529a...b224a2f`); the `bounds` knob ties the control on the declared
primary (`smoke.structural_similarity`) and every other quality metric — only
p50 latency moved, which alone is not a metric win. Ship gates fail as
expected (`insufficient_n`, missing `held_out`/`adversarial`/`ood`/`rico_held`
suites) — fixture screening only, not a ship claim.

Per `sdlc` autotrain-iteration-delivery: **no stacked PR** for this cycle —
docs-only local commit. Checkpoints are scratch (`sync_checkpoints=false`),
never reusable/promotable/syncable/shippable; no MODEL_CARD promotion entry
(a scratch-checkpoint history row is still recorded per existing convention).

## Environment note

This session's fresh `.venv` (Python 3.12, `pip install -e .`) initially
lacked `torch`, reproducing the already-documented and already-fixed
infra gap in
[`autotrain-cycle-c1-torch-missing-infra-failure.md`](autotrain-cycle-c1-torch-missing-infra-failure.md)
(fix commit `72fdffa`). Installing the pinned CPU wheel
(`scripts/setup_dev_env.sh`'s documented `torch==2.5.1+cpu` step) before this
measured cycle resolved it; no new harness repair was required.

## Next priorities (ranked by the driver)

1. `component-plan` quality hypothesis — already reproduced positive 3 times
   (#1369, #1376, #1378); screen it next (cycle 2).
2. Keep the matched control as the baseline every cycle.
3. Do not re-select the now-exhausted `bounds` arm without a new hypothesis.

Machine evidence:
[`continuous-openui-local-n8vwtq-c1-results.json`](continuous-openui-local-n8vwtq-c1-results.json).
