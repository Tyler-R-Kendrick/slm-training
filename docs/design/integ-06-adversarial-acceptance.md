# INTEG-06 — Adversarial end-to-end acceptance (SLM-573)

**Status:** release-blocking fixture suite. Certifies that the ten required
cross-module adversarial scenarios fail at the intended existing gate and that
positive controls still pass. Does **not** invent a third evidence stack.

## Artifacts

| Artifact | Path |
| --- | --- |
| Matrix | `src/slm_training/resources/formal/integ06_adversarial_matrix.v1.json` |
| Orchestrator | `src/slm_training/formal/integ06_acceptance.py` |
| Verify | `scripts/verify_integ06_adversarial_acceptance.py` |
| Results | [`integ-06-adversarial-acceptance-results.json`](integ-06-adversarial-acceptance-results.json) |
| Tests | `tests/test_formal/test_integ06_acceptance.py` |

## Scenario → gate map

| Scenario | Positive gate | Adversarial fault → gate | Source suites |
| --- | --- | --- | --- |
| S01 singleton zero-neural | `singleton_neural_policy` | incomplete must not force | KERN-07, INTEG-01 |
| S02 exhaustive vs incomplete refutation | `evidence_authorizes_removal` | timeout/skipped/incomplete preserve candidate | EVID-09/11 |
| S03 encoding-bound certificate | honest encoding authorizes | valid cert / wrong encoding rejected | EVID-10/11 |
| S04 exact proposition binding | sealed binding holds | same name / changed prop rejected | EVID-07/11 |
| S05 source/toolchain/axiom drift | (covered by S04 positive) | drift invalidates formal evidence | EVID-07/11 |
| S06 runtime trace integrity | frozen fixture refines | illegal / reorder / stale / hidden forward | KERN-11, INTEG-01 |
| S07 self-healing freeze | (adversarial-only) | weaken / assumption / budget / corpus blocked | HARN-09 |
| S08 trigger preflight | absent → skip | unknown/incomplete → run | INTEG-02, KERN-08 |
| S09 dominance identity | dominance emits | corpus/model/cost identity drift | KERN-08 |
| S10 revmath classifications | corpus + campaign + profile | HARN-11 mutation gates fire | HARN-11, INTEG-03/05 |

## Run

```bash
export PATH="$HOME/.elan/bin:$PATH"
PYTHONPATH=src uv run python -m scripts.verify_integ06_adversarial_acceptance --check
PYTHONPATH=src uv run python -m scripts.verify_integ06_adversarial_acceptance --write
PYTHONPATH=src uv run pytest tests/test_formal/test_integ06_acceptance.py -q
```

## Acceptance

- Every required scenario has ≥1 adversarial case (and a positive control where applicable).
- Every adversarial case `rejected=true` at its declared gate/layer.
- Positive controls pass; `no_failure_becomes_destructive_authority=true`.
- S10 exercises HARN-11 corpus identities plus a hermetic `run_revmath_profile` /
  CampaignStore surface (toolchain + campaign digests in the case detail).

Component: `formal.objects` (bumped with this change).
