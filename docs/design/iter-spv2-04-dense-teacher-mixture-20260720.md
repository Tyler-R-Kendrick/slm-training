# SPV2-04: Dense teacher distributions on the winning gold/on-policy mixture (SLM-152)

**Linear issue:** SLM-152
**Branch:** `agent/slm-152`
**Date:** 2026-07-20
**Status:** wiring fixture / data scaffolding; SLM-152 acceptance incomplete

Evidence: [iter-spv2-04-dense-teacher-mixture-20260720.json](iter-spv2-04-dense-teacher-mixture-20260720.json).
Harness: [`src/slm_training/evals/dense_teacher_mixture.py`](../../src/slm_training/evals/dense_teacher_mixture.py),
fixture runner: [`scripts/run_spv2_04_dense_teacher_mixture_fixture.py`](../../scripts/run_spv2_04_dense_teacher_mixture_fixture.py).
Tests: [`tests/test_evals/test_dense_teacher_mixture.py`](../../tests/test_evals/test_dense_teacher_mixture.py).

## What changed

Extended the EFS3-01 (SLM-118) state-source comparison with SPV2-03 (SLM-151)
dense legal-set teacher labels, producing immutable DAgger-style round snapshots.

- `src/slm_training/evals/dense_teacher_mixture.py`
  - `DenseTeacherExampleV1` — solver-state row optionally carrying a complete
    legal-set teacher probability distribution aligned to `legal_actions`.
  - `AcquisitionPolicy` — uniform, high-divergence, low-accepted-rank,
    verifier-failure-cone, high-regret, and stratified-mixture policies.
  - `attach_teacher_distribution()` — binds SPV2-03 teacher traces to EFS3-01
    solver-state rows by `state_fingerprint`, leaving gold rows as hard-label
    controls.
  - `acquire_teacher_labeled_states()` — deterministic selection under a fixed
    teacher-label budget.
  - `build_dense_teacher_snapshot()` — builds matched arms with a enforced gold
    replay floor and the same decision budget:
    `gold_only`, `mixed_no_teacher`, `mixed_teacher_argmax`,
    `mixed_teacher_kl`, `targeted_teacher_kl`, `on_policy_teacher_kl`.
  - `compare_dense_teacher_mixtures()` — multi-seed wrapper used by the fixture.
- `scripts/run_spv2_04_dense_teacher_mixture_fixture.py`
  - Deterministic fixture that synthesizes solver-state rows and SPV2-03 teacher
    traces, then emits the six canonical arms over three seeds.
- `tests/test_evals/test_dense_teacher_mixture.py`
  - Regression tests for teacher alignment, gold-row exclusion from teacher
    labels, acquisition policies, cross-split leak rejection, held-out exclusion,
    budget limits, and argmax-arm conversion.
- `src/slm_training/resources/versions.json`
  - Bumped `evals.scoring` to `v11`.

## Fixture run

Command:

```bash
python -m scripts.run_spv2_04_dense_teacher_mixture_fixture --run-id iter-spv2-04-20260720-v11
```

Recipe: CPU; synthetic solver states; fixture teacher distributions; no external
model; deterministic seeds 0, 1, 2.

### Aggregate arm sizes (mean over seeds)

| Arm | Mean size | Min | Max |
| --- | ---: | ---: | ---: |
| gold_only | 64.0 | 64 | 64 |
| mixed_no_teacher | 64.0 | 64 | 64 |
| mixed_teacher_argmax | 64.0 | 64 | 64 |
| mixed_teacher_kl | 64.0 | 64 | 64 |
| on_policy_teacher_kl | 51.7 | 50 | 54 |
| targeted_teacher_kl | 64.0 | 64 | 64 |

The `on_policy_teacher_kl` arm is smaller because many on-policy states belong to
held-out problem groups and are removed by the cross-split leak guard; the other
arms refill from the gold pool to meet the decision budget.

### Honesty invariants exercised

- Gold rows do not receive dense teacher distributions; they remain hard-label controls.
- Teacher probabilities are aligned to the identical `legal_actions` list and renormalized.
- Held-out split rows are excluded from every train arm.
- Cross-split leakage guard rejects any `split_group_id` appearing in both train and held-out.
- The gold replay floor is enforced and cannot be silently lowered.
- Acquisition policy and teacher-label budget are fixed per round and reported in lineage.

## Honest caveats

- **Wiring-only / no checkpoint loaded.** No solver trace replay, external teacher model,
  model decode, or real student-teacher divergence was measured.
- **Synthetic rows and teacher distributions.** The fixture uses deterministic synthetic
  states and logits, not real on-policy rollouts or the SLM-108 external scorer.
- **No quality claim.** This fixture proves the schema, acquisition, and snapshot merge
  work; it does not claim that any mixture improves meaningful-parse or ship gates.
- **Default training path unchanged.** The new module is only invoked by the fixture script
  and tests; existing training loops are unaffected.

## Verification checklist

- [x] `pytest tests/test_evals/test_dense_teacher_mixture.py` — 11 passed.
- [x] `python -m scripts.run_spv2_04_dense_teacher_mixture_fixture --run-id iter-spv2-04-20260720-v11` — bundle written.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — passed.
- [x] `python -m scripts.verify_version_stamps --check` — ok.
- [x] `git diff --check` — clean.
