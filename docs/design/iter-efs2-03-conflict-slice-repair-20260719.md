# EFS2-03 — Conflict-slice localized remask vs full remask and suffix rollback (2026-07-19)

Fixture-grade implementation requested by SLM-113. Machine-readable evidence:
[`iter-efs2-03-conflict-slice-repair-20260719.json`](iter-efs2-03-conflict-slice-repair-20260719.json).
Linear SLM-113.

## What ran

The new `slm_training.harnesses.experiments.conflict_slice_repair` module was
exercised by `scripts/run_efs2_03_conflict_slice_fixture.py` over three
synthetic injected topology conflicts and five repair policies:

- `none` — no repair, preserves the failed state for attribution;
- `suffix_rollback` — retract decisions from the highest implicated decision
  level to the end;
- `full_remask` — remask all mutable program nodes/holes;
- `conflict_slice` — remask the certified slice plus dependency frontier only;
- `conflict_slice_expanded` — one-hop expansion of the slice for underlocalized
  cases.

### Synthetic fixtures

| Fixture | Stage | Completeness | Failing nodes | Frontier | Protected |
| --- | --- | --- | --- | --- | --- |
| wrong_production | grammar | EXACT | 4 | 1, 3 | 2 |
| dangling_binding | binding | SOUND_OVERAPPROX | 5, 9 | 2, 10 | 2 |
| heuristic_schema | schema | HEURISTIC | 3, 6 | 1, 7 | 2 |

### Policy outcomes (mean over seeds 0–2)

| Fixture | Policy | Recovery | Mean remasked | Mean preserved | Protected mutations |
| --- | --- | ---: | ---: | ---: | ---: |
| wrong_production | none | 0.00 | 0.0 | 11.0 | 0 |
| wrong_production | suffix_rollback | 0.00 | 3.0 | 8.0 | 0 |
| wrong_production | full_remask | 0.00 | 9.0 | 2.0 | 0 |
| wrong_production | **conflict_slice** | **1.00** | **3.0** | **8.0** | 0 |
| wrong_production | conflict_slice_expanded | 1.00 | 4.0 | 7.0 | 0 |
| dangling_binding | conflict_slice | 0.00 | 3.0 | 8.0 | 0 |
| heuristic_schema | conflict_slice | 0.00 | 0.0 | 11.0 | 0 |

The fixture demonstrates that localized repair recovers only when the slice is
`EXACT` and the policy touches the true failing nodes.  `SOUND_OVERAPPROX` and
`HEURISTIC` slices correctly refuse to authorize hard localized repair in this
synthetic contract.

## Added artifacts

- `src/slm_training/harnesses/experiments/conflict_slice_repair.py` —
  `ConflictSliceV1`, `TopologyNode`, `RepairTrace`, `RepairOutcome`, five repair
  policies, deterministic injected fixtures, matched-budget comparison, and
  replayable trace persistence.
- `scripts/run_efs2_03_conflict_slice_fixture.py` — wiring fixture that builds
  three synthetic conflicts, runs all policies over seeds `0,1,2`, and writes
  durable evidence JSON.
- `tests/test_harnesses/experiments/test_conflict_slice_repair.py` — 13
  regression tests covering slice authorization, every policy, protected-node
  preservation, fingerprint fail-closed behavior, determinism, and budget
  truncation.
- `docs/design/iter-efs2-03-conflict-slice-repair-20260719.md` and `.json`.
- `outputs/runs/efs2-03-conflict-slice/iter-efs2-03-20260719/summary.json` and
  per-fixture `outcomes_*.json`.
- `src/slm_training/resources/versions.json` — `harness.experiments` bumped to
  `v5`.

## Verdict

`diagnostic_only`

The conflict-slice schema, repair policies, matched-budget accounting, and trace
replay machinery are implemented and tested.  The fixture uses simplified
synthetic topology trees, not the live `GrammarDiffusionModel` decode loop, and
makes no quality or ship claim.  A production EFS2-03 run must integrate the
slicers with real compiler/solver evidence, run against canonical frontier
checkpoints, and demonstrate recovery on natural failure traces before any
runtime policy adoption.

## Honesty and limits

- **Wiring evidence only, not a ship claim.** No checkpoint is loaded, no model
  runs, and the conflicts are hand-designed.
- **Simplified topology model.** The fixture does not exercise the actual
  `GrammarDiffusionModel._decode_one` loop or finite-domain solver; it proves
  the policy contract and accounting in isolation.
- **Synthetic recovery signal.** Recovery is a structural predicate in the
  fixture, not a binding-aware semantic outcome.
- **Three seeds only.** The issue asks for at least three seeds; the fixture
  uses exactly three.
- Component `harness.experiments` was bumped to `v5`.
