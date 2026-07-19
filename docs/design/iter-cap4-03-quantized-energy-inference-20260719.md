# CAP4-03: Quantize local legal-action energies and compare greedy with exact lattice inference (SLM-97)

**Linear issue:** SLM-97
**Branch:** `agent/slm-97-cap4-03-quantized-energy-inference`
**Date:** 2026-07-19
**Status:** wiring fixture / eval scaffolding; SLM-97 acceptance incomplete

Evidence: [iter-cap4-03-quantized-energy-inference-20260719.json](iter-cap4-03-quantized-energy-inference-20260719.json).
Harness: [`src/slm_training/evals/quantized_energy_inference.py`](../../src/slm_training/evals/quantized_energy_inference.py),
fixture runner: [`scripts/run_quantized_energy_inference_fixture.py`](../../scripts/run_quantized_energy_inference_fixture.py).
Tests: [`tests/test_evals/test_quantized_energy_inference.py`](../../tests/test_evals/test_quantized_energy_inference.py).

## What changed

Added eval-only wiring for the CAP4-03 hypothesis that very low-arity local
action/edge energies can still rank compiler-live candidates globally when
combined with exact inference, and that coarse quantization creates path
score collisions that greedy local selection cannot resolve.

- `src/slm_training/evals/quantized_energy_inference.py`
  - `ScoreSemantics` enum distinguishing additive-edge and cost-to-go score
    interpretations.
  - `LegalAction`, `EnergyStage`, `EnergyProblem` — small, replayable acyclic
    lattices where each stage lists legal actions with local energies.
  - `EnergyQuantizer` — calibrates a `QuantFormat` level grid to the observed
    energy range and quantizes local energies (fp16 reference, binary,
    ternary, symmetric four-level, learned four-level-with-zero, INT4).
  - `InferenceMode` enum: `greedy_local` and `exact_viterbi`.
  - `PathSelection` / `FormatResult` / `CompareResult` — typed outcomes with
    path, total original/quantized energy, exactness flag, tie-class size,
    and score distribution.
  - `compare_quantized_energy_inference()` — runs the same problem across all
    requested formats and both inference modes, with a version stamp.
- `scripts/run_quantized_energy_inference_fixture.py`
  - Synthetic fixture generating random 4-stage lattices, emitting per-format
    greedy/exact totals, greedy/exact agreement counts, max tie-class sizes,
    and total enumerated path counts.
- `tests/test_evals/test_quantized_energy_inference.py`
  - Regression tests for greedy/exact agreement on independent additive stages,
    binary score collapse, tie-class detection, UNKNOWN action exclusion,
    default format set, and JSON round-trip.
- `src/slm_training/resources/versions.json`
  - Bumped `evals.scoring` to `v4`.

## Fixture run

Command:

```bash
python -m scripts.run_quantized_energy_inference_fixture --run-id iter-cap4-03-20260719
```

Recipe: CPU; synthetic additive local energies; no checkpoint or learned scorer.

### Aggregate comparison

| Format | Problems | Greedy total | Exact total | Greedy==Exact | Max tie class | Paths enumerated |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 8 | 103.101 | 103.101 | 8 | 1 | 384 |
| binary | 8 | 290.736 | 290.736 | 8 | 48 | 384 |
| ternary | 8 | 63.741 | 63.741 | 8 | 24 | 384 |
| symmetric_four_level | 8 | 133.660 | 133.660 | 8 | 24 | 384 |
| learned_four_level_zero | 8 | 109.832 | 109.832 | 8 | 18 | 384 |
| int4 | 8 | 104.012 | 104.012 | 8 | 9 | 384 |

For these independent additive lattices greedy and exact select the same path,
but coarse formats (binary/ternary) produce large tie classes that would make
global inference ambiguous on dependent stages. The fixture intentionally
exposes this collision pathology rather than hiding it.

## Honest caveats

- **Wiring-only / no learned scorer.** Energies are synthetic; the harness does
  not wrap `CandidateEnergyScorer` or train a cost-to-go model.
- **Small acyclic lattices.** Exact inference is exhaustive enumeration over a
  deliberately tiny decision space, not the bounded solver/closure controller.
- **Independent stages.** Real compiler lattices have dependencies between
  decisions; this fixture isolates quantization/inference mechanics.
- **No quality claim.** The numbers prove the schema, quantizer, and comparator
  work; they do not claim a Pareto win for any format.

## Verification checklist

- [x] `pytest tests/test_evals/test_quantized_energy_inference.py` — 8 passed.
- [x] `python -m scripts.run_quantized_energy_inference_fixture --run-id iter-cap4-03-20260719` — bundle written.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — 428 passed, 12 deselected.
- [x] `python -m scripts.verify_version_stamps --check` — ok.
- [x] `git diff --check` — clean.
