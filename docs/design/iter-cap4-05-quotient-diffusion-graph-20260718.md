# CAP4-05: Quotient-state diffusion graph diagnostics (SLM-99)

**Linear issue:** SLM-99
**Branch:** `agent/slm-99-cap4-05-quotient-diffusion-graph`
**Date:** 2026-07-18
**Status:** wiring fixture / exact small-graph diagnostics; SLM-99 acceptance incomplete

Evidence: [iter-cap4-05-quotient-diffusion-graph-20260718.json](iter-cap4-05-quotient-diffusion-graph-20260718.json).
Harness: [`src/slm_training/dsl/analysis/arity/diffusion_graph.py`](../../src/slm_training/dsl/analysis/arity/diffusion_graph.py),
runner: [`scripts/run_quotient_diffusion_fixture.py`](../../scripts/run_quotient_diffusion_fixture.py).
Tests: [`tests/test_dsl/test_diffusion_graph.py`](../../tests/test_dsl/test_diffusion_graph.py).

## What changed

Added a Torch-free quotient-state diffusion graph diagnostics module under the existing arity analyzer:

- `src/slm_training/dsl/analysis/arity/diffusion_graph.py`
  - `QuotientDiffusionGraph` built from `(state, action, next_state)` transitions or from ordered `GrammarDecisionTrace` records.
  - Exact weak/strong connected components (Tarjan), diameter, average path length, stationary distribution, spectral gap, conductance, and mixing-time bound for manageable graphs.
  - Estimated fallbacks for diameter, spectral gap, and conductance when graphs are too large for exact computation; every estimated quantity is labeled.
  - Reversibility / detailed-balance check against the stationary distribution.
  - Six matched diffusion kernels: surface-token mask, independent production mask, AST-subtree mask, typed-hole mask, quotient random walk, and posterior-weighted walk.
  - `information_schedule()` and `recommend_information_balanced_schedule()` from per-timestep `posterior_entropy_bits` / `completion_support_size_exact`.
  - `compare_kernels_at_matched_loss()` so kernels are compared at equal conditional information loss rather than equal nominal timestep.
- `src/slm_training/dsl/analysis/arity/__init__.py`
  - Re-exports the new public API.
- `src/slm_training/resources/versions.json`
  - Registers `analysis.arity.diffusion_graph` at `v1`.
- `scripts/run_quotient_diffusion_fixture.py`
  - Builds exact directed-ring, barbell, and path-with-self-loops fixtures; builds a trace-derived graph from synthetic `GrammarDecisionTrace` rows; runs diagnostics; compares kernels; writes a version-stamped JSON bundle.
- `tests/test_dsl/test_diffusion_graph.py`
  - Regression tests for graph diagnostics on known small graphs, trace graph construction, kernel normalization/seed determinism, information schedules, and matched-loss comparison.

## Fixture run

Command:

```bash
python -m scripts.run_quotient_diffusion_fixture \
  --run-id fixture-20260718 \
  --seed 42 \
  --output-dir outputs/runs/cap4-05-quotient-diffusion
```

Recipe: CPU; synthetic states; exact small-graph computation; no model train.

### Graph diagnostics summary

| graph | vertices | edges | strongly connected | diameter | spectral gap | conductance | mixing bound (eps=0.25) | reversible |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| directed ring n=6 | 6 | 6 | yes | 5 | ~0 | 0.333 | ∞ (periodic) | no |
| undirected barbell 4+4 | 8 | 26 | yes | 3 | 0.113 | 0.077 | 31.27 | yes |
| path with self-loops n=5 | 5 | 9 | no | 4 | 0.500 | 0.000 | 2.77 | no |
| trace graph | 5 | 19 | yes | 4 | ~0 | 0.500 | ∞ (periodic) | no |

The barbell fixture exposes the expected bottleneck: low conductance (0.077) and a spectral gap of 0.113, predicting slow mixing despite being connected. The directed ring and trace graph are periodic, so the spectral gap is effectively zero and the Cheeger-style mixing bound is infinite; any production kernel built on these transitions must add laziness or a restart.

### Kernel contract summary

All six kernels expose transition probabilities, schedule, terminal distribution, invalid-state policy, reversibility, posterior exactness, and compute cost. The four masking kernels are compared at a matched 2-bit conditional information loss in the fixture; the quotient-random-walk and posterior-weighted kernels are instantiated from the ring/barbell graphs.

### Information schedule

The synthetic trace schedule starts at ~4.0 bits of posterior entropy and ~16 completion candidates at timestep 0, falling to ~0.19 bits and 1 candidate by timestep 19. The recommended equal-loss schedule over 8 steps is included in the JSON artifact.

## Honest caveats

- **Wiring-only / diagnostic evidence.** No diffusion model was trained; no denoising quality, parse, meaningful-program, or latency measurement is claimed.
- **Abstract kernels.** The six kernels are defined and normalized on small abstract state spaces; they are not yet wired into `grammar_diffusion.py::_corrupt_topology`.
- **Exact only for tiny graphs.** Conductance and spectral gap are exact for graphs up to 12 vertices; larger graphs fall back to sampled estimates.
- **No checkpoint.** `model_card_updated: false`; nothing is promoted or recorded in `docs/MODEL_CARD.md`.
- SLM-99 remains incomplete until bounded OpenUI quotient graphs are built from real compiler states, reversible topology-edit edges are integrated from SLM-71, and a denoising matrix is run at matched conditional information loss.

## Verification checklist

- [x] `pytest tests/test_dsl/test_diffusion_graph.py` — 15 passed.
- [x] `python -m scripts.run_quotient_diffusion_fixture --run-id fixture-20260718 --seed 42` — bundle written.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — 776 passed, 5 skipped, 12 deselected.
- [x] `python -m scripts.verify_version_stamps --check` — ok.
- [x] `git diff --check` — clean.
