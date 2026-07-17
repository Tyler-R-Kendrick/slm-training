# A2 (SLM-38): distribution-aware constrained diffusion decoding (ASAp for MaskGIT)

Date: 2026-07-17 · Track: A2 · Linear: SLM-38 · Motivated-by A1 (SLM-20)

## Motivation (the A1 emptiness diagnosis)

MODEL_CARD E224-E236 report syntax parse = 1.0 with meaningful parse ≈ 0: the
constrained decoder reliably emits grammatically valid but empty layouts. The A1
emptiness probe (`src/slm_training/evals/emptiness_probe.py`, SLM-20) tests the
hypothesis that this is a **decode-time constraint-distortion artifact**, not a
representation failure. Grammar-Aligned Decoding / ASAp (Park et al., NeurIPS
2024, "Grammar-Aligned Decoding", arXiv:2405.21047) shows that hard constraint
masking — zeroing illegal-token logits and renormalizing over only the
grammar-legal set at each step — **discards the removed probability mass and
provably distorts** the sampled distribution away from the model's true
grammatical preferences. Combined with a length prior it can make the shortest
valid program (the empty document) the argmax. A2 is the direct decode-side fix.

## What shipped

The **single-step ASAp approximation** wired into the MaskGIT unmask loop. Full
ASAp maintains a per-prefix approximation of *expected future grammaticality*
and re-weights across repeated samples; the honest minimal version here corrects
using the removed mass **observed at the current position** instead — labelled
throughout as the single-step approximation, not full ASAp.

### Primitives — `src/slm_training/dsl/grammar/fastpath/gate.py`

- `AsapLedger` — records, per committed MaskGIT position, the model probability
  mass the grammar constraint removed (`positions`, `removed_mass_sum`,
  `mean_removed_mass`, `max_removed_mass`, `nonzero_removed`). Live telemetry
  that quantifies the distortion; **not a ship metric**.
- `removed_mass(probs_1d, legal_ids)` — `1 − Σ p(t) over legal t`. Returns
  `1.0` when the legal set is empty (fail-closed) and clamps to `[0, 1]` (no
  NaN / negative ledger entries).
- `asap_reweight(probs_1d, legal_ids, *, alpha)` — the corrected legal
  distribution. Plain constraint decode renormalizes `q0(t) = p(t) / S`
  (`S = 1 − M`, `M` = removed mass), inflating every legal token's confidence by
  exactly the removed mass `M`. The single-step correction damps that:

  ```
  gamma = clip(1 − alpha · M, 1e-3, 1)      # alpha in [0, 1]
  q(t)  = p(t) ** gamma / Σ_t p(t) ** gamma
  ```

  Because `x ** gamma` is monotone for `x > 0`, the **winning legal token never
  changes and no illegal token is ever admitted (fail-closed)** — the legal
  distribution is only *flattened* where the constraint removed a lot of mass,
  so the confidence-scheduled MaskGIT unmask loop **defers those high-distortion
  positions**. `alpha = 0` or `M = 0` reproduces plain renormalization exactly.

### Commit gate — `src/slm_training/models/parallel_decode.py`

- `asap_filter_commits(flat_idx, probs, *, length, legal_ids_fn, ledger, alpha,
  defer_mass, last_step)` — the ASAp correction realized in the discrete
  schedule. For each batch-0 commit candidate it records the removed mass and
  **defers commits whose removed mass exceeds `defer_mass`** (leaves them masked
  for a later step, when more context has resolved). Fail-safe guarantees: the
  single lowest-removed-mass candidate is always kept (decode always makes
  progress); nothing is deferred on the final step (decode terminates);
  legality-unknown positions pass through unrecorded.

### Decode wiring — `src/slm_training/models/twotower.py`

`_generate_maskgit_one` builds an `AsapLedger` when `asap_reweight` is set and
grammar is active, and calls `asap_filter_commits` before the commit loop with a
`legal_ids_fn` that computes the grammar-legal set (surface DFA
`allowed_id_set(next_terminals())`) from the committed left prefix. Measurement
is **frontier-only**: a candidate position is measured only when its entire left
context (positions `1..t−1`; position 0 is BOS) is already committed, so the
prefix is a clean, parseable grammar prefix — that is the only place the
constraint renormalization is well-defined on a mask-punctured parallel canvas.
The ledger totals are merged into `DecodeStats`
(`asap_positions`, `asap_removed_mass_sum`, `asap_max_removed_mass`,
`asap_nonzero_removed`).

**Default off = byte-identical.** With `asap_reweight=False` the ledger is never
created and `flat_idx` is untouched; existing decode is unchanged.

## Config wiring (standard 3-point)

`asap_reweight` (bool, default `False`), `asap_alpha` (float, default `1.0`),
`asap_defer_mass` (float, default `0.5`) on **`TwoTowerConfig`**,
**`ModelBuildConfig`**, and the matrix **`Experiment`** dataclass, threaded
through `harnesses/model_build/factory.py` (build + `RUNTIME_OVERRIDE_FIELDS`)
and `scripts/run_quality_matrix.py` (`_train_cfg`, the eval overlay, and
`_apply_decode_overrides`).

## Matrix row — E268 (matrix `v13`)

`_v13_experiments()` registers **E268** (`qx_e268_a2_asap_reweight`), matched to
the A5 lattice-campaign baseline **E240** (strict compiler-tree policy, eval-only
from the same frozen checkpoint), **differing only by `asap_reweight`** (pinned
by `tests/test_scripts/test_quality_matrix_v13.py`). E-id **E268** chosen because
main has advanced through ~E263 and concurrent open PRs claimed E259-E267.

**Honesty caveat (important).** ASAp lives in the MaskGIT unmask loop, but the A5
recipe decodes **LTR through the compiler-tree path** (`grammar_ltr_primary=True`,
`compiler_decode_mode="tree"`). Under the exact matched recipe decode never
enters `_generate_maskgit_one`, so the ASAp ledger stays **dormant** and E268
reproduces E240 byte-for-byte. Registering the row as a pure `asap_reweight`
delta keeps the "differ only by the flag" contract; the mechanism's liveness is
therefore proven on the MaskGIT decode path by tests, not by this row. A future
A2 quality verdict needs the A5 frozen checkpoint on a real suite **and** a
MaskGIT-decode configuration for ASAp to engage.

## Measured evidence (fixture / CPU — ledger liveness only)

Driving the constrained MaskGIT loop on a single fixture record with a
random-init tiny denoiser (`d_model=64`, `denoiser_layers=2`, `gen_steps=8`,
`seed=0`, CPU, scratch backend; `grammar_ltr_primary=False`,
`compiler_decode_mode="off"`):

| Arm | asap_positions | Σ removed mass | mean | nonzero | max |
| --- | --- | --- | --- | --- | --- |
| `asap_reweight=False` | 0 | 0.0 | — | 0 | — |
| `asap_reweight=True` | 8 | 8.0 | 1.0 | 8 | 1.0 |

The ledger is **LIVE**: 8 frontier positions measured, all with nonzero removed
mass. `mean removed mass = 1.0` is the **expected** random-init result — the
untrained denoiser assigns ~zero probability to grammar-legal continuations, so
the constraint removes ~all mass; the legal sets themselves are non-empty (7/1/10/1
ids at `""`/`root`/`root=`/`root=Card`). The **partial-distortion regime**
(`0 < M < 1`) that ASAp is designed to correct, and any change in decoded output,
require a **trained checkpoint** — the deferral produced no net output change at
this tiny budget (both arms decoded identically). **meaningful_parse is ≈ 0 on
both arms at fixture scale (random init); the frontier quality verdict vs A5 is
unrun-at-scale.**

Run metadata: device CPU · scratch backend · gen_steps 8 · matrix set v13 ·
suite n = 1 fixture record · honesty mode = wiring + ledger-liveness · ship-gate
n/a (decode lever; no gate changed or weakened).

## Tests

- `tests/test_dsl/test_asap_gate.py` (12) — `removed_mass` records masked-out
  mass and fails closed on empty legal set; `asap_reweight` is a valid
  distribution that deterministically differs from plain renormalization when
  `alpha·M > 0`, equals it at `alpha=0`, is identity at `M=0`, preserves the
  argmax, and returns empty on an empty legal set; `AsapLedger` aggregation +
  clamping; `asap_filter_commits` defers high-distortion positions, keeps all on
  the last step, keeps the lowest for progress, passes unknown legality through
  unrecorded, and is a no-op on empty input.
- `tests/test_models/test_asap_constrained_decode.py` (3) — end-to-end on the
  MaskGIT decode path: flag-off ledger is dormant (mechanism inert); flag-on
  records nonzero removed mass (live); flag-off decode is deterministic.
- `tests/test_scripts/test_quality_matrix_v13.py` (1) — E268 registered and
  differs from E240 only by `asap_reweight` (matched control).

New: 16 tests. Full touched suites via `.githooks/check-changed`:
**439 passed, 4 skipped, 15 deselected** (`tests/test_dsl`,
`tests/test_harnesses/model_build`, `tests/test_models`, `tests/test_scripts`).
Checks: ruff (changed files) pass; `python -m scripts.repo_policy` ok.

## Not done (honest)

- **No quality win claimed.** meaningful_parse on both arms is fixture-zero;
  the A5-vs-A2 frontier verdict is unrun-at-scale.
- **No checkpoint promoted** → `docs/MODEL_CARD.md` and the README model-card
  summary are intentionally unchanged.
- **Single-step, not full ASAp.** No expected-future grammaticality estimate or
  cross-sample refinement; the correction uses only the current position's
  removed mass.
- **E268 dormant under the A5 recipe** (LTR compiler-tree decode); a MaskGIT
  decode configuration is required to engage the lever at scale.
