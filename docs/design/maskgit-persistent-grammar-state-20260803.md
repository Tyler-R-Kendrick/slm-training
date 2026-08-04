# MaskGIT persistent grammar state — preregistered, falsified, reverted (2026-08-03)

Decode-lever experiment on the MaskGIT generate path. Hypothesis wiring was
implemented, measured against a preregistered criterion, **falsified**, and
**fully reverted**. No production code changed; four regression tests pinning
engine-reuse equivalence were kept. Profile artifacts:
`outputs/runs/profile_maskgit_before.json`, `profile_maskgit_after.json`,
`profile_maskgit_after_noinc.json`, `profile_maskgit_pristine_noinc.json`.

## Hypothesis

`_generate_maskgit_one` passes no `GrammarDecodeState` to
`pick_constrained_token` / `force_emit_token_id` (unlike every LTR path), so
each constrained pick decodes the full prefix and builds a cold
`OpenUIIncrementalEngine` (`_openui_engine` in `dsl/pack.py` returns a fresh
instance per call), and each candidate probe builds another
(`dfa_admits_token`'s copy-probe path requires `state is not None`). Claim:
one persistent state per trajectory + one hoisted `admit_fill` engine cuts
MaskGIT generate wall time.

Preregistered criterion (from the approved plan, before measurement):
(i) `sample_outputs` byte-identical to baseline, else reject outright;
(ii) `dfa_sync_ms_mean` −≥30% **and** `sec_per_generate` −≥10%;
(iii) `dfa_incremental_advances > 0` with full-sync ratio < 0.5.
Failure of (ii)/(iii) ⇒ revert as pure complexity.

## Setup

`python -m scripts.profile_generate --checkpoint
outputs/runs/s1_d64/checkpoints/last.pt --maskgit --rounds 2` (CPU, WSL2,
3 fixture prompts × 2 rounds; every run ≪ `MAX_RUN_MINUTES`). The committed
playground demo checkpoint fails `require_current_output_contract`
(`symbol_only/v2`), so the most recent lever-loop checkpoint (`s1_d64`,
SLM-304) was used. All runs emit `root = Separator()` for all prompts.

## Results

| run | code | `grammar_incremental_state` | sec/generate | `dfa_sync_ms` mean | `pick_ms` mean | outputs |
| --- | --- | --- | --- | --- | --- | --- |
| before | pristine | True (default) | **18.89** | 9.91 | 615 | baseline |
| after | wired | True | **20.16** (+6.7%) | 9.92 (+0.1%) | 710 | byte-identical |
| after-noinc | wired | False | **6.83** | 19.03 | 987 | byte-identical |
| pristine-noinc | pristine | False | **5.25** | 19.46 | — | byte-identical |

Criterion: (i) **pass** — byte-identical everywhere; (ii) **fail**; (iii)
**fail** — folded engine counters were ~zero. **Falsified; reverted.**

## Why it failed (root causes, measured)

1. **The guard excludes most picks.** A stateful pick is only sound when the
   prefix contains no mask holes (a spliced-mask prefix must stay on today's
   stateless behavior, and appended unadvertised tokens can poison a bound
   completion session). Instrumented single generate: **20 of 149 picks
   (13%)** were eligible. MaskGIT's confidence-order unmasking keeps holes
   left of most positions — the mechanism barely engages.
2. **The targeted cost was already negligible.** `dfa_sync_ms` ≈ 10 ms of an
   18,890 ms generate (0.05%). There was never 10% of wall to win here.
3. **The dominant cost is the repair phase, not the pick loop.** cProfile of
   one pristine generate (80.9 s total): `build_completion_forest` 75.6 s
   (124 calls), of which `_constrained_ltr_repair` 72.9 s and
   `_ensure_valid_openui` → `_ltr_repair_from_bos` 53.5 s;
   `exact_forced_token_id` → `completion_domain` → `terminal_witness` = 71.0 s
   over just 36 calls. `_propose` (the loop this experiment targeted): 5.5 s.
   This reconfirms SLM-304's finding — grammar forest rebuilds, not model
   forwards or engine syncs, are the eval cost.

## Discovery: `--no-incremental` is ~3–3.6× faster on this path

`grammar_incremental_state=False` cut sec/generate from 18.89 to **6.83**
(wired code) and **5.25** (pristine code) with byte-identical outputs.

**HEAD re-validation (eba6db30, 2026-08-03).** The original runs executed on a
stale 2026-07-30 working tree (2581bf49-era; an accidental reset later
recovered — twotower.py differs by 852 lines). Re-measured on HEAD:
incremental 7.49 s/gen vs no-incremental **3.27 s/gen** (**2.29×**),
byte-identical outputs (`outputs/runs/profile_maskgit_head_inc.json`,
`profile_maskgit_head_noinc.json`). HEAD itself is ~2.5× faster than the
Jul-30 tree on this fixture, and the no-incremental advantage persists —
the discovery replicates across both code states. The persistent-state machinery (P1) binds packed completion sessions
whose forest work dominates the repair phase; the stateless path does
strictly less of it here. P1's equivalence evidence came from the LTR path —
its cost profile does not transfer to the MaskGIT+repair path at this scale.
This inverts the experiment's premise: on this fixture the fastest correct
configuration is *less* persistent state, not more.

Follow-up candidates (not acted on, need their own preregistered runs):
- Screen `grammar_incremental_state=False` for the MaskGIT/eval path on the
  quality suite (wall-clock is eval-evidence quality: SLM-304 showed the 12 s
  decode-timeout knife-edge corrupts headline metrics).
- Attack `build_completion_forest` in `_ltr_repair_from_bos` /
  `completion_domain` (`terminal_witness` recursion) — the actual 74–94% cost
  center, unchanged since SLM-304.

## Telemetry gap (still open)

Engine counters `full_syncs` / `incremental_advances` / `copy_probes` /
`sync_ms` (`OpenUIIncrementalEngine.stats`) are computed but consumed nowhere
in `decode_stats.py`; the incremental-vs-full-sync ratio cannot be read from
any shipped telemetry. The experiment's temporary fold worked and was
reverted with the wiring; re-adding it is trivial if a future experiment
needs it (map the seven counters into `DecodeStats` next to
`collect_completion_session_delta`).

## Kept artifacts

- `tests/test_dsl/test_grammar_fastpath.py`: four new pins —
  reused-engine ≡ fresh-engine across extension / non-monotonic / illegal /
  recovery prefixes; `admit_fill` engine reuse ≡ fresh; stateful pick parity
  with stateless; and the packed-session contract that an unadvertised
  appended token detaches the session (`require_advertised=False`) instead of
  raising — the graceful form of the hazard this experiment had to guard
  (HEAD hardened it between the Jul-30 measurement and this landing).
- Profile JSONs under `outputs/runs/` (paths above).

No version bump: no stamped component changed (wiring reverted; tests and
docs only).

## Honesty

Fixture-profile evidence on one checkpoint (`s1_d64`) and three fixture
prompts; single-run timings on a shared WSL2 box (denoiser_ms varied ±11%
between identical-code runs — treat ±10% wall as noise; the 3–3.6×
no-incremental delta is far outside it). No quality suite was run; no ship
claim; the `--no-incremental` discovery is a screening lead, not a default
change.
