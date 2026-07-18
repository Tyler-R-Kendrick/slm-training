# LDI0-03 — objective-support admission and bounded diagnostics

Date: 2026-07-18
Status: **objective-support admission gate + bounded-diagnostic runner landed with
tests; trainer-entry wiring and the full Tier-1/Tier-2 diagnostic content remain
follow-on. No training, checkpoint, model-quality, or ship claim.**

## Why this exists

E283 repaired every held-out grammar-**state** support signature, yet E284 still
found **35 held-out objective conflicts**: the FTPO objective depends on the
sampled bad-action set, which is *not* part of the state-support signature, so a
corpus can pass state support while its objective (action-partition) support is
deficient. E285 and E286 then tried a more exact profile but blew the experiment
envelope and produced **invalid evidence**:

- **E285** — no cumulative wall-clock deadline; ran past 25 minutes, operator-stopped,
  **no report** → invalid evidence.
- **E286** — chunked batched vector-Jacobian products; killed at 283.62 s by the
  five-minute envelope, **no report**; the implementation was removed rather than
  retained on a fast unit test.

Both are marked invalid historical evidence here; no metric, comparison, or training
decision may be inferred from them.

## Support criteria compared

| Criterion | Keys on | Catches E284? |
| --- | --- | --- |
| V1 state support (`decision_support_signature`) | decision kind + legal tokens + **good** tokens | No — excludes the sampled negatives |
| V1 objective signature (`decision_signature`) | + **bad** tokens | Diagnoses it, but not wired as an admission gate |
| **V2 objective-view support** (`objective_view_support`, this issue) | materializer id/config hash + **good/bad** partition, split-aware | **Yes** — held-out objective signatures without train support fail admission |

The V2 gate is the fix: `admit_semantic_corpus` fails closed before training when a
corpus (a) contains a non-trainable view (e.g. the `constraint_shadow` diagnostic),
(b) carries a materializer that does not match the requested objective, or (c) lacks
train support for any held-out objective signature — naming what is missing so
support can be repaired (never by copying held-out programs).

## Bounded runtime (the E285/E286 fix)

`decision_diagnostics.run_bounded_stages` runs diagnostic stages under one cumulative
`DiagnosticBudget` (default and hard cap five minutes), read through a `time`
reference so a deterministic fake clock drives the tests. On expiry the run is
`expired` with **`result: None`** — no partial result is ever presented as a
diagnostic result — and a full-parameter Tier-2 request is refused as
`not_authorized` rather than replaying the invalid E285 full-parameter profile.
`write_diagnostic_report` persists reports atomically (mkstemp + fsync + os.replace).

## Landed in this iteration

- `src/slm_training/harnesses/preference/decision_events_v2.py`:
  `objective_view_signature`, `objective_view_support`, `admit_semantic_corpus`.
- `src/slm_training/harnesses/preference/decision_diagnostics.py`: `DiagnosticBudget`,
  `Deadline`, `run_bounded_stages`, `not_authorized_report`, `write_diagnostic_report`.
- Tests: the E284 pattern (passes state support, fails objective support),
  constraint-shadow and materializer-mismatch refusals
  (`tests/test_harnesses/preference/test_decision_events_v2.py`), and the deterministic
  fake-clock deadline / no-result-on-expiry / not-authorized cases
  (`tests/test_harnesses/preference/test_decision_diagnostics.py`). `ruff` and
  `python -m scripts.repo_policy` clean.

## Honest remaining scope

- The Tier-1 logit-space geometry content (parent legal-space good/bad mass and
  margins, objective overlap/contradiction, dominance under raw vs unit-normalized
  scaling, held-out signature distances) and the Tier-2 adapter-subspace gradient
  interface that *run on* `run_bounded_stages`.
- Wiring `admit_semantic_corpus` into `train_local_from_paths` (and a
  `--require-admission` / `--materializer` CLI flag) so the semantic trainer refuses
  a non-admitted corpus before the first optimizer step — mirroring the existing
  `constraint_shadow` refusal.
- These are the next commits. This iteration adds no token/component special cases,
  runs no training, and makes no model-quality claim.
