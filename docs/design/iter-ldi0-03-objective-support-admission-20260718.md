# LDI0-03 — objective-support admission and bounded diagnostics

Date: 2026-07-18
Status: **objective-support admission gate, bounded-diagnostic runner, the Tier-1
objective-geometry pass, the Tier-2 refuse-full-parameter interface, and a V1
objective-signature trainer-entry refusal all landed with tests. The V2
`admit_semantic_corpus` CLI wiring (`--require-admission` / `--materializer`) remains
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
  `Deadline`, `run_bounded_stages`, `not_authorized_report`, `write_diagnostic_report`,
  and — on that bounded runner — `tier1_objective_geometry` and
  `tier2_subspace_gradients`.
- `tier1_objective_geometry` reads already-materialized objective views (`getattr` on
  `good_action_ids` / `bad_action_ids`, so it stays decoupled from
  `decision_events_v2`) and reports, per corpus, the count of states with an
  objective **contradiction** (an action scored good by one view and bad by another —
  the logit-space shadow of the E284 conflict) and the mean pairwise Jaccard overlap
  of the per-state good-action sets. It computes no gradient and trains nothing.
- `tier2_subspace_gradients` refuses an empty / `None` (full-parameter) request as
  `not_authorized` rather than replaying the invalid E285 full-parameter profile; for
  an explicit adapter subset it records a bounded plan only, with the gradient
  computation deferred to a model stage this module never runs.
- A V1 objective-signature pre-flight refusal now guards `train_local_from_paths`
  (`require_objective_support=True`): `objective_signature_support` (keyed on the
  good+bad `decision_signature`) fails the entry before the checkpoint is loaded when
  a held-out objective signature lacks train support — the E284 blocker.
- Tests: the E284 pattern (passes state support, fails objective support),
  constraint-shadow and materializer-mismatch refusals
  (`tests/test_harnesses/preference/test_decision_events_v2.py`), the deterministic
  fake-clock deadline / no-result-on-expiry / not-authorized cases plus the Tier-1
  contradiction/agreement and Tier-2 refuse/plan cases
  (`tests/test_harnesses/preference/test_decision_diagnostics.py`), and the trainer
  objective-support refusal (`tests/test_harnesses/preference/test_local_decisions.py`).
  `ruff` and `python -m scripts.repo_policy` clean.

## Landed in this follow-on (V2 admission wiring)

- `src/slm_training/harnesses/preference/local_decisions.py`: added V2
  `objective_view_signature`, `objective_view_support`, and `admit_semantic_corpus`
  (mirroring the `decision_events_v2.py` gate but operating on the
  `local_decisions.py` V2 types used by the trainer). `ObjectiveView` now carries an
  explicit `trainable` flag; `materialize_constraint_shadow` sets it to `False`.
  `admit_semantic_corpus` verifies materializer ID **and** optional config hash.
- `src/slm_training/harnesses/preference/local_train.py`: `train_local_from_paths`
  gained `require_admission`, `materializer_id`, and `materializer_config_hash`. When
  loading V2 events, it materializes each event with the requested materializer and
  runs `admit_semantic_corpus` before loading the checkpoint, so a mismatched
  materializer, config hash, or non-trainable view is refused before any optimizer
  step.
- `scripts/train_preference.py`: the `train-local` subcommand exposes
  `--require-admission` / `--no-require-admission`, `--materializer`, and
  `--materializer-config-hash`, passing them through to `train_local_from_paths`.
- Tests:
  - `tests/test_harnesses/preference/test_local_decisions.py`: V2 objective-support
    E284 pattern, materializer-ID mismatch, config-hash mismatch, constraint-shadow
    non-trainable refusal, and passing admission.
  - `tests/test_harnesses/preference/test_local_train.py`: trainer refuses V2
    materializer/config-hash mismatch and constraint shadows, and admits a V2 corpus
    end-to-end.

## Honest remaining scope

- The Tier-2 gradient computation itself (the deferred model stage) and the richer
  Tier-1 logit-space content (parent legal-space good/bad mass and margins, dominance
  under raw vs unit-normalized scaling, held-out signature distances) that build on the
  bounded runner landed here.
- This iteration adds no token/component special cases, runs no full training campaign,
  and makes no model-quality claim.
