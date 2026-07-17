# G1 — Track A encoded as an autoresearch matrix (2026-07-17)

Wiring for Track G1 (Linear SLM-46): the program's experiments route through
the existing hypothesizer machinery, not a parallel loop. Code:
[`src/slm_training/autoresearch/program_matrices.py`](../../src/slm_training/autoresearch/program_matrices.py).

## What was built

- **Typed knobs for the emptiness levers**: A3 coverage-energy remasking and
  A4 minimum-content decode contracts were real `ModelBuildConfig`/model
  levers (`remask_policy="coverage"`, `decode_min_content`) but had no typed
  `ExperimentKnobs` field, no allowlist entry, and no `evaluate_model` CLI
  flag — so the autoresearch engine could not route them. This change adds
  `remask_policy` and `decode_min_content` to `ExperimentKnobs` +
  `DEFAULT_ALLOWED_KNOBS`, a `compile_commands` emit branch (decode-time, so
  they ride the eval stage), and the matching `--remask-policy` /
  `--decode-min-content` CLI flags.
- **`track_a_matrix()`**: the valid-but-empty-wall attack as five authentic
  `ExperimentSpec`s with distinct knob signatures — A3 coverage remask, A4
  auto content floor, A4 fixed content floor, A3+A4 combined, A5 lattice
  search — each grounded in a real `EvidenceSnapshot` whose items are the
  committed E248 probe, E250 min-content, E251 coverage-remask, and lattice
  campaign iter docs (real sha256/size computed at build time), with a
  `research`-role citation to `research-lineage.md`.

## Verification (end-to-end through the real engine)

`tests/test_autoresearch/test_program_matrices.py`:
- `validate_hypothesis_matrix` accepts the matrix and each candidate passes
  `validate_experiment` (evidence-role coverage, citation grounding,
  five-candidate gate, one regime-transition novelty, distinct signatures).
- Each candidate `compile_commands` to bounded CPU argv arrays (never a shell
  string), and the emptiness knobs (`--remask-policy`, `--decode-min-content`,
  `--compiler-search-mode`) actually reach the eval command.
- `create_hypothesis_feedback` closes the loop with a well-formed
  `feedback-<16hex>` id a successor matrix must acknowledge.
- Full autoresearch suite green (49 tests), including the frozen
  hypothesizer-benchmark tests — **G1 does not touch `hypothesizer_cases.json`
  or `hypothesizer_eval.py`**.

## Honesty and limits

- This encodes Track A and closes the routing gap for its levers; it does
  **not** run the experiments (no training, no checkpoint, nothing promoted).
  The matrix is a reviewable plan that `compile_commands` turns into bounded
  commands — executing them is the campaign operator's step.
- **A2 (ASAp reweighting) is deliberately not encoded as a routable knob**
  because no model-side ASAp lever exists yet; fabricating one would be
  dishonest. It stays a future matrix row, noted in the selection rationale.
  A1 (the emptiness probe) is the grounding diagnostic, already run as E248.
- The other program tracks (B/C/D/E/F/G) are not yet encoded as matrices;
  `program_matrices.py` is the pattern to extend, one track at a time.
