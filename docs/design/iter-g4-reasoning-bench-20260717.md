# G4 — sketch-vs-direct reasoning bench fixture run (2026-07-17)

Fixture-grade wiring run for Track G4 (Linear SLM-36). Machine-readable
evidence:
[reasoning-bench-results-iter-g4-20260717.json](reasoning-bench-results-iter-g4-20260717.json).
Design: [reasoning-sketch-harness.md](reasoning-sketch-harness.md). Code:
[`src/slm_training/harnesses/reasoning/bench.py`](../../src/slm_training/harnesses/reasoning/bench.py)
+ [`src/slm_training/dsl/packs/arith_sketch.py`](../../src/slm_training/dsl/packs/arith_sketch.py).

## Recipe

`g4_fixture_20260717`: 96 template-generated word-problem records, 24
held-out test problems (prompt-disjoint), 120 CPU steps per arm, d_model 64,
seed 0, compositional corpus-derived tokenizer, unconstrained parallel
MaskGIT decode in both arms, one deterministic numeric oracle for both.

## Result (wiring evidence only)

| Arm | answer accuracy | trace/output validity |
| --- | --- | --- |
| sketch (program trace, executed) | 0.0 | 0.0 |
| direct (bare numeric answer) | 0.0 | 1.0 (parses as number) |

Both arms sit at zero accuracy at this budget — the run proves the loop
(pack-generated corpus → matched training → decode → single-oracle
fail-closed scoring → persisted summary), not a reasoning capability. No
gate weakened; nothing promoted; no checkpoint kept.

## Failure modes (instructive, recorded for the follow-up)

- **Sketch arm** emits near-grammatical programs (`x = 9 * 6`, correct
  statement shape) but with *forward references* (`x = y * 6` before `y`
  is bound) and *missing `root` bindings* — every trace rejected by the
  fail-closed oracle. These are exactly the scope/completion errors that
  grammar+scope-constrained decode eliminates by construction (cf. X22,
  where all-valid search produced parse rate 1.0): the strongest possible
  motivation for the stated follow-up of wiring the incremental engine
  (which already accepts `arith_sketch.lark`) into non-OpenUI decode.
- **Direct arm** collapses to a constant answer ("33" for every problem) —
  the classic no-trace failure the PAL/PoT line predicts.

## Verification

- `tests/test_harnesses/reasoning/test_reasoning_bench.py`: the evaluator
  is the single oracle (valid + all five invalid shapes), generator gold
  matches the oracle and is deterministic, scoring is fail-closed
  (invalid trace ≠ valid-but-wrong ≠ correct), and a tiny end-to-end bench
  (both arms, 2 steps) runs through the full loop.
- Pack contract invariants now parametrize over all three packs
  (`openui`, `toy-layout`, `arith-sketch`); grammar-backend suite green
  with the fifth backend registered; `repo_policy`, `ruff`,
  `git diff --check` clean.

## Honesty and limits

- Fixture scale: 6 templates, tiny model, 120 CPU steps, one seed. The
  PAL/PoT-analog comparison (does an executed trace beat a direct answer?)
  is **unanswered** here — both arms are at zero, so the bench currently
  measures the wiring, not the hypothesis. The comparison becomes
  meaningful once (a) constrained decode lands for non-OpenUI DSLs or
  (b) the budget is large enough for either arm to leave zero.
- A frozen-large-LLM PAL baseline is deliberately out of scope (stated in
  the design doc); the honest baseline at this scale is the matched
  direct-answer arm.
