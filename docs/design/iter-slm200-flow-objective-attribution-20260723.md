# SLM-200 (VFA1-02): objective × state-weighting attribution

**Status:** measured non-publishable fixture screen; confirmation not touched.
**Verdict:** `no_conclusion_underpowered_fixture`.

## Frozen matrix and parity

- Arms: A0=unavailable, A1=measured_fixture, A2=measured_fixture, A3=measured_fixture, A4=measured_fixture, A5=measured_fixture, A6=measured_fixture, A7=measured_fixture, A8=measured_fixture, A9=measured_exact_fixture.
- Production parameter counts: `[4772]`.
- Identical train-row order: `True`.
- Identical decoder: `True`.
- Seeds/steps: `[0, 1, 2, 3, 4]` / `8`.

## Measured fixture results

- Development-selected simpler control: `A1`.
- A7/control target-exact rate: `0.000` / `0.000`.
- Paired descriptive delta: `+0.000`.
- A9 max rate error: `0.000311375`.
- A9 analytic endpoint/event-count TV: `0.000000000` / `0.000000000`.

These numbers are descriptive fixture wiring only. The SLM-196 corpus
contains two independent targets and is explicitly non-publishable; A0
has no hash-pinned identical-state input. SLM-183 measured only 0.11
power at the preregistered 0.08 MDE. Therefore no equivalence, weighting,
hazard, or flow-transport causal attribution is licensed.

## Confirmation and disposition

- Confirmation status: `not_touched`; touch ledger is empty.
- Checkpoint: none.
- Claim class: wiring.
- Decision: no conclusion; preserve all objectives as experimental and
  require a publishable bridge corpus, complete A0 input, frozen full
  confirmation suite, and powered checkpoints before a single touch.

AgentV: `{'total': 5, 'passed': 5, 'failed': 0, 'executionErrors': 0, 'durationMs': 20, 'meanScore': 1}`.
