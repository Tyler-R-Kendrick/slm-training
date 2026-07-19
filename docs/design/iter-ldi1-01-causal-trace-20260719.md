# LDI1-01 (SLM-119): exact causal decision-state trace and replay

Date: 2026-07-19 · Track: LDI · Linear: SLM-119 · Blocked-by: none

## What shipped

A torch-free fixture that proves the causal decision-state trace/replay contract
already implemented in `slm_training.models.causal_trace` and
`slm_training.models.causal_lm_openui`.  The fixture uses synthetic logits and a
hand grammar to show:

- **Exact prefix token ids as state authority**: every `DecisionStateV2`
  stores the full prefix (`context_ids`) that produced it.
- **Raw argmax vs constrained selection**: each step records the raw model
  preference and the grammar-legal selection.
- **Constraint-shadow detection**: when the raw argmax is illegal and the
  legal selection differs, the observation is flagged as a shadow.
- **Forced-action replay from the stored prefix**: a stored state can replay
  with an alternative legal action and continue deterministically through the
  same forward seam.
- **TraceStore persistence + fail-closed identity loading**: traces are
  written through `TraceStore`, a manifest is emitted, and loading refuses to
  return states unless checkpoint / tokenizer SHAs match the stored identity.

Artifacts:

- `scripts/run_ldi1_01_causal_trace_fixture.py` — fixture runner (no model,
  no tokenizer, no torch).
- `tests/test_harnesses/experiments/test_ldi1_01_causal_trace_fixture.py` —
  six regression tests covering shadows, exact-prefix authority, forced-action
  replay, TraceStore round-trip, and replay determinism.
- `docs/design/iter-ldi1-01-causal-trace-20260719.json` — full JSON evidence
  mirror (summary, observations, decision rows, replays, manifest, version
  stamp).

No source code in `src/slm_training/models/causal_trace.py` or
`causal_lm_openui.py` was changed; this iteration wires evidence around the
existing implementation.

## Fixture design

Vocabulary (6 tokens): `0=EOS`, `1=root`, `2=child_A`, `3=child_B`,
`4=illegal_shadow`, `5=filler`.

Grammar legal sets:

| Suffix length after root | Legal tokens |
| ---: | --- |
| 0 | `2`, `3` (token `4` is raw-argmax but illegal) |
| 1 | `5` (forced continuation) |
| 2 | `0`, `5` |

Logits at the root prefix make token `4` the raw argmax while the legal set is
`{2, 3}`, producing a clean constraint shadow.  Subsequent decisions include one
forced step (legal set singleton) and one free EOS/continue step.

## Measured results

JSON mirror:
[iter-ldi1-01-causal-trace-20260719.json](iter-ldi1-01-causal-trace-20260719.json).

| Quantity | Value |
| --- | ---: |
| Observations | 3 |
| Constraint shadows | 2 |
| Forced-action replays | 2 |
| Replay errors | 0 |
| Loaded states (fail-closed) | 3 |
| Manifest state count | 3 |

Replay examples:

| State | Forced action | Replay program |
| --- | ---: | --- |
| first_decision | `3` (child_B) | `3 5 0` |
| continuation | `5` (filler) | `2 5 5 0` |

Both replays start from the stored exact prefix, append the forced action, and
continue through the same synthetic logits/grammar seam until EOS.

## Test results

```text
tests/test_harnesses/experiments/test_ldi1_01_causal_trace_fixture.py::test_fixture_produces_observations_and_replays PASSED
tests/test_harnesses/experiments/test_ldi1_01_causal_trace_fixture.py::test_first_observation_is_constraint_shadow PASSED
tests/test_harnesses/experiments/test_ldi1_01_causal_trace_fixture.py::test_decision_state_carries_exact_prefix PASSED
tests/test_harnesses/experiments/test_ldi1_01_causal_trace_fixture.py::test_forced_action_replay_uses_stored_prefix PASSED
tests/test_harnesses/experiments/test_ldi1_01_causal_trace_fixture.py::test_manifest_matches_loaded_states PASSED
tests/test_harnesses/experiments/test_ldi1_01_causal_trace_fixture.py::test_capture_raw_steps_replay_determinism PASSED
```

## Run metadata

| Field | Value |
| --- | --- |
| device | CPU (torch-free) |
| steps | n/a — synthetic forward seam |
| backend | `capture_raw_steps` + `TraceStore` |
| n | 3 observations, 2 replays |
| honesty | fixture / wiring evidence only; no model, no quality claim |
| gate | none claimed |

## Tradeoffs and caveats

- **Fixture only**: synthetic logits replace a real model.  This validates the
  trace/replay data contract, not predictive quality.
- **No checkpoint created or promoted**: `docs/MODEL_CARD.md` and README summary
  are intentionally unchanged.
- **No ship-gate movement**: this is wiring evidence; ship-grade claims still
  require full `rico_held` / HF-context runs and standard evals.
- **Existing code untouched**: `causal_trace.py` and `causal_lm_openui.py`
  already contained the trace/replay machinery; the new script and tests pin
  the contract against regression.
