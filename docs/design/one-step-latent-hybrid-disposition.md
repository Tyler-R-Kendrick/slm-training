# AP-035: one-step MeanFlow-style prior plus discrete refinement (SLM-336)

**Status:** blocked / not started.
**Claim class:** `wiring`.
**Honest verdict:** `no_go_defer` -- AP-035's own admission gate is not cleared.
**Depends on:** SLM-335 (AP-034, Done, admission-denied) and SLM-330 (AP-031, Done, diagnostic-only).

AP-035's "Implementation" contract is explicit: *"Start only after AP-034
clears its oracle-support gate."* Its "Acceptance" section repeats the same
condition as a hard requirement: *"AP-034 is within 0.10 of oracle strict
semantics and AP-031 proves causal latent use."*

## Audit (2026-07-26)

* AP-034's own disposition
  (`docs/design/ap034-conditional-latent-prior-disposition.md`) records an
  **admission-denial**, not a cleared gate: *"Conditional-prior integration
  is unavailable; no model, connector, or training path was added."* It names
  three missing successor artifacts:
  1. an immutable, train-only prompt-to-`SemanticPlanV1` manifest with
     leakage audits,
  2. a constrained, replay-verified latent-to-program decode bridge, and
  3. a matched oracle/nearest-encoder/random/conditional-prior protocol over
     that manifest.
  None of the three exist in this repository as of this audit.
* AP-031's causal-use audit (`docs/design/program-latent-causal-use.md`)
  proved causal necessity for the `continuous` codec directionally (CI lower
  bound > 0) but was **not CI-significant** for `lfq`/`fsq_typed` at `n=7`,
  and flagged `effective_rank≈1.00-1.06` as a near-collapse boundary. It is
  default-off and diagnostic, not a ship-eligible result.

Building the one-step MeanFlow-style prior AP-035 asks for would require
conditioning on the same prompt-to-plan interface AP-034 explicitly declined
to fabricate. Doing so here -- by inventing an unaudited condition, reusing
the factor-reconstruction MSE proxy as a semantic-quality metric, or
declaring the AP-034 gate cleared by fiat -- would be exactly the shortcut
AP-034's disposition refused to admit.

## Disposition

**AP-035 does not start.** No prior, refinement loop, Pareto-frontier
comparison, or bitrate/latency claim is made by this change. This is not a
rejection of the AP-035 goal (a candidate that improves the quality/p95-latency
frontier); it is a scoping decision that the goal is unreachable until AP-034
delivers its three named successor artifacts.

`src/slm_training/harnesses/experiments/one_step_latent_hybrid.py` encodes
this precondition as `require_ap034_gate_cleared`, so that any future AP-035
implementation attempt fails fast and names the exact missing artifacts
instead of silently proceeding on an unmet gate. `CURRENT_AP034_ARTIFACTS`
records today's state (all three artifacts absent);
`tests/test_harnesses/experiments/test_one_step_latent_hybrid.py` pins the
current blocked state as a regression test and proves the gate clears once
all three artifacts are marked present.

## Next steps

A follow-on issue against AP-034's three named artifacts (prompt-to-plan
manifest, verified latent-to-program decoder, matched prior protocol) must
land, with its own honest evidence, before AP-035 can be attempted again.
