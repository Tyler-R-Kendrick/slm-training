# SLM-199 (VFA1-01): discrete legal-edit rate matching

**Status:** measured fixture wiring; no ship or flow-win claim.
**Verdict:** `adapted_time_conditioned_edit_policy_fixture_only`.

## Exact closed CTMC oracle

- Closed choice graph: `True` (6 states, 5 transitions).
- Illegal edge-rate sum: `0.000000`.
- Exact-rate fit MSE/max error: `0.00000063` / `0.00031137`.
- Exact/predicted/empirical terminal mass: `1.000000` / `1.000000` / `1.000000`.
- Analytic/empirical endpoint TV at matched horizon `50.0`: `0.000000` / `0.000000`.
- Exact/empirical event-count distributions: `{'5': 1.0}` / `{'5': 1.0}` (TV `0.000000`).

This acceptance oracle deliberately uses the closed acyclic choice fixture.
The earlier SLM-190 toy/canonical graphs were bounded/inconclusive and
are not re-labelled as passing evidence.

## OpenUI production adapter

- Fidelity: `adapted_path_approximation`.
- Rows/train rows: `4` / `2`.
- UNKNOWN supervised as negative: `False`.
- UNKNOWN rate mass after fit: `0.011323`.
- Parser-verified outputs: `2/2`.
- Fixture target-exact rate: `0.000` (descriptive only).

The adapter re-enumerates exact live candidates after every one-edit
commit, keeps UNKNOWN candidates live, uses positive rates and a frozen
fixed-K termination policy, and returns verified syntax or explicit
UNKNOWN/abstention. Bridge holding times are unobserved, so unit-hazard
targets are labelled adapted path approximations, not faithful DFM.
UNKNOWN receives no direct edge or hazard regression label, though it
remains in normalization and therefore receives disclosed indirect
set-ranking pressure.

## Recipe and disposition

- Device/backend: `cpu` / `torch+numpy exact closed fixture`.
- Train steps/seeds: `8` / `[0, 1, 2, 3, 4]`.
- Exact samples: `256`.
- Wall: `1.115s` (cap `3.0m`).
- AgentV: `{'total': 5, 'passed': 5, 'failed': 0, 'executionErrors': 0, 'durationMs': 21, 'meanScore': 1}`.

Flow is default-off, no checkpoint was written, and existing direct
training/decode paths are unchanged. VFA1-02 owns powered confirmation;
this fixture does not establish a held-out flow improvement.
