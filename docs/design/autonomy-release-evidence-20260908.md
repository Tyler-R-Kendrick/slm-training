# Autonomy release evidence — 2026-09-08

This record covers the merged harness slices currently reachable from `main`.
It is an engineering evidence record, not a model-quality or champion claim.

## Source and delivery

- Base work was integrated through GitHub pull requests #1737, #1738, #1739,
  #1740, and #1741; the current autonomy slice is `a809e818`.
- Continuous training remains stopped. No champion was promoted, no serving
  pointer was changed, and no HF or paid service write was performed.
- The authoritative Linear parent remains open until the remaining A01–A18
  acceptance obligations have current producer-to-consumer evidence.

## Verified implementation slices

- Logical continuation grants are bound to the actual cursor and direct
  execution path; driver-level per-stage accounting no longer prematurely
  parks a multi-stage logical grant.
- Exit-10/yielded outcomes reconcile through the same durable event path as
  terminal outcomes, preserving the command cursor's attempt accounting.
- Checkpoint bundle publication keeps the fencing scope active for the whole
  publication body, so stale owners cannot write after the guard is entered.
- Cross-owner adversarial pairing tests reject equal-sized arms with disjoint
  record identities and mutation-test the identity-keyed pairing contract.
- `python -m scripts.verify_agent_surfaces` passes with 19 obligations and 17
  configured surfaces.
- `python -m scripts.verify_version_stamps --check` passes on the candidate
  tree.

Subsequent focused slices merged after the initial record:

- `3f35459c` — cached evaluation publication boundary and cache-hit regressions.
- `df8665bb` — checkpoint bundle, exposure, and trial-continuation tests.
- `2cae246f` — owned-process activity restart and continuation tests.
- `e764d583` — governed repair dispatch and independent-verifier tests.
- `08eccb33` — diagnostic/confirmation evidence-boundary adversarial tests.
- `a5d41135` — complete merge-gate collection, isolation, runtime, and scheduling tests.
- `9ced4cc8` — composed runtime-safety adversary suite.

## Evidence limits

The candidate verification environment was restored with the repository's pinned
Python environment and ran focused pytest, compile, version-stamp, and agent
surface checks. The candidate workspace still lacks the OpenUI bridge
`node_modules` dependency, so the three bridge-dependent learning/data
tests were not executable there; they are recorded as capability-limited rather
than passed. The primary checkout has the bridge dependency, but it was not
used to claim candidate-tree evidence. Hypothesis is also unavailable in the
restored environment, so property/stateful evidence requiring it remains
unexecuted. Full model training, real AgentV evaluation, and live noninteractive
source repair were not executed. These limitations are scoped and do not
fabricate success.

## Bounded operational probe

The canonical command

`PYTHONPATH=src:. timeout -s INT -k 10 170 python -m scripts.verify_autonomy --root outputs/runs/autonomy-verify-20260909 --batch-size 20`

returned exit `10` with typed status `waiting_capability` and capability
`rootless_isolated_verifier`. The host probe reported that bubblewrap
could not create its required NETLINK_ROUTE socket (`Operation not permitted`).
No unisolated fallback was used, no fixture attempts were counted, and no
success or repair evidence was fabricated. This is a scoped capability result;
it does not stop unrelated source verification.

## Current focused verification

- Search contracts: `43 passed` in `15.86s`.
- Runtime/repair/operations focused subset: `110 passed, 5 skipped, 4 deselected`
  in `39.46s`.
- Checkpoint bundle/exposure/continuation/grant subset: `63 passed` across the
  published focused runs.
- Agent-surface and version-stamp checks pass on the candidate tree.
- Learning/data diagnostic subset: `25 passed, 3 capability-limited` because
  the candidate workspace bridge dependency is absent.
- The 3-minute command cap was preserved; no training loop was started and no
  champion or serving pointer was changed.
