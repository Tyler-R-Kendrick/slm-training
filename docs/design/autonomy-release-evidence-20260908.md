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

## Evidence limits

The current environment does not provide the complete project test dependency
set (`pytest`, `pydantic`, `torch`, and `lark` are unavailable in the candidate
interpreter), so full pytest, model training, real AgentV evaluation, and live
noninteractive source repair were not executed here. Compile-time checks and
the connector-reported PR checks are not substitutes for those obligations.
The absence of live repair authority is a scoped capability limitation, not a
successful repair result. Existing user worktrees and training state remain
untouched.

## Required next evidence

The remaining owners must publish focused fixes and rerun the frozen candidate
through the complete local verification manifest before Linear A18 or the
parent issue is marked complete. In particular, the final report must bind the
exact source tree, selected test nodes/options, environment identity, repair
receipts, data/evaluation bundle identities, and any unavailable capabilities.
