# LDI0-02 (SLM-116): DecisionEventV2 — state + action-verdict tables

**Date:** 2026-07-18
**Scope:** wiring evidence for the V2 decision-event contract. **No trainer was
changed, no rollout policy or adapter was added, no model ran. These fixture rows
establish wiring only and make no model-quality claim.**

## Why V2

`DecisionEventV1` (`harnesses/preference/local_decisions.py`) freezes one
`good_token_ids` / `bad_token_ids` partition *into* the event. That conflates the
exact model **state** with a sampled **action partition**, which is the E284
blocker: grammar-state support can be complete while the sampled partition used by
the objective is unsupported or incompatible. V2 separates three concerns
(`harnesses/preference/decision_events_v2.py`):

- **`DecisionStateV2`** — `state_id` is a canonical hash of the exact model state
  and immutable runtime identities **only** (architecture, context/canvas ids,
  decision position, generation step, legal-action set, grammar/policy/tokenizer/
  decode/verifier identities). It never includes sampled labels, rollout outcomes,
  file position, or candidate order, so it is identical when action evidence is
  reordered or augmented.
- **`ActionOutcomeV2`** — append-only, content-deduplicated evidence per
  `(state_id, action_id)`: legality, rollout seeds + one output hash per seed, the
  complete **ordered G0–G12 verifier vector**, named judge evidence, and reward
  vectors. A scalar `mean_value` may be derived but never replaces the raw vectors.
- **`ObjectiveView`** — pure, versioned materializers turn one state/action table
  into `good` / `bad` / `ambiguous` / `unobserved` partitions. Five are provided:
  `pareto_pass_fail`, `thresholded_value`, `single_best_worst`, `set_valued`, and
  the non-semantic `constraint_shadow` legality diagnostic (which semantic trainers
  are guarded from consuming).

V1 stays fully readable; `migrate_v1_event` performs a one-way migration that marks
the derived action evidence `migrated_incomplete` and **never fabricates** rollout
seeds or verifier vectors the V1 record did not have.

## Fixture: one state, two materializers, same evidence

One exact `DecisionStateV2` (`state_id 827ff5a5…`), three independently-replayed
legal actions (`3`, `4`, `5`) each carrying a full ordered G0–G12 vector. Two
materializers over the **same** evidence:

| Materializer | good | bad | ambiguous | semantic | view fingerprint |
| --- | --- | --- | --- | --- | --- |
| `set_valued_v1` | `(3, 4)` | `(5,)` | `()` | yes | `71ac4acf985b` |
| `thresholded_value_v1` (θ=0.5) | `(3,)` | `(4, 5)` | `()` | yes | `e9697e4827f5` |

The **state fingerprint is stable** across both materializations (`827ff5a5…`)
while the **objective fingerprint changes** (`71ac4acf…` → `e9697e48…`) — exactly
the state-vs-objective separation V2 exists to provide. The manifest fingerprints
the three layers independently:

| Layer | fingerprint (16 hex) |
| --- | --- |
| states | `b3942b4bf3327ea7` |
| action evidence | `519571b184e23aef` |
| objective views | `fd7f7b40d6fba41e` |

## Invariants proven by the fixture tests

`tests/test_harnesses/preference/test_decision_events_v2.py` (14 tests):

- `state_id` is independent of declared legal-action order and of action evidence.
- Append-only merge deduplicates by content identity, order-independent.
- A wrong explicit `state_id` and a single state straddling train/held-out are
  rejected (fail closed).
- Missing runtime identity (e.g. verifier bundle) is rejected.
- A legal outcome outside the declared legal set is rejected.
- Materializer identity reflects its config hash; the constraint-shadow view is
  non-semantic and blocked from semantic training.
- Set-valued materialization yields multi-good/multi-bad with unobserved (legal but
  un-rolled-out) actions surfaced.
- V1 semantic + constraint-shadow migration is non-mutating and marks partial
  evidence incomplete.
- Causal states require `context_ids` (no retokenization) and TwoTower states
  require `canvas_ids`; both round-trip and reproduce their replay inputs.
- Manifest fingerprints are order-independent but evidence-sensitive, and the three
  layers are fingerprinted separately.
- Full ordered G0–G12 evidence survives a write/read round trip; writes are atomic
  and duplicate-safe.

## Honesty / non-goals

No trainer change beyond the compatibility contract, no new counterfactual rollout
policy, no adapter, no support synthesis from held-out prompts. V1 corpora are
untouched and still load/fingerprint unchanged. This issue delivers the schema and
its guards only — **no model-quality claim** is made or implied.

## Verification

```
python -m pytest tests/test_harnesses/preference/test_decision_events_v2.py \
    tests/test_harnesses/preference/test_local_decisions.py -q          # 25 passed
python -m pytest tests/test_harnesses/preference -k "counterfactual or decision" -q  # 41 passed
python -m ruff check src/slm_training/harnesses/preference tests/test_harnesses/preference  # clean
python -m scripts.repo_policy                                            # ok
git diff --check                                                         # clean
```
