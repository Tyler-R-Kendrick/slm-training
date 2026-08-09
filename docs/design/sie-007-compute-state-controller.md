# SIE-007 — ComputeStateFeaturesV1 and the replay-only symbolic controller

Issue: [SLM-473](https://linear.app/quickdeploy-ai/issue/SLM-473) ·
Campaign identity: `EXP-SR-5`, registered by SIE-001 in
`src/slm_training/harnesses/experiments/exp_sr_catalogue.py` — this issue
enables nothing and executes nothing.
Code: `src/slm_training/harnesses/experiments/compute_state_controller.py`.

## What this is

A substrate for *studying* value-of-compute rules — "given the decision state,
is the next unit of compute better spent on the deterministic ranker, a neural
forward, bounded search, or repair?" — without letting a learned formula touch
the product. It runs only over frozen `DecodeStatsRecordV1` records; there is no
registration into any serving path.

## Features (`ComputeStateFeaturesV1`)

A versioned, *total* vector over the frozen `FEATURE_ORDER` vocabulary; index
`i` binds `v<i>` in every rule, so a rule's meaning is pinned to that order.
Appending a feature is a new schema version, never an in-place reorder.

**UNKNOWN is not zero.** A measurement the record does not carry is `None`, and
a rule that references it abstains. Encoding "unmeasured" as `0.0` would read to
a scoring rule as "measured and empty" — the single most likely way this
substrate would quietly start inventing decisions from missing telemetry.

## Rule language

The rule language is the symbolic-regression IR (`dsl/symbolic_expr_ir`),
**reused unchanged**. Its token grammar admits only `(`, `)`, allowlisted
operators, and `v<i>`/`c<i>` symbol references, so an arbitrary identifier,
string, or free numeric literal cannot even lex — `eval`, `import`, calls, and
reflection are unreachable by construction rather than by denylist. Depth/node
budgets tighten to 8/48 and are enforced at construction, so an over-budget or
malformed rule fails before any replay.

A rule carries one arithmetic scoring expression per candidate action; the
recommendation is the argmax. Comparisons and booleans are expressed as score
differences plus the abstain margin instead of a second operator set, which
keeps exactly one expression language in the repo to fuzz and certify.

## Authority limits

| Guarantee | Mechanism |
| --- | --- |
| Cannot change legality | `controller_can_change_legality()` is a hard `False`; the API returns an action recommendation and nothing else |
| Exact singleton bypass (I2) | A *complete* legal domain of size 1 commits with `expressions_evaluated == 0`; an incomplete/unknown domain of apparent size 1 is not a singleton and does not bypass |
| No guessing | Any UNKNOWN feature referenced by the rule, or any numeric-domain violation while scoring, yields `abstain` |
| Bounded cost | Depth ≤ 8, nodes ≤ 48, checked at construction |
| Not enabled in production | Nothing here is wired into a decode path by this issue |

## Replay and regret

`replay_rule` scores a candidate against frozen records. The oracle action is
derived only from counters the record already carries (retries → repair,
multiple attempts → bounded search, forwards → neural forward, otherwise the
deterministic ranker), never from a label supplied by the rule under test.
Abstention is scored as coverage loss rather than as a wrong answer, so a rule
cannot buy regret `0.0` by abstaining everywhere — the abstention count is
reported beside the regret and both are needed to read a result.

## Validation

`tests/test_harnesses/experiments/test_compute_state_controller.py`: malicious
and malformed rule rejection (including `__import__`/`eval`/free-literal
forms), budget bounds, unbound coefficients, UNKNOWN-feature abstention,
singleton zero-work, incomplete-domain non-bypass, abstain-margin tie
suppression, frozen-trace determinism, oracle-action fixtures, and
schema round-trip plus foreign-version rejection (SGS-010).
