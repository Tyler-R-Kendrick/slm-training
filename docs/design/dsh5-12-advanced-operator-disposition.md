# DSH5-12 advanced operator disposition (SLM-420)

SLM-420 (stable alias `DSH5-12`) is the capstone issue closing "DSH5 — Bulk
Operators, Transactions & Control Plane" (milestones M1–M4). It is a
synthesis/measurement issue, not an implementation issue: it publishes
`AdvancedOperatorDispositionV1`
(`src/slm_training/evals/advanced_operator_disposition.py`), an independent,
evidence-bound verdict for each of eleven advanced-operator abstractions, so
that the real, tested M1 work (exact selectors, atomic bulk actions, atomic
transaction contracts/execution, N-step sequence merge, compiler-owned
control-plane execution) can never be read as evidence for the abstractions
that were never measured, never trained, or never even attempted (learned
transaction/router/control policies, event memory's held-out benefit,
parameterized templates). The generated artifact is
[`dsh5-12-advanced-operator-disposition-20260727-local/report.json`](dsh5-12-advanced-operator-disposition-20260727-local/report.json)
/ [`summary.md`](dsh5-12-advanced-operator-disposition-20260727-local/summary.md),
published by `scripts/publish_dsh5_12_advanced_operator_disposition.py` and
re-checked by `scripts/validate_advanced_operator_disposition.py`.

## What DSH5 actually contains

Eleven prior issues (`SLM-409`..`SLM-419`) plus the required prerequisite
`SLM-408` (DSH3-33). Reading each issue's own committed design doc (never
re-derived speculatively) gives:

| Sub-issue | Alias | What it actually built | Own stated disposition |
| --- | --- | --- | --- |
| SLM-409 | DSH5-01 | `SelectorRefV1`/`SelectorDescriptorV1` contracts, finite selector domains | "compiler contract/unit fixtures... no train/eval/benchmark/matrix/checkpoint/model-card/ship-gate/model-quality claim" |
| SLM-410 | DSH5-02 | `openui.map_set_property` atomic bulk operator | same disclaimer, plus a proven atomicity argument |
| SLM-411 | DSH5-03 | Bulk-vs-primitive-vs-full-generation crossover preflight | "the report is unavailable, not a crossover result" |
| SLM-412 | DSH5-04 | `OperatorTransactionV1` base-state/dependency/conflict contracts | schema/safety layer only, "ships no execution path" |
| SLM-413 | DSH5-05 | `compose_operator_transaction`/`commit_operator_transaction` | atomic composition proven for the disjoint/declared-commuting subset |
| SLM-414 | DSH5-06 | Bounded-K set-valued transaction-policy decode wiring | "fixture/wiring evidence only... stop rule cannot be evaluated by this change" |
| SLM-415 | DSH5-07 | N-step conservative branch `sequence_merge` | proven for AST_EDIT-only chains; TRANSACTION_COMMIT/mid-sequence FORK unrepresentable |
| SLM-416 | DSH5-08 | Conversation control-action legal set + deterministic baseline | execution/baseline shipped; "no neural scorer is trained or shipped here" |
| SLM-417 | DSH5-09 | Adaptive-router admission preflight | "**unavailable**... no matched group-disjoint all-arm corpus exists" |
| SLM-418 | DSH5-10 | Replay-grounded preference-row extraction (2 of 7 patterns) | "partial slice, in progress... not yet dispositioned" |
| SLM-419 | DSH5-11 | Repository-wide TSA/template-manifest audit | "Integration unavailable; no model or registry change made" |
| SLM-408 | DSH3-33 | Required prerequisite: rebased CAP2 operator-policy disposition | `dsh5.may_start = False`, empty allowed heads/objectives/actions |

Every one of the eleven sub-issue docs already states, in its own words, that
no train/eval/benchmark/matrix/checkpoint/model-card/ship-gate/model-quality
claim follows from it. DSH5-12's job is not to overturn any of those
self-assessments; it is to bind them into one machine-checkable ledger so a
reader cannot accidentally generalize "bulk atomicity is proven" into
"bulk *policy* is beneficial," or "the router preflight ran" into "the router
works."

## Inherited policy inventory (required prerequisite)

Per the issue text, DSH5-12 must "record the exact allowed primitive
policy/head/objective/action inventory inherited from" SLM-408/DSH3-33 rather
than re-derive it. That inventory, read directly from
`docs/design/dsh3-33-rebased-cap2-disposition-20260726-local/report.json`'s
`disposition.dsh5` block, is:

```json
{
  "may_start": false,
  "allowed_heads": [],
  "allowed_objectives": [],
  "allowed_actions": [],
  "reason": "No typed action form improved held-out CAP2 semantics; DSH5 remains closed."
}
```

This is empty. DSH5's compiler-contract work (selectors, bulk atomicity,
transaction contracts/execution, sequence merge, control-plane execution) did
not need this gate — it introduces no learned head, objective, or action; it
is a pure extension of already-shipped compiler machinery (typed reference
tables, the operator registry, `merge.py`'s conflict classifier). The two
places DSH5 *did* touch a learned component — DSH5-06's transaction-policy
ranking (which reuses DSH3-28's already-rejected `TypedOperatorPolicyScorer`
heads purely as a wiring precondition, never claiming a benefit) and DSH5-09's
router preflight (which concluded `unavailable` before any training could
even start) — never claimed the SLM-408 gate was cleared, and this
disposition confirms it was not: `AdvancedOperatorDispositionV1.inherited_policy`
carries the identical empty inventory, unchanged.

## The eleven claims

`AdvancedOperatorDispositionV1` (see the module docstring for the full
rationale) requires each of the following eleven `AdvancedOperatorClaim`
values to carry one headline verdict *and* one explicit verdict for each of
four independent dimensions — `runtime_correctness`,
`learned_semantic_benefit`, `partial_coverage_safety`, `systems_efficiency` —
so that success on one axis structurally cannot satisfy another (a claim's
headline verdict must equal one of its own per-dimension verdicts; it can
never blend axes into an invented fifth state).

| Claim | Headline verdict | Runtime correctness | Learned benefit | Partial-coverage safety | Systems efficiency |
| --- | --- | --- | --- | --- | --- |
| `selector_correctness` | **supported** | supported | unrun_conditional | supported | unavailable |
| `bulk_atomicity` | **supported** | supported | unrun_conditional | supported | unavailable |
| `crossover_work` | **unavailable** | fixture_only | unavailable | unrun_conditional | unavailable |
| `transaction_contracts_execution` | **supported** | supported | unrun_conditional | supported | unavailable |
| `set_valued_selection` | **unrun_conditional** | supported | unrun_conditional | supported | unrun_conditional |
| `sequence_merge` | **supported** | supported | unrun_conditional | supported | unavailable |
| `control_plane_execution_learning` | **supported** | supported | unrun_conditional | supported | unavailable |
| `adaptive_routing` | **unavailable** | supported | unavailable | supported | unavailable |
| `event_memory` | **unrun_conditional** | supported | unrun_conditional | unrun_conditional | unrun_conditional |
| `parameterized_templates` | **unavailable** | unavailable | unavailable | unavailable | unavailable |
| `systems_efficiency` | **unavailable** | unavailable | unavailable | unavailable | unavailable |

Read this table as a matrix, not a single column: `selector_correctness` is
*runtime-correctness* supported, and explicitly *not* a learned-benefit or
efficiency claim (both remain unrun/unavailable on the same row).
`control_plane_execution_learning`'s own name names two things; the table
(and the claim's `dimension_reasons`) makes explicit that only the execution
half is supported and the learning half is unrun. `adaptive_routing`'s
runtime-correctness cell is "supported" because the router *preflight
contract itself* (its own DEFER-on-unavailable logic) passed 2/2 AgentEvals
criteria — that is not router benefit, and the headline verdict is
`unavailable`, matching the decision-relevant `learned_semantic_benefit`
column, not the less central `runtime_correctness` column.

### Why `set_valued_selection` and `event_memory` are `unrun_conditional`, not `negative`

Both delivered a real, tested wiring precondition — DSH5-06's bounded-K
decode never commits a real conflict even under an adversarially corrupted
ranking belief (`test_shuffled_conflict_belief_never_commits_a_real_conflict`);
DSH5-10's row extraction replay-verifies exactly against the legal set for
the two patterns it covers. Neither has a corpus, baseline, or held-out
measurement to answer the actual decision question ("does this help?").
`negative` would overclaim a measured failure that did not happen;
`unrun_conditional` is the accurate state, and per the issue's stop rule this
is "a fully legitimate, expected possible outcome."

### Why `crossover_work`, `adaptive_routing`, `parameterized_templates`, and `systems_efficiency` are `unavailable`, not `unrun_conditional`

Each of these four claims corresponds to a sub-issue that **ran a real
preflight or audit and got a negative-availability answer**, not an
unattempted question: DSH5-03 executed real legal-set/apply/replay/lowering
checks at fanout 1/2/4/8 and concluded no matched five-arm serving rows
exist; DSH5-09 executed a real admission-preflight contract (2/2 AgentEvals
criteria pass) and concluded no matched group-disjoint corpus exists; DSH5-11
executed a real repository-wide tracked-path audit and confirmed zero TSA/
template-manifest artifacts exist anywhere; and no DSH5 abstraction anywhere
has a completed systems/serving-work harness run. `unavailable` marks "we
looked, and the prerequisite does not exist," distinct from
`unrun_conditional`'s "no one has attempted this yet."

## CAP0/CAP1/CAP2 retention

| Capability | Verdict | Reason |
| --- | --- | --- |
| CAP0 | `unrun_conditional` | No learned CAP2/DSH5 candidate reached a retention evaluation; inherited unchanged from SLM-408. |
| CAP1 | `unavailable` | No CAP1-compatible learned candidate or retention artifact exists; inherited unchanged from SLM-408. |
| CAP2 | `supported` | `CERT_CAP2` remains not-issued per SLM-408. DSH5's supported compiler-contract claims are runtime utilities, never a learned CAP2 capability — the certificate posture is retained unchanged, not advanced. |

## Adversarial controls this disposition enforces structurally

* **Fixture, trained, runtime, and deployment verdicts remain distinct.**
  Every claim's `dimension_verdicts` names its runtime-correctness axis
  separately from its learned-benefit axis; `crossover_work`'s
  `fixture_only` runtime cell is never conflated with a systems or
  deployment claim.
* **A semantic gain cannot establish efficiency and a latency gain cannot
  establish semantics.** `systems_efficiency` (the claim) and every other
  claim's own `systems_efficiency` dimension are independently `unavailable`
  — no claim's semantic verdict is used to backfill a missing efficiency
  measurement or vice versa.
* **Unrun conditional work stays `UNRUN_CONDITIONAL`.**
  `AdvancedOperatorClaimV1.__post_init__` requires the headline verdict to
  equal one of its own per-dimension verdicts, so a claim can never report
  "supported" while quietly meaning "the safety wiring was supported but
  the actual question is unrun" — `set_valued_selection` and `event_memory`
  report `unrun_conditional` as their headline precisely because that is the
  decision-relevant, unresolved axis.
* **Failed attempts and fallbacks remain in denominators.** DSH5-06's five-arm
  fixture matrix (`disabled`/`random_score`/`shuffled_conflict_graph`
  reaching the exact accepted set 0/2 times) is cited as-is in the inherited
  evidence chain, never dropped because it is unflattering.
* **Historical DSH3/DSH4 evidence is referenced, not rewritten.**
  `AdvancedOperatorDispositionV1.historical_disposition` points at the
  immutable `dsh3-33-rebased-cap2-disposition-20260726-local/report.json`;
  nothing in this module regenerates or edits that artifact.
* **No advanced path is enabled by default from this disposition alone.**
  `AdvancedOperatorDispositionV1.__post_init__` raises `ValueError` if
  `advanced_path_enabled_by_default` is ever `True`; the published artifact
  sets it to `False` unconditionally.

## AgentV/AgentEvals publication was intentionally skipped

Every prior disposition in this lineage (`dsh3-33-...`, `dsh5-03-...`,
`dsh5-09-...`) published an `agentv` block via
`slm_training.evals.agentv.publish_agentv_evaluation`, which shells out to
the Node.js `@agentv/core` SDK. That SDK (and Node `node_modules` generally)
is not installed in every environment this script runs in, and installing it
here would also pull the `@playwright/test`/`@playwright/mcp` browser-binary
dependencies declared alongside it in `package.json` — multiple gigabytes
this task does not need and this sandbox's disk did not comfortably have
available. `scripts/publish_dsh5_12_advanced_operator_disposition.py`
therefore emits a `self_check` block instead: the same pass/fail facts a
hand-written AgentEvals case would assert (exactly eleven claims, every claim
addressing all four dimensions, unrun/unavailable claims retained as such,
no default-on path, an empty inherited policy inventory, CAP2 retained not
advanced), computed in pure Python and re-verified independently by
`scripts/validate_advanced_operator_disposition.py`. This is recorded here
rather than silently substituted, per this repository's own honesty
convention: a workaround is not a benefit and is not hidden as one.

## Recommendation

**`retain_as_compiler_utility`.** Selector correctness, bulk atomicity,
transaction contracts/execution, sequence merge, and control-plane execution
are proven, tested, replay-exact compiler-owned runtime infrastructure and
should be kept exactly as that — utilities, not shipped model capabilities.
Crossover work, adaptive routing, parameterized templates, and systems
efficiency are `unavailable`: their measurement prerequisites (matched
serving arms, a group-disjoint router corpus, TSA template artifacts, a
completed systems harness) do not exist anywhere in this repository today,
so no claim can be adjudicated positive or negative for them yet.
Set-valued selection and event memory are `unrun_conditional` wiring
preconditions whose held-out-benefit question remains genuinely open and may
continue as **default-off research** (not itself authorized here — a future
issue that actually builds the missing corpus/baseline would make that call)
without contradicting this disposition. Per the issue's own stop rule: no
advanced abstraction has been shown to improve held-out semantics or
measured work beyond the already-proven primitive path, so this is
published as a genuinely mixed, partly negative disposition rather than
forced into a positive spin — exactly the "fully legitimate, expected
possible outcome" the stop rule names.

## Reproducibility

```bash
python -m pytest -q tests/test_evals/test_advanced_operator_disposition.py \
    tests/test_scripts/test_validate_advanced_operator_disposition.py

python -m scripts.publish_dsh5_12_advanced_operator_disposition \
    --output-dir docs/design/dsh5-12-advanced-operator-disposition-20260727-local

python -m scripts.validate_advanced_operator_disposition \
    docs/design/dsh5-12-advanced-operator-disposition-20260727-local/report.json
```
