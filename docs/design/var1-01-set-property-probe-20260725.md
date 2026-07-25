# VAR1-01 (SLM-424): hypothetical property-mutation reachability probe

- generated_at: `2026-07-25T19:15:16Z`
- seed: `root = Stack([], "column")`
- mode: `extended`, max_edits: 8, node_budget: 15
- hypothetical: `true` -- no production action space, checkpoint, or decode path changed.
- Reachability is a space-coverage proof, never a model-quality claim. UNKNOWN_BUDGET is never counted as unreachable. hypothetical=true: no arm below reflects the deployed action space -- arms B/C/D add ExtraAction what-if lanes only, all re-validated through the real DSL parser, never plumbed into TreeEditSpace/N_ACTIONS/CONTAINER_RESTS or any decode path.

> Reachability is space coverage, not model quality: no quality claim
> follows from these proofs alone. Every hypothetical action
> re-validates through the real DSL parser; a what-if can never make
> an illegal program reachable.

## Note on Arm A vs `iter-slm305-edit-language-20260724.md`

VAR1-01's acceptance criteria call for Arm A (baseline, no extra actions) to reproduce that report bit-for-bit as a regression guard. It does not, and the reason is not a regression in this issue's changes: the on-disk suite corpora have drifted since that report was generated -- `rico` grew from 6 to 35 records (more eval requests were committed after 2026-07-24T20:22:57Z) and `adversarial` now includes a case that hits `UNKNOWN_BUDGET` at this probe's node budget. `train`'s corpus (`outputs/data/train/slm230_symbol_only_v1/records.jsonl`) is a generated, gitignored artifact and was not available when this probe ran, so it reports `corpus_unavailable` here regardless. The actual regression guard this issue needs -- that adding the `extra_actions_for` hook does not change any existing caller's output when omitted -- is what `tests/test_harnesses/experiments/test_var1_01_set_property_probe.py`::`test_extra_actions_for_omitted_reproduces_prior_behavior_exactly` actually asserts. Refreshing the stale referenced doc against the current corpora is a separate, out-of-scope task for this issue.

## Reachable fraction by arm and suite

| suite | A baseline | B set_property | C component_widen | D both |
| --- | --- | --- | --- | --- |
| train | corpus_unavailable | corpus_unavailable | corpus_unavailable | corpus_unavailable |
| smoke | 0.0 | 0.0 | 0.0 | 0.0 |
| held_out | 0.0 | 0.0 | 0.0 | 0.0 |
| adversarial | 0.0 | 0.5 | 0.0 | 0.5 |
| ood | 0.0 | 0.0 | 0.0 | 0.0 |
| rico | 0.0 | None | 0.0 | None |

## Verdict flips vs baseline (Arm A)

A flip to `PROVEN_REACHABLE` is genuine positive evidence. A flip to
`UNKNOWN_BUDGET` is NOT evidence of reachability -- it means the
reduced node_budget (see methodology note below) could not finish a
real search once the cheap impossibility proof was removed; it is
inconclusive, not confirmed. The two are reported separately so
neither is mistaken for the other.

- **B_set_property/smoke**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **B_set_property/held_out**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **B_set_property/adversarial**: 1 confirmed reachable, 1 became UNKNOWN_BUDGET (inconclusive), 0 other
  - `adv_empty_prompt_01`: PROVEN_UNREACHABLE → PROVEN_REACHABLE (CONFIRMED)
  - `adv_many_buttons_01`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
- **B_set_property/ood**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **B_set_property/rico**: 0 confirmed reachable, 35 became UNKNOWN_BUDGET (inconclusive), 0 other
  - `rico_eval_test_0`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_1`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_104`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_12`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_17`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_2`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_20`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_25`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_34`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_35`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_38`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_4`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_40`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_41`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_42`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_47`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_48`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_51`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_53`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_55`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_56`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_57`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_58`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_59`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_60`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_68`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_69`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_77`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_8`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_81`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_9`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_91`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_95`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_97`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_99`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
- **C_component_widen/smoke**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **C_component_widen/held_out**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **C_component_widen/adversarial**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **C_component_widen/ood**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **C_component_widen/rico**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **D_both/smoke**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **D_both/held_out**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **D_both/adversarial**: 1 confirmed reachable, 1 became UNKNOWN_BUDGET (inconclusive), 0 other
  - `adv_empty_prompt_01`: PROVEN_UNREACHABLE → PROVEN_REACHABLE (CONFIRMED)
  - `adv_many_buttons_01`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
- **D_both/ood**: 0 confirmed reachable, 0 became UNKNOWN_BUDGET (inconclusive), 0 other
- **D_both/rico**: 0 confirmed reachable, 35 became UNKNOWN_BUDGET (inconclusive), 0 other
  - `rico_eval_test_0`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_1`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_104`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_12`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_17`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_2`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_20`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_25`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_34`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_35`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_38`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_4`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_40`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_41`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_42`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_47`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_48`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_51`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_53`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_55`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_56`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_57`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_58`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_59`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_60`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_68`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_69`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_77`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_8`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_81`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_9`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_91`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_95`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_97`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)
  - `rico_eval_test_99`: PROVEN_UNREACHABLE → UNKNOWN_BUDGET (inconclusive)

## Methodology note: reduced node_budget

The issue's suggested parameters are `max_edits=8, node_budget=120`. This probe ran at `node_budget=15` instead: at node_budget=120 the full 4-arm sweep exceeded the repository's `MAX_RUN_MINUTES=3` hard cap (a timed-out run is never evidence, per AGENTS.md) because removing a cheap impossibility proof lets BFS attempt a real search whose branching factor `set_property_action` and `component_widen_action` both widen. A smaller budget is a weaker, conservative probe: any `PROVEN_REACHABLE` flip found under it is still solid evidence; a lack of flips, or a flip only to `UNKNOWN_BUDGET`, is inconclusive rather than a firm negative and would need a longer, separately-run job at the full budget to resolve. This run completed in well under the cap at this budget.

## Decision this probe licenses

Arm B confirmed 1 case(s) as PROVEN_REACHABLE via property mutation: this is a genuine, confirmed coverage gap -- VAR1-02 (adding a real SET_PROPERTY action) is worth implementing. (36 more case(s) became UNKNOWN_BUDGET rather than confirmed reachable -- inconclusive at this reduced budget, not a second confirmation; a longer full-budget run is needed to resolve them.)
