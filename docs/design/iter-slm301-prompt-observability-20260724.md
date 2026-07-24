# SLM-301 (LAR1-04): prompt observability — as-is vs +inventory arms

- generated_at: `2026-07-24T18:18:24Z`
- claim_class: `diagnostic` (diagnostic; fixture-scale, not ship evidence)
- predeclared: n=16 test_seeds, seeds=[0], min useful paired delta=0.1 v2-strict on OBSERVABLE_REQUEST_ONLY/UNKNOWN rows
- rico: **not_run** — SLM-294 measured >280s for 16 tokens with this checkpoint at rico prompt scale; 35 rico rows declared not_run (classification only, phase 0).

## Phase 0: observability coverage classes (model-free)

| corpus | n | OBSERVABLE_PROMPT | OBSERVABLE_REQUEST_ONLY | UNKNOWN |
| --- | --- | --- | --- | --- |
| test_seeds | 16 | 0 | 16 | 0 |
| slm294_rico | 35 | 0 | 35 | 0 |

### Per-row classes

- **test_seeds**:
  - `smoke_hero_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=4, decode=phase1)
  - `smoke_button_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=1, decode=phase1)
  - `smoke_callout_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=3, decode=phase1)
  - `held_out_form_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=7, decode=phase1)
  - `held_out_dual_card_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=5, decode=phase1)
  - `held_out_input_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=2, decode=phase1)
  - `held_out_tabs_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=6, decode=phase1)
  - `held_out_settings_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=5, decode=phase1)
  - `adv_empty_prompt_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=1, decode=phase1)
  - `adv_dual_card_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=4, decode=phase1)
  - `adv_deep_nest_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=3, decode=phase1)
  - `adv_many_buttons_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=3, decode=phase1)
  - `ood_dashboard_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=5, decode=phase1)
  - `ood_gallery_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=6, decode=phase1)
  - `ood_modal_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=3, decode=phase1)
  - `ood_auth_01` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=3, decode=phase1)
- **slm294_rico**:
  - `rico_eval_test_0` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_1` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=6, decode=not_run)
  - `rico_eval_test_2` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_4` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_8` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_9` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_12` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=12, decode=not_run)
  - `rico_eval_test_17` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_20` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=4, decode=not_run)
  - `rico_eval_test_25` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=8, decode=not_run)
  - `rico_eval_test_34` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=4, decode=not_run)
  - `rico_eval_test_35` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_38` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_40` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=10, decode=not_run)
  - `rico_eval_test_41` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=12, decode=not_run)
  - `rico_eval_test_42` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=8, decode=not_run)
  - `rico_eval_test_47` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=7, decode=not_run)
  - `rico_eval_test_48` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_51` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_53` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=10, decode=not_run)
  - `rico_eval_test_55` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=9, decode=not_run)
  - `rico_eval_test_56` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_57` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=9, decode=not_run)
  - `rico_eval_test_58` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=4, decode=not_run)
  - `rico_eval_test_59` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=7, decode=not_run)
  - `rico_eval_test_60` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=4, decode=not_run)
  - `rico_eval_test_68` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_69` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=4, decode=not_run)
  - `rico_eval_test_77` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=8, decode=not_run)
  - `rico_eval_test_81` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=12, decode=not_run)
  - `rico_eval_test_91` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_95` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_97` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=12, decode=not_run)
  - `rico_eval_test_99` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)
  - `rico_eval_test_104` OBSERVABLE_REQUEST_ONLY (inventory_in_prompt=False, slots=11, decode=not_run)

## Phase 1: decode outcomes (16 test_seeds, seed 0)

### x22_deterministic

| slice | n_measured | v2-strict (Wilson 95%) | syntax |
| --- | --- | --- | --- |
| arm_A | 16/16 | 0.000 [0.000, 0.194] | 1.0 |
| arm_A.observable_prompt | 0/0 | unmeasured | — |
| arm_A.observable_request_only | 16/16 | 0.000 [0.000, 0.194] | 1.0 |
| arm_A.unknown | 0/0 | unmeasured | — |
| arm_B | 16/16 | 0.312 [0.142, 0.556] | 1.0 |
| arm_B.observable_prompt | 0/0 | unmeasured | — |
| arm_B.observable_request_only | 16/16 | 0.312 [0.142, 0.556] | 1.0 |
| arm_B.unknown | 0/0 | unmeasured | — |

- paired **target**: delta=0.3125 (min useful 0.1, meets=True, n_measured=16) table={'both_pass': 0, 'a_pass_b_fail': 0, 'a_fail_b_pass': 5, 'both_fail': 11, 'unpaired': 0}
- paired **no_regression**: delta=None (min useful 0.1, meets=False, n_measured=0) table={'both_pass': 0, 'a_pass_b_fail': 0, 'a_fail_b_pass': 0, 'both_fail': 0, 'unpaired': 0}
- paired **all**: delta=0.3125 (min useful 0.1, meets=True, n_measured=16) table={'both_pass': 0, 'a_pass_b_fail': 0, 'a_fail_b_pass': 5, 'both_fail': 11, 'unpaired': 0}

### ar_tiny

| slice | n_measured | v2-strict (Wilson 95%) | syntax |
| --- | --- | --- | --- |
| arm_A | 2/16 | 0.000 [0.000, 0.658] | 0.0 |
| arm_A.observable_prompt | 0/0 | unmeasured | — |
| arm_A.observable_request_only | 2/16 | 0.000 [0.000, 0.658] | 0.0 |
| arm_A.unknown | 0/0 | unmeasured | — |
| arm_B | 4/16 | 0.000 [0.000, 0.490] | 0.0 |
| arm_B.observable_prompt | 0/0 | unmeasured | — |
| arm_B.observable_request_only | 4/16 | 0.000 [0.000, 0.490] | 0.0 |
| arm_B.unknown | 0/0 | unmeasured | — |

- paired **target**: delta=0.0 (min useful 0.1, meets=False, n_measured=1) table={'both_pass': 0, 'a_pass_b_fail': 0, 'a_fail_b_pass': 0, 'both_fail': 1, 'unpaired': 15}
- paired **no_regression**: delta=None (min useful 0.1, meets=False, n_measured=0) table={'both_pass': 0, 'a_pass_b_fail': 0, 'a_fail_b_pass': 0, 'both_fail': 0, 'unpaired': 0}
- paired **all**: delta=0.0 (min useful 0.1, meets=False, n_measured=1) table={'both_pass': 0, 'a_pass_b_fail': 0, 'a_fail_b_pass': 0, 'both_fail': 1, 'unpaired': 15}

## AgentEvals

```json
{
  "artifacts": {
    "benchmarkPath": "outputs/runs/slm301_prompt_observability/agentv/slm301-prompt-observability/benchmark.json",
    "indexPath": "outputs/runs/slm301_prompt_observability/agentv/slm301-prompt-observability/index.jsonl",
    "runDir": "outputs/runs/slm301_prompt_observability/agentv/slm301-prompt-observability",
    "timingPath": "outputs/runs/slm301_prompt_observability/agentv/slm301-prompt-observability/timing.json"
  },
  "format": "AgentEvals JSONL",
  "sdk": "@agentv/core",
  "spec": "outputs/runs/slm301_prompt_observability/agentv/slm301-prompt-observability.eval.jsonl",
  "summary": {
    "durationMs": 35,
    "executionErrors": 0,
    "failed": 0,
    "meanScore": 1,
    "passed": 2,
    "total": 2
  }
}
```

## Required-slot judge/metric audit (100 corpus records)

One blind agent rater pass (non-human, disclosed) over 100 corpus prompts (97 train + 16 test_seeds sample), labeling whether exact required slots are determinable from the prompt alone, versus the v2 coverage classification:

- Judge: 3 `observable` / 97 `not_observable`; metric: 0 `OBSERVABLE_PROMPT` / 100 not-prompt-observable.
- Raw agreement 97% (3 mismatches); Cohen κ = 0.0 — κ collapses under the extreme marginal skew (97/100 one label), so raw agreement is the honest summary.
- **Finding** (`docs/design/slm301-slot-observability-audit-20260724.json`): the 3 mismatches (`train_hero_01_fd_0/1/2`) name exact slots inline — e.g. `Lay out a single hero card in a column using :hero.title and :hero.body.` — but the v2 prompt-visible coverage detector only recognizes inventory **sections** (`Placeholders:`/`Inventory:` lines), not inline slot enumeration, producing false `prompt_contract_unknown` on inline-enumerated prompts. Metric revision is SLM-288's owner's call; this is an append-only finding — no metric change made here.
