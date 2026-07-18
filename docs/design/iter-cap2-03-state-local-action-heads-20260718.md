# CAP2-03: state-local action heads and semantic-cost-aware ternary ECOC

**Linear issue:** SLM-88  
**Branch:** `agent/slm-88-state-local-action-heads`  
**Date:** 2026-07-18  
**Run artifact:** `outputs/runs/cap2_state_local_action/cap2_state_local_action_7b7fad506c4bce14.json`

## What changed

Added three new model modules and a fixture harness for CAP2-03:

- `src/slm_training/models/action_code_registry.py`
  - `ActionSchema`, `CodeAssignment`, `ActionCodeEntry`, `ActionCodeRegistry`
  - Deterministic schema/entry hashing; duplicate-schema guard; `codeword_for` / `action_for_codeword` lookups.
- `src/slm_training/models/semantic_cost.py`
  - Ternary codeword generation, base-3 encoding, parity-trit distance-2 code.
  - Uniform and fingerprint-derived pairwise cost matrices.
  - Exact cost-aware assignment for `b <= 6`; deterministic greedy heuristic for larger sets.
  - `build_ternary_ecoc_entry` integration helper.
- `src/slm_training/models/local_action_head.py`
  - `LocalActionHead` interface, `StateContext`, `LocalActionOutput`, `ActionDecision`.
  - Five head families:
    1. `GlobalMaskedHead` — global logits + legal mask.
    2. `LocalFlatHead` — direct scoring of legal actions.
    3. `TernaryDigitHead` — base-3 digit encoding.
    4. `TernaryECOCHead` — distance-2 ternary ECOC with semantic-cost-aware codewords.
    5. `GrammarFactorizedHead` — production-family / slot / ref-class / template factors.
  - Single-legal-action states bypass the learned head and return a forced decision.
- `src/slm_training/harnesses/experiments/cap2_state_local_action.py`
  - Fixture harness that runs all five families on deterministic states.
- `scripts/run_cap2_state_local_action.py`
  - CLI entry point; writes JSON + markdown reports.

## Regression tests

New test file: `tests/test_models/test_local_action_head.py` (18 tests, all passing):

- Forced decision bypasses learned head.
- Base-3 encode/decode round-trip for `TernaryDigitHead`.
- Distance-2 ternary ECOC detects every single-trit corruption.
- Spare-codeword counterexample (`b=5, m=2` without detection) motivates the parity code.
- Semantic-cost assignment places a catastrophic pair at maximum distance.
- Invalid code follows configured fallback and never returns an illegal action.
- Action-schema registry stability / migration (hash determinism).
- Global / local / ternary / ECOC / factorized heads agree on forced single-action states.
- `GlobalMaskedHead` masks illegal actions with `-inf`.
- `GrammarFactorizedHead` reconstructs a legal action.
- Default decode unchanged when the feature is disabled.

## Fixture harness results

Command:

```bash
python -m scripts.run_cap2_state_local_action --out-dir outputs/runs/cap2_state_local_action
```

| head_family      | oracle_accuracy | random_init_accuracy | forced | abstain | detected_error |
| ---------------- | --------------- | -------------------- | ------ | ------- | -------------- |
| global_masked    | 1.0000          | 1.0000               | 1      | 0       | 0              |
| local_flat       | 1.0000          | 1.0000               | 1      | 0       | 0              |
| ternary_digit    | 1.0000          | 1.0000               | 1      | 0       | 0              |
| ternary_ecoc     | 1.0000          | 1.0000               | 1      | 0       | 0              |
| grammar_factorized | 1.0000        | 1.0000               | 1      | 0       | 0              |

Hard gates:

- **PASS** every head family recovered the oracle-encoded action.
- **PASS** no head family emitted an illegal action.

## Honest caveats

- This is a fixture CPU run.  It proves wiring and invariants only.
- A production ship-quality claim requires integration into `TwoTowerModel` training with full `--ship-gates` eval on real OpenUI states and a checkpoint sync to the HF bucket.
- `GrammarFactorizedHead` uses a simplified factor schema; a full implementation must consume the production production-codec sigil alphabet.
- Meaningful-parse remains the primary metric; syntax-parse alone is not sufficient.

## Verification checklist

- [x] `pytest tests/test_models/test_local_action_head.py` — 18 passed.
- [x] `.githooks/check-changed` — 627 passed, 15 deselected.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `git diff --check` — clean.
- [x] Fixture harness `python -m scripts.run_cap2_state_local_action` — 5/5 oracle recovery.
