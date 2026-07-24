# SLM-303 (LAR1-05): decode-budget audit — census, 10x sweep, disposition

- generated_at: `2026-07-24T19:08:59Z`
- claim_class: `diagnostic` (diagnostic; not ship evidence)
- taxonomy: `model_valid, model_invalid, model_abstain, runtime_timeout, fallback_output, harness_error`
- budget evidence: inferred from recorded eval_smoke.json: all 3 smoke rows share latency_ms=3333.79 (one chunk of 3 killed at ~10001ms → 10s timeout); the original CLI flag was not persisted (config=null), disclosed as inferred

## Census

- local checkpoints enumerated: 216 (roster rows: 115)
- committed SHA verified: 1; mismatch: 0; unverifiable (no committed SHA): 305
- scoreboard classes: `{"fallback_interference": 2, "model_behavior": 1134, "runtime_timeout_interference": 10, "unmeasured": 4}`

### Hash pins (sweep checkpoints)

| key | path | sha256 | matches scoreboard record |
| --- | --- | --- | --- |
| ltr2 | `outputs/runs/iter-published-remediated-64step-ltr2-20260715/checkpoints/best_weighted_nll.pt` | `653e71b286964373eda30ff7fb28a836bf0514c6bf78cac9fb07a12223f8fa3a` | True |
| lexer_ltr2 | `outputs/runs/iter-published-remediated-64step-lexer-ltr2-20260715/checkpoints/best_weighted_nll.pt` | `4573bf7d0f606e9f91471a402988ebdcbb7596cf8425ac95fea54866f360310c` | True |

## Budget sweep (preregistered, paired)

| checkpoint | record | baseline outcome | 10x outcome | baseline ms | 10x ms |
| --- | --- | --- | --- | --- | --- |
| lexer_ltr2 | smoke_button_01 | not_rerunnable | not_rerunnable | 2091.33 | 1683.37 |
| lexer_ltr2 | smoke_callout_01 | not_rerunnable | not_rerunnable | 1643.82 | 1693.5 |
| lexer_ltr2 | smoke_hero_01 | not_rerunnable | not_rerunnable | 1901.12 | 1589.12 |
| ltr2 | smoke_button_01 | not_rerunnable | not_rerunnable | 1276.21 | 1519.6 |
| ltr2 | smoke_callout_01 | not_rerunnable | not_rerunnable | 1196.94 | 1842.18 |
| ltr2 | smoke_hero_01 | not_rerunnable | not_rerunnable | 1217.16 | 2402.0 |

- outcome counts (baseline): `{"model_valid": 0, "model_invalid": 0, "model_abstain": 0, "runtime_timeout": 0, "fallback_output": 0, "harness_error": 0}`
- outcome counts (10x): `{"model_valid": 0, "model_invalid": 0, "model_abstain": 0, "runtime_timeout": 0, "fallback_output": 0, "harness_error": 0}`

## Disposition (append-only annotations)

- `outputs/runs/iter-published-remediated-64step-lexer-ltr2-constrained-fullsmoke-20260715/eval.json`: runtime/harness artifact: 3/3 decode timeouts at canvas cap 128; zero parse rate on these rows is budget interference, NOT a model verdict
- `outputs/runs/iter-published-remediated-64step-lexer-ltr2-constrained-fullsmoke-20260715/eval_smoke.json`: runtime/harness artifact: 3/3 decode timeouts at canvas cap 128; zero parse rate on these rows is budget interference, NOT a model verdict
- `outputs/runs/iter-published-remediated-64step-ltr2-constrained-fullsmoke-20260715/eval.json`: runtime/harness artifact: 3/3 decode timeouts at canvas cap 256; zero parse rate on these rows is budget interference, NOT a model verdict
- `outputs/runs/iter-published-remediated-64step-ltr2-constrained-fullsmoke-20260715/eval_smoke.json`: runtime/harness artifact: 3/3 decode timeouts at canvas cap 256; zero parse rate on these rows is budget interference, NOT a model verdict
- `docs/design/iter-e497-current-main-playground-provenance-smoke-20260718.json`: runtime/harness artifact: 1/3 decode timeouts at canvas cap None; zero parse rate on these rows is budget interference, NOT a model verdict
- `docs/design/iter-slm303-decode-budget-audit-20260724.json`: runtime/harness artifact: 3/3 decode timeouts at canvas cap 128; zero parse rate on these rows is budget interference, NOT a model verdict
- `docs/design/iter-slm303-decode-budget-audit-20260724.json`: runtime/harness artifact: 3/3 decode timeouts at canvas cap 128; zero parse rate on these rows is budget interference, NOT a model verdict
- `docs/design/iter-slm303-decode-budget-audit-20260724.json`: runtime/harness artifact: 3/3 decode timeouts at canvas cap 256; zero parse rate on these rows is budget interference, NOT a model verdict
- `docs/design/iter-slm303-decode-budget-audit-20260724.json`: runtime/harness artifact: 3/3 decode timeouts at canvas cap 256; zero parse rate on these rows is budget interference, NOT a model verdict
- `docs/design/iter-slm303-decode-budget-audit-20260724.json`: runtime/harness artifact: 1/3 decode timeouts at canvas cap None; zero parse rate on these rows is budget interference, NOT a model verdict
- `docs/design/iter-e763-e764-symbol-only-heldout-fallback-20260722.json`: fallback interference: fallback_count=4; fallback outputs never classify as model success
- `docs/design/iter-slm303-decode-budget-audit-20260724.json`: fallback interference: fallback_count=4; fallback outputs never classify as model success

- sweep verdict: not_rerunnable: 6/6 paired cells blocked by the output-contract gate (checkpoint v0 vs required symbol_only/v2, no migration path); the 10x-budget flip question is UNANSWERED by re-decode — census + taxonomy carry the audit
- powered-rerun recommendation: at most one: retrain the remediated recipe from symbol-only targets (output contract v2 — the v0 checkpoints are blocked by require_current_output_contract with no migration path), then a preregistered full-smoke re-eval at recorded vs 10x decode budget with per-record decode_outcome fields (now emitted by eval_runner), n≥16 smoke+fixture records for Wilson resolution
  - rationale: census localizes all timeout interference to the two remediated checkpoints; no other hash-verifiable checkpoint carries a timeout-flagged scoreboard, and re-decode of the v0 checkpoints under current code is impossible

Historical iter docs and MODEL_CARD rows are untouched; these annotations are additive.
