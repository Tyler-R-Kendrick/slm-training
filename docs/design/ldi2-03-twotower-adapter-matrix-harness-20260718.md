# LDI2-03 — TwoTower adapter-vs-full-update campaign matrix (harness)

**Issue:** SLM-126 (LDI2-03). **Date:** 2026-07-18. **Status:** authorization gate
+ arm plan + guard only. **No training or evaluation run was executed and no
quality claim is made.**

## What this delivers

A Torch-free orchestration layer for the matched TwoTower exact-state
intervention comparison, gated on the LDI2-02 (SLM-125) adapter-subspace
diagnostic:

- `src/slm_training/harnesses/preference/twotower_adapter_matrix.py` —
  `read_authorization` (diagnostic → decision), `build_arms` (T0–T5),
  `exact_signature_guard`, and a guarded executor.
- `scripts/run_twotower_adapter_matrix.py` — dry-run/classify CLI.
- `tests/test_harnesses/preference/test_twotower_adapter_matrix.py` — 9 invariant
  tests.

## Authorization gate (fail-closed — the honesty-critical part)

`read_authorization` maps the SLM-125 report to one of four decisions and
**only ``authorized`` permits training**:

| Diagnostic report | Decision | Effect |
| --- | --- | --- |
| `result.decision == "authorized"` | `authorized` | trainable arms may run |
| `result.decision == "repair_evidence"` | `repair_evidence` | stop — link missing strata |
| `result.decision == "no_safe_direction"` | `no_safe_direction` | stop — do **not** tune to bypass |
| `status == "expired"` | `expired` | stop — fix the diagnostic |
| completed w/ no result, unknown decision, empty report | `no_safe_direction` | **fail closed** |

The unknown/absent-decision cases fail closed to `no_safe_direction` — the harness
never trains on an unauthorized direction, and never tunes LR/duration to bypass
a stop.

## Matched arms & guard

`build_arms` produces T0 parent · T1 full-update control · T2 adapter@authorized
rank · T3 lower-rank · T4 higher-rank (capacity controls) · T5 tether ablation.
`exact_signature_guard` rejects any update that regresses a protected
exact-objective-signature metric — `held_out_loss`/`bad_mass`/`locality` must not
rise, `good_mass`/`margin` must not fall — so an accepted step must pass the guard
or be backtracked (a restored parent is a valid negative safety result).

## Commands (this environment)

```bash
python -m pytest tests/test_harnesses/preference/test_twotower_adapter_matrix.py -q  # 9 passed
PYTHONPATH=src python -m scripts.run_twotower_adapter_matrix --decision authorized \
    --authorized-rank 8 --lower-rank 4 --higher-rank 16 --corpus-admitted          # -> status_counts {'expired': 6}
PYTHONPATH=src python -m scripts.run_twotower_adapter_matrix --decision no_safe_direction \
    --authorized-rank 8                                                             # trainable arms blocked
python -m scripts.repo_policy                                                       # ok
```

Even under an `authorized` decision the arms return `expired` here (no GPU policy
+ admitted corpus) — the harness refuses to invent results.

## What remains (the ship-grade run — deferred)

Out of scope for this environment: GPU + the parent checkpoint used to mine the
V2 events, an admitted identity-homogeneous corpus, the SLM-123 removable adapter
backend / full-update trainer behind a real `policy_factory`, and the five-suite /
AgentV evaluation. At that point admissible arms produce `completed`/`restored_parent`
results with pre/post guarded metrics, and the quality-matrix / model-card /
research-lineage evidence is written per `documenting-experiment-results`. This
note is wiring evidence only.
