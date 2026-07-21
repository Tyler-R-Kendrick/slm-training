# E634 — required-slot coverage on an 800-step scratch checkpoint

Date: 2026-07-20  
Status: completed negative; checkpoint rejected; not ship

E619 removed the last evaluator ambiguity from this lineage: coverage was fully
judged, binding correctness passed, and strict meaning-v2 remained zero because
the model did not emit the complete visible placeholder inventory. E634 tests
the simplest remaining capacity hypothesis by training the same scratch
TwoTower recipe for 800 steps instead of 80, then replaying E619's matched OOD
`n=4` control/treatment recipe.

## Recipe and persistence

The train used the committed E530 corpus (244 records), scratch context, choice
output tokenizer, seed 0, batch size 1, and explicit
`--no-sync-checkpoints`. It completed all 800 steps in 80.96 seconds under
`max_wall_minutes=3`; loss reached 4.0680. The clean serving checkpoint is:

`outputs/runs/e620-required-slot-coverage-scratch800-20260720/checkpoints/last.pt`

SHA-256: `3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`.
It is local-only, rejected, and not promoted.

Both eval arms used the identical checkpoint and E619 policy:
`honest_slot_contract`, `slot_contract_constrained_decode`,
`slot_contract_in_context`, public semantic-role schema candidates, and the
retained semantic-plan weights. Only `schema_role_slot_decode_weight` changed:
0 for control, 8 for treatment. Both runs emitted AgentEvals JSONL and an
AgentV SDK bundle.

## Measured result

| Metric | Control | Treatment | Delta |
| --- | ---: | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 | 0 |
| meaningful v1 | 0.5000 | 0.5000 | 0 |
| strict meaning v2 | 0.0000 | 0.0000 | 0 |
| v2 judgment coverage | 1.0000 | 1.0000 | 0 |
| placeholder fidelity | 0.5083 | 0.5500 | +0.0417 |
| placeholder validity | 0.7050 | 0.7300 | +0.0250 |
| structural similarity | 0.4569 | 0.4886 | +0.0317 |
| component recall | 0.5417 | 0.4792 | -0.0625 |
| reward | 0.7910 | 0.8140 | +0.0230 |
| AST node F1 | 0.5556 | 0.5437 | -0.0119 |
| AST edge F1 | 0.3750 | 0.3750 | 0 |
| latency p50 | 3366.95 ms | 3116.67 ms | -250.28 ms |
| latency p95 | 14562.73 ms | 13704.44 ms | -858.29 ms |
| AgentV | 0/1 | 0/1 | 0 |

There were no decode timeouts or fallbacks. The E633 object-property role bias
still produces a real matched improvement in fidelity, validity, structure,
reward, and latency, but it does not increase strict meaning or required-slot
coverage.

## Capacity hypothesis result

Duration scaling is rejected for this scratch recipe. Compared with E619's
80-step treatment, 800 steps lowers loss from 26.5243 to 4.0680 but regresses
fidelity from 0.7833 to 0.5500, validity from 0.8700 to 0.7300, and structure
from 0.5548 to 0.4886. Strict meaning remains 0.0. This is train-distribution
fit without the expected OOD coverage improvement.

The per-record failures are now concrete:

- Dashboard emits one `Card` and one of five slots; components and slots are
  missing, and the surviving body slot is placed in `Card.variant`.
- Gallery correctly assigns `img` to `src` and `alt` to `alt`, but still omits
  caption, CTA, and both hint slots.
- Modal emits title/body but omits confirm and misuses schema-value positions.
- Auth emits every slot and every component but swaps `Input.name` roles, so it
  fails semantic-role correctness rather than mechanical coverage.

## Decision

Reject and do not sync or promote the E634 checkpoint. Retain E633's generalized
typed-object role bias. The next useful lever is coverage-aware closure across
component and property boundaries, with role-correctness preserved; another
duration-only scratch train is not justified by this result.

Raw evidence: [JSON](iter-e634-required-slot-coverage-scratch800-20260720.json).
