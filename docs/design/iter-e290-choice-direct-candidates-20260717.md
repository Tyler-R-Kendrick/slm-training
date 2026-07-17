# E290 — grammar-derived choice candidates (2026-07-17)

## Hypothesis and implementation

Cold choice-decoder states should not scan every vocabulary ID. E290 derives a
candidate superset from production categories (statement marker, expression
start, object key, literal byte, and frame closer), then removes unavailable
reference and slot IDs. The existing `advance_id` transition and exact minimum
completion check remain authoritative. Tests compare these candidates with the
exhaustive oracle across reachable frame categories; no prompt, component, or
fixture literals were added.

New telemetry records candidates considered and whole-vocabulary candidates
avoided.

## Bounded recipe

The matched B3 CPU choice recipe used width 64, depth 2, seed 0, a 5,000-token
arm budget, 200-step ceiling, batch size 2, and `max_wall_minutes=5`. It stopped
on the token budget at 107 steps / 5,022 target tokens after 29.22 seconds.
Weighted NLL was 7.09848 (still incomplete for binding). The checkpoint is
byte-identical to E288/E289.

## Results

Two standalone all-suite evaluations each emitted AgentEvals JSONL and a pinned
AgentV bundle (`0/5`, zero execution errors). Quality is unchanged: parse 1.0,
zero dead ends, and meaningful parse / fidelity / reward all 0.0.

| suite | n | avoided cold probes | E290 p50 median | vs E289 p50 | E290 p95 median | vs E289 p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 34.8% | 2,628 ms | 0.89× | 7,527 ms | 1.15× |
| held_out | 5 | 34.8% | 2,038 ms | 0.59× | 5,024 ms | 1.18× |
| adversarial | 4 | 34.8% | 1,951 ms | 0.73× | 5,139 ms | 1.15× |
| ood | 4 | 34.8% | 1,735 ms | 0.67× | 5,177 ms | 1.19× |
| rico_held | 3 | 34.8% | 1,399 ms | 0.74× | 5,243 ms | 1.14× |

Values above 1× are faster. E290 consistently improves the cold p95 tail by
14–19%, but p50 regresses by 11–41% against E289's single baseline run.

## Verdict and feedback

Exact behavior is preserved and cold probing is reduced, but this is a mixed
runtime tradeoff and is not promoted as a default performance win. Semantic
quality and AgentV remain failed.

The next generalized lever is exact grammar-derived lower bounds for completion
length, plus immutable/precomputed candidate partitions, so cache misses avoid
recursive completion searches and transient set construction rather than merely
scanning fewer IDs.

Machine-readable evidence:
[`iter-e290-choice-direct-candidates-20260717.json`](iter-e290-choice-direct-candidates-20260717.json).
