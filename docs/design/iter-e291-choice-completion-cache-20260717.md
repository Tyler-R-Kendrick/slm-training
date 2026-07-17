# E291 — exact completion-state cache (2026-07-17)

## Hypothesis and implementation

Many token choices produce the same post-transition symbolic state. E291 caches
the exact minimum completion length by `ChoiceDecodeState.signature()` and
reuses expression partitions keyed by request-local slot/reference counts.
Legality is not approximated: the production transition, completion result, and
exhaustive oracle remain authoritative.

New telemetry records completion-cache hits and misses.

## Bounded recipe

The matched B3 CPU recipe used width 64, depth 2, seed 0, a 5,000-token arm
budget, 200-step ceiling, batch size 2, and `max_wall_minutes=5`. It stopped at
107 steps / 5,022 target tokens after 24.58 seconds. Weighted NLL was 7.09848,
still incomplete for binding. The checkpoint SHA is identical to E288–E290.

## Results

A schema-warm state-level control produced the same 442 legal IDs in 4.34 ms
with completion caching versus 30.35 ms without it (7.0×). The cache recorded
375 hits and 74 misses. A separate 2.38-second process-cold measurement exposed
lazy component-schema construction as the remaining first-request cost.

Two standalone all-suite evaluations each emitted AgentEvals JSONL and a pinned
AgentV bundle (`0/5`, zero execution errors). Parse remains 1.0 with zero dead
ends; meaningful parse, fidelity, and reward remain 0.0.

| suite | n | completion hit rate | p50 median | vs E290 | p95 median | vs E290 | p95 vs E289 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 91.4% | 1,666 ms | 1.58× | 5,000 ms | 1.51× | 1.73× |
| held_out | 5 | 91.9% | 1,085 ms | 1.88× | 2,702 ms | 1.86× | 2.19× |
| adversarial | 4 | 90.7% | 982 ms | 1.99× | 2,761 ms | 1.86× | 2.14× |
| ood | 4 | 91.4% | 1,343 ms | 1.29× | 2,677 ms | 1.93× | 2.30× |
| rico_held | 3 | 91.2% | 863 ms | 1.62× | 2,775 ms | 1.89× | 2.16× |

Relative to E289, four p50s improve 1.10×–1.44×; OOD p50 is 0.87×. All p95s
improve 1.73×–2.30×.

## Verdict and feedback

E291 is an exact and repeatable runtime improvement over E290, especially in
the tail. It does not make this checkpoint promotable or ship-ready because all
semantic gates and AgentV still fail.

The next runtime iteration should build the pinned component contracts during
model/tokenizer initialization and report startup separately from request
latency. The next quality iteration still needs semantic/data improvement; no
amount of deterministic syntax acceleration fixes zero fidelity.

Machine-readable evidence:
[`iter-e291-choice-completion-cache-20260717.json`](iter-e291-choice-completion-cache-20260717.json).
