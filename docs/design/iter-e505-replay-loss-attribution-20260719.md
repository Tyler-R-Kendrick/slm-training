# E505 — source-stratified replay loss attribution

E505 asks whether the E500 primary objective diverges while the easier E357
replay objective improves. Train v6 now summarizes the existing detached
per-example masked-token loss proxy by source, using bounded first-20 and
last-20 windows. This is telemetry only: it changes neither the scalar objective
nor gradients.

## Matched replication

The single training arm exactly repeats E504's 50% replay recipe: E396 bucket
initialization, E500 primary corpus, immutable E357 replay corpus, CPU/frozen
local SmolLM2-135M, choice output, d128/h4/c2/dn4, batch 2, LR `3e-4`, seed 0,
and 5,000 target tokens. It completed 101 steps in 93.82 seconds with
`max_wall_minutes=3.0` and an external 170-second cap.

| Source | Examples | Overall proxy | First 20 | Last 20 | Decline |
| --- | ---: | ---: | ---: | ---: | ---: |
| E500 primary | 100 | 3.5924 | 3.8422 | 3.3724 | 12.23% |
| E357 replay | 102 | 3.1212 | 3.4087 | 2.9217 | 14.29% |

Both source losses decline, so simple primary-loss divergence is falsified.
Primary examples remain harder throughout: the primary-minus-replay gap is
0.4335 in the first window and 0.4506 in the last, a 3.95% widening. These
scalar proxies do not establish whether the two sources' gradients conflict.

The matched honest smoke `n=3` exactly reproduces E504: syntax 1.0, structure
0.2469, recall 0.0833, AST node F1 0.3148, and meaningful rate, fidelity,
reward, and AgentV all zero. E504 and this matched eval use honest slot-contract
scoring with slot-contract decode bias **off**.

## Decode-policy ablation

An additional evaluation of the same checkpoint enabled constrained
slot-contract decode. Fidelity rose to 0.1667 and reward to 0.2623, while
structure fell to 0.2039 and AST node F1 to 0.0952. Meaningful rate remained
zero and AgentV remained 0/1. This is promising wiring evidence, not a ship or
promotion result; it needs a larger capped diagnostic.

## Decision

Keep the bounded source-loss telemetry and reject the E505 checkpoint. Do not
change synthesis or replay scheduling based on this run. The next discriminating
test is gradient alignment or a larger constrained-slot-contract decode
diagnostic.

Exact hashes and metrics:
[machine-readable record](iter-e505-replay-loss-attribution-20260719.json).
