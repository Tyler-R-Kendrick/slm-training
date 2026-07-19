# E503 — initialized-weight retention during E396→E500 continuation

E503 tests whether the E502 5k collapse is caused by trainable-weight drift.
The canonical warm-start loop now optionally contracts trainable weights toward
their initialized checkpoint after every optimizer step. A coefficient of zero
is the matched control; one exactly retains the initialized weights. The train
summary records the coefficient, anchored parameter count, and final RMS drift.

## Matched recipe

All four arms use the same E396 checkpoint and committed 260-row E500 corpus,
CPU, frozen local SmolLM2-135M context, choice output, d128/h4/c2/dn4, batch 2,
LR `3e-4`, seed 0, uniform record sampling, and the E396 slot/component recipe.
Each arm stops at 5,019 target tokens after 99 steps. Every train summary records
`max_wall_minutes=3.0`, and every process had an external 170-second cap.

Evaluation is the same honest diagnostic smoke `n=3` used by E502:
prompt-derived slot contracts, constrained LTR decode, no fallback, four
generation steps, one attempt, and a 96-token cap. Every evaluation emitted
AgentEvals plus a pinned AgentV bundle without execution errors.

| Retention | RMS drift | Last loss | Structure | Recall | AST node F1 | Meaningful / fidelity / reward | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0% control | 0.003123 | 12.8937 | 0.0927 | **0.1667** | 0.1972 | 0 / 0 / 0 | 0/1 |
| 1% | 0.002071 | 13.4907 | 0.0900 | **0.1667** | 0.1942 | 0 / 0 / 0 | 0/1 |
| 3% | 0.001163 | 14.4501 | 0.1667 | 0.0833 | 0.2917 | 0 / 0 / 0 | 0/1 |
| 5% | **0.000811** | 14.7205 | **0.2029** | 0.0 | **0.3175** | 0 / 0 / 0 | 0/1 |

The v4 zero-retention control exactly reproduces E502 structure and recall,
confirming attribution. Retention monotonically reduces weight drift. Stronger
retention suppresses duplicate-subtree spam and recovers structure, but it also
suppresses adaptation: the 5% arm emits two trivial/empty layouts and loses all
component recall. The 3% midpoint preserves some recall but remains below the
frozen parent's `0.2117` structure. No arm moves a semantic gate.

## Decision

Keep the explicit retention lever and drift telemetry because they make
continuation experiments measurable. Reject all E503 checkpoints for promotion
or bucket sync. Weight anchoring exposes a structure/recall frontier but does not
solve semantic adaptation; the next matched lever should interleave provenance-
preserving parent replay instead of tightening retention further.

Exact hashes and metrics:
[machine-readable record](iter-e503-initialized-weight-retention-20260719.json).
