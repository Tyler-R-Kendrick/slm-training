# Continuous self-heal pipeline (2026-08-03)

## Problem

Repeated thrash bank exhaust / harness incompletes required a human to prompt
an agent to “diagnose the blocker and restart.” That violates the continuous
law: soft and healable failures are cycle inputs, not session ends.

## Law

```text
The continuous driver self-heals known recoverable blockers in-process.
Human re-prompt is not a control plane for thrash bank or harness park.
```

## Mechanisms

| Trigger | Heal | Artifact / log |
| --- | --- | --- |
| Static thrash bank multi-seed empty | Compose size-matched lever-pair successors | `loops/<id>/dynamic_thrash_arms.jsonl`, `SELF_HEAL_BANK_EXHAUST` |
| Causal CAP empties multi-seed-open set | Drop CAP skips | `THRASH_CAUSAL_CAP_RELAX` |
| Bank empty + promote head | Promote fallback | `BANK_EXHAUST_PROMOTE_FALLBACK` |
| Harness park + new integration tip | Rearm champion | `CHAMPION_HARNESS_RETRY` |
| Cycle error healable | Clear BLOCKED, continue | `SELF_HEAL continue kind=…` |
| Startup state BLOCKED + healable | Clear blocker before first cycle | `SELF_HEAL_CLEAR_BLOCKER` |

Hard stop (exit 2 after three identical fingerprints) remains only when heal
returns no recovery path.

## Non-goals

- Recycling multi-seed-closed *named* approaches without a new recipe
- Weakening ship gates or multi-seed close policy
- Silent unconstrained decode / gate gaming
