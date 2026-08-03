# Continuous autotrain: 2026-08-03 cycle 3 — bounds arm null delta (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `318492c5` (`main` tip)

**Verdict:** `bounds` arm ties its size-matched control exactly on the
declared primary at this seed. Non-positive — no new stacked PR. Fixture
screening only, not a ship or promotion claim.

| Arm | Params | Seed | structural_similarity | component_type_recall | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 100001 | .05750 | 0 | .63333 | 1331.46 |
| bounds | 1,608,962 | 100001 | .05750 | 0 | .63333 | 1713.89 |

Primary improvement `+0.0`. `parse_rate` (1.0), `meaningful_program_rate`
(0), `component_type_recall` (0), and `binder_reference_f1` (.6333) are
identical on both arms. The only observed difference is p50 latency
(1331.46ms control vs 1713.89ms bounds) — a **regression**, not a win, and it
does not clear the quality-aware latency-win bar (`meaningful_program_rate`
is 0 on both arms, so held-parse/mpr cannot be satisfied).

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` suites were not run.

## Campaign-ID collision note

This session's `outputs/autoresearch/` root is git-untracked and does not
persist across scheduled sessions, so its campaign counter independently
restarted at `cycle_index=1` and produced the on-disk id
`continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1` — a string
byte-identical to the campaign_id already documented in
[`continuous-openui-20260803-c1-results.json`](continuous-openui-20260803-c1-results.json)
from an earlier, unrelated session today. The two docs describe different
runs with different data despite the shared campaign_id string; this doc
uses the day's next free numbered slug (`c3`) so neither doc is overwritten.
Flagged as a harness observation: campaign_id generation is not
collision-resistant across independently-provisioned ephemeral sessions that
share a `loop_id` and UTC date.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse` + `fixture_insufficient_n_alone`).
Per `sdlc` autotrain-iteration-delivery, this cycle stays a **local commit
only** — no stacked PR is opened for a null-delta screening result.

```json
{
  "reasons": [
    "fixture_insufficient_n:c20260803-continuous-openui-local-8c0b60dd-c1-control",
    "fixture_insufficient_n:c20260803-continuous-openui-local-8c0b60dd-c1-bounds",
    "primary_metric_null_or_worse:smoke.structural_similarity:control=0.057499999999999996 candidate=0.057499999999999996 improvement=0.0",
    "fixture_insufficient_n_alone"
  ]
}
```

## Next priorities (ranked)

1. The completed non-positive `bounds` arm is exhausted; test the distinct
   size-matched `component-plan` quality hypothesis next (confidence 0.90).
2. Keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).
3. Rotate the thrash recommendation across the lever bank instead of
   re-running `bounds`-only knobs (confidence 0.65).

Full JSON: [`continuous-openui-20260803-c3-results.json`](continuous-openui-20260803-c3-results.json).
