---
name: evidence-brief
description: Use when a scheduled or interactive session should report the current autotrain evidence state — open hypothesis families, closed approaches, preflight blocks, pending confirmations, promoted-model status — as a short read-only markdown brief in chat
---

# Evidence brief

## Overview

**Strictly read-only.** This skill writes **nothing** to the repository: no
files, no branches, no commits, no PRs, no Linear mutations. The deliverable is
a short markdown brief posted to chat (or the scheduled session's summary).
If you are tempted to persist the brief, stop — persistence belongs to
`documenting-experiment-results` runs, not this digest.

## Sources (all guarded — read whichever exist, skip the rest)

Check each path before reading; several are being added incrementally, so a
missing file is normal and is reported as "not present", never as an error.

| Source | Path (repo-relative) | Use |
| --- | --- | --- |
| Climb policy | `src/slm_training/resources/experiments/autotrain_climb/policy.v2.json`, **else** `policy.v1.json` | Policy version, screening/promotion primary metrics, caps |
| Evidence ledger | `src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json` | Per-arm `n_obs` / `mean_delta` / `n_positive` — which hypothesis families are open vs exhausted |
| Closed approaches | `src/slm_training/resources/experiments/autotrain_climb/closed_approaches.v1.json` | Recently closed approaches (a rejected experiment closes an approach, never a goal) |
| Evidence store | `src/slm_training/resources/evidence_store/local_index.jsonl` — query via `python -m scripts.query_evidence` **when that script exists**, else read the JSONL directly | Recent runs, preflight blocks, pending confirmations |
| Merge preflight | `python -m scripts.verify_merge_ready` output **when present** (run read-only, report only) | Current preflight-block status |
| Model card | `docs/MODEL_CARD.md` (plus README "Model card (summary)") | Currently promoted model / roster status |

## Brief shape (post to chat)

Keep it under ~40 lines. Sections, in order:

```markdown
# Evidence brief — <UTC date>

## Open hypothesis families
- <family>: n_obs=<n>, mean_delta=<d>, n_positive=<p>  (from evidence_ledger)

## Recently closed approaches
- <approach> — closed <date/reason>   (from closed_approaches.v1.json; "none recorded" if file absent)

## Recent preflight blocks
- <block source + one-line reason>    (evidence store / verify_merge_ready; "none observed" otherwise)

## Pending confirmations
- <promotion candidates awaiting locked-eval / multi-seed confirmation>  (policy `promotion_primary` vs ledger)

## Promoted model status
- <roster role, run id, checkpoint URI, claim level>  (from docs/MODEL_CARD.md)

Sources read: <list>  ·  Sources absent: <list>
```

Always end with the sources-read / sources-absent line so a thin brief is
distinguishable from a broken one.

## Honesty rules

- Report numbers exactly as stored; never recompute a gate or restate a
  fixture/diagnostic result as a ship claim (`honest-ship-eval` claim levels).
- An absent artifact is "not present", not "zero" and not "failing".
- Do not summarize away caveats the model card attaches to a promotion
  (suite `n`, honesty mode, diagnostic-only labels).
- Never suggest weakening a gate or bypassing `scripts.verify_merge_ready` as
  a remedy for a block — briefs report state; levers change elsewhere.
