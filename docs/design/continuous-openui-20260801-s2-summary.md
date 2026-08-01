# Continuous autotrain loop `continuous-openui-20260801`, second session summary (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

This is a second, independent container instance of today's
`continuous-openui-20260801` continuous-autotrain loop (see the `autotrain`
skill, bare `/autotrain` continuous mode). Because `outputs/autoresearch/`
(the campaign bundle root, and the only place loop cycle state persists) is
gitignored and therefore not shared across sessions, this session's driver
started its own local cycle counter at 1, colliding in name (but not in
git-tracked content, since no branch here had merged first) with cycle
numbers already used and documented in three other open PRs from earlier
sessions today:

- [#1246](https://github.com/Tyler-R-Kendrick/slm-training/pull/1246) —
  `docs(autotrain): continuous loop 2026-08-01 — cycles c1-c4 (non-positive)`
- [#1247](https://github.com/Tyler-R-Kendrick/slm-training/pull/1247) —
  `docs(autotrain): document continuous loop cycles 1-4 (env repair +
  non-positive screening)`
- [#1248](https://github.com/Tyler-R-Kendrick/slm-training/pull/1248) —
  `docs(autotrain): continuous loop 2026-08-01 cycles 1-4 — steps lever mpr
  win`

To avoid re-claiming those PRs' `docs/design/continuous-openui-20260801-c{1..4}-results.*`
filenames (which would create a merge conflict against whichever of those
three PRs lands first), this session's cycle docs are filed under the
`continuous-openui-20260801-s2-*` prefix instead:

| File | Cycles | Verdict |
| --- | --- | --- |
| [c1c2](continuous-openui-20260801-s2-c1c2-results.md) | 1-2 | environment bootstrap failures (torch extra, AgentV `npm ci`), self-healed |
| [c3](continuous-openui-20260801-s2-c3-results.md) | 3 | `grammar_completion_bounds` + `compact_active_canvas`: latency-only win, rejected (low mpr) |
| [c4](continuous-openui-20260801-s2-c4-results.md) | 4 | steps=42 vs steps=21 held_out `structural_similarity`: **cross-session synthesis** — this session's result matches #1248's "positive" numbers exactly, but #1247 measured a regression on the same nominal cycle; traced to a wall-clock completion race on the control arm, not a real lever effect |
| [c5](continuous-openui-20260801-s2-c5-results.md) | 5 | `batch_size=1` vs `batch_size=2`: much worse latency, no quality offset, non-positive |
| [c6c7](continuous-openui-20260801-s2-c6c7-results.md) | 6-7 | levers from c3 isolated: `grammar_completion_bounds` alone regresses latency; `compact_active_canvas` alone reproduces c3's latency-only win (still rejected, mpr=0.0) |

## Headline finding: the open steps-lever PRs disagree, and now we know why

The most useful output of this session is not a new lever win — it's
diagnosing *why* three sessions running the same nominal "cycle 4" recipe
(`steps=21` control vs `steps=42` candidate, `wf_smoke_v2`/`e938_role_safe_all_targets_v2`,
3-minute wall) landed on three different verdicts (positive in #1248 and
here, a regression in #1247). See
[c4's write-up](continuous-openui-20260801-s2-c4-results.md) for the full
analysis: the `held_out.structural_similarity` primary is unreliable when
the control arm doesn't finish the full `held_out` suite inside the wall
cap, and that completion count varies by container/session, not by the
steps lever.

## SDLC Phase A across this session

None of this session's 7 cycles earn a new stacked layer:

- c1, c2: infrastructure-only, no metrics (`no_stack_layer_non_positive`).
- c3, c5, c6, c7: real screening cycles, all non-positive.
- c4: this session's own driver would call it `positive`
  (`primary_metric_win`), matching #1248 exactly — but the cross-session
  synthesis above shows that specific "win" is not trustworthy evidence
  (a sibling open PR measured the opposite sign on the identical nominal
  recipe). Promoting it to a stacked layer here would add a fourth
  conflicting claim about the same finding rather than resolve the
  disagreement. This write-up documents the disagreement and its root
  cause instead.

Docs-only, local commits, no new stacked PR from this session — consistent
with `autotrain-iteration-delivery.md`'s non-positive path.

## Next steps for a future loop session

1. Route a harness signal (family: `model_build`) to `improve-openui-harnesses`:
   require `completed_document_n == n` on the compared suite before Phase A
   calls a primary-metric win/regression, or raise the promotion-cycle wall
   cap so `held_out` reliably completes on both arms.
2. Once that lands, re-run steps=42 vs steps=21 for a trustworthy delta, and
   only then consider whether #1246/#1247/#1248's steps-lever claims should
   be reconciled or superseded.
3. Consider a session-start hook that runs `uv pip install -e ".[torch]"`
   and `env -u NODE_OPTIONS npm ci` once per fresh container so future
   sessions don't spend a cycle on the same two bootstrap gaps documented in
   [c1c2](continuous-openui-20260801-s2-c1c2-results.md).
