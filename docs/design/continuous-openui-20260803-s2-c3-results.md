# Continuous autotrain: 2026-08-03 session 2, cycle 3 — fresh-seed confirmation REJECTS the component-plan champion

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Base commit:** `20665535` (this session's cycle-2 docs commit)
**Seed:** `100003` — genuinely distinct from `100002`, the seed used by both
today's earlier session's merged c2 ([#1369](https://github.com/Tyler-R-Kendrick/slm-training/pull/1369))
and this session's cycle 2 (which was a bit-exact replay, not an
independent measurement — see
[`continuous-openui-20260803-s2-c2-results.md`](continuous-openui-20260803-s2-c2-results.md)).

| Arm | Params | MPR | structural_similarity | binder F1 | reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,764 | .3333 | .23083 | .4889 | .283 | 2231.70 |
| confirm (component-plan) | 1,755,764 | .3333 | .23083 | .2667 | .283 | 2251.55 |

**Verdict: the previously-reported component-plan win does not reproduce.**
Under this fresh seed, `structural_similarity` (the declared primary) ties
*exactly* between control and candidate — no improvement — and
`binder_reference_f1` **regresses** by −0.222 relative to control, which is
flagged as a real non-regression failure, not noise. This is the actual
independent fresh-seed confirmation both the original (`#1369`) and this
session's cycle 2 called for and could not themselves supply (both used the
same deterministic `seed=100002`).

**Conclusion: the queued champion is falsified.** The driver auto-rejected
`champ-continuous-openui-local-2-e19bda467f7df6df` (the `component-plan`
candidate queued after cycle 2). The `+0.0561` structural-similarity win
recorded twice today (once yesterday's session, once this session's replay)
was seed-specific noise at fixture scale (`n=3`), not a real effect of the
`component-plan` knob. **Do not promote or re-select it without a new
hypothesis.**

Per `sdlc` autotrain-iteration-delivery: **no stacked PR** — non-positive
cycle, docs-only local commit.

## Next priorities

1. Do not re-queue `component-plan` from the `seed=100002` measurement alone
   — it's now falsified.
2. Keep training loss as a diagnostic only; it diverged from certified
   program quality on this confirmation.
3. Preregister a new, distinct structural/quality hypothesis rather than
   recycling the now-exhausted `component-plan`/`bounds` pair.
4. Merge [#1351](https://github.com/Tyler-R-Kendrick/slm-training/pull/1351)
   — every cycle this session needed a local `npm ci` + `NODE_OPTIONS`
   workaround to get a real AgentV evaluation at all.

Machine evidence:
[`continuous-openui-20260803-s2-c3-results.json`](continuous-openui-20260803-s2-c3-results.json).
