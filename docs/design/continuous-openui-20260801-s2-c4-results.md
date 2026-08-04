# Continuous autotrain cycle 4, second session (2026-08-01) — cross-session synthesis

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (second session; see [summary](continuous-openui-20260801-s2-summary.md)) |
| Campaign | `continuous-loop-20260801-c4` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` (`smoke` n=3, `held_out` n=5) |
| Primary endpoint | `held_out.structural_similarity`, direction increase |

## Run matrix (this session)

| Arm | Levers | Suite | n | completed | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c4-control | steps=21 | smoke | 3 | 1/3 | 1.0 | 0.0 | 0.4167 | 13700.22 |
| c4-control | steps=21 | held_out | 5 | 1/5 | 1.0 | 0.0 | 0.3417 | 19458.37 |
| c4-steps | steps=42 | smoke | 3 | 3/3 | 1.0 | 0.3333 | 0.51 | 14430.36 |
| c4-steps | steps=42 | held_out | 5 | 5/5 | 1.0 | 0.0 | 0.37006 | 13944.15 |

This session's own driver classifies this **positive**
(`primary_metric_win`, `held_out.structural_similarity` 0.3417 → 0.37006,
+0.02836) — numerically identical to what's already documented in open PR
[#1248](https://github.com/Tyler-R-Kendrick/slm-training/pull/1248)
("steps lever mpr win").

## The problem: three open PRs, three different verdicts for "cycle 4"

Because `outputs/autoresearch/` is gitignored, each fresh session's driver
starts its own local cycle counter at 1 for the loop id
`continuous-openui-20260801`. Three parallel sessions today all ran a
"cycle 4" `steps=21` vs `steps=42` comparison and got three different
answers, all currently sitting as open, unmerged PRs:

| PR | control held_out.structural_similarity | steps held_out.structural_similarity | Delta | Verdict |
| --- | ---: | ---: | ---: | --- |
| [#1248](https://github.com/Tyler-R-Kendrick/slm-training/pull/1248) | 0.3417 | 0.37006 | +0.0284 | positive |
| [#1247](https://github.com/Tyler-R-Kendrick/slm-training/pull/1247) | 0.38248 | 0.37006 | -0.0124 | **regression** |
| This session (#1249-adjacent, unopened) | 0.3417 | 0.37006 | +0.0284 | positive (matches #1248) |

`c4-steps` (`held_out.structural_similarity = 0.37006`) is **identical across
all three sessions** — that arm reliably completes all 5/5 held_out
documents inside the 3-minute wall cap every time. `c4-control` is the one
that varies (0.3417 in two sessions, 0.38248 in the third), and its held_out
p50 latency clusters near 19.1-19.5s in all three (19067.94ms, 19458.37ms,
19385.10ms) — consistent with a single slow first document dominating a
1-document (or otherwise partial) completed sample, not a stable 5-document
average.

**Root-cause hypothesis:** the steps lever's apparent effect on
`held_out.structural_similarity` is confounded with how many `held_out`
documents the **control** arm happens to finish before the shared 3-minute
wall — a real-machine-speed race that differs by container, not a property
of doubling training steps. `c4-steps` looks stable only because doubling
steps happens to make the model converge to something that finishes the
suite reliably; that's a plausible real effect, but the *headline delta*
(`+0.0284` or `-0.0124` depending on session) is not trustworthy evidence of
its size or even its sign, because the control side of the comparison isn't
sampling the same population run to run.

## Recommendation

Do **not** credit the steps lever with the `held_out.structural_similarity`
delta, and do **not** promote or add further stacked layers on top of the
steps=42 vs steps=21 finding in #1246, #1247, or #1248, until the harness
either:

1. requires `held_out.completed_document_n == n` on **both** arms before
   Phase A calls a `primary_metric_win`/regression on that suite, or
2. raises the promotion-cycle wall cap enough that `control` reliably
   completes the full `held_out` suite too.

This is a measurement-integrity gap in the `model_build`/eval harness family
(partial-suite averaging racing a wall-clock cap), not a validated lever
result — route to `improve-openui-harnesses` as a harness signal rather than
treating any of the three PRs' deltas as ship-relevant evidence.

## SDLC Phase A

This session's driver alone would call `positive=True`,
`stack_layer=False`, `action=positive_no_tracked_delta_skip_stack` (a pure
recipe finding, no code delta). Given the cross-session synthesis above
shows the same nominal experiment flips sign in a sibling open PR, this
write-up treats the cycle as **not eligible for a new stacked layer**:
promoting either sign would misrepresent an unreliable measurement as
settled evidence. Docs-only, local commit.

## Next-run priorities

1. File a harness signal (family: `model_build`) via `improve-openui-harnesses`:
   gate primary-metric comparisons on full suite completion before Phase A
   calls a win/regression.
2. Until that lands, treat the steps=42 vs steps=21 held_out finding across
   #1246/#1247/#1248/this doc as an open question, not a result.
3. Re-run after the completion-gate fix, at the same or larger `held_out` n.
