# Continuous autotrain cycle 2 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Frozen replay of cycle 1's identical `c1-control` / `c1-bounds` arms
(`replay_of continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1`),
after installing the missing AgentV npm deps
([cycle 1 doc](continuous-openui-local-20260802-c1-results.md)).

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2` |
| Source / integration | `b8188a49` / `b51ba33e` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Ship gates | requested (`--ship-gates`), evaluated |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c2-control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 1232.53 | scoreboard complete; **ship gates reject** |
| c2-bounds | bounds on | 3 | 1.0 | 0.0 | 0.0575 | 1243.80 | scoreboard complete; **ship gates reject** |

Primary metric (`smoke.structural_similarity`) delta: **0.0** (tied).

Both arms fail the same gates: `insufficient_n` (3 < 20),
`meaningful_program_rate` (0.0 < 0.66), `structural_similarity` (0.0575 <
0.35), `component_type_recall`, `ast_beq_rate`, `canonical_beq_rate`,
`reward_score`, plus missing `held_out`/`adversarial`/`ood`/`rico_held`
suites (smoke-only run).

## Diagnostics

1. **Executable unblocking confirmed:** the identical frozen arms that
   crashed on `missing_scoreboard` in cycle 1 now complete cleanly and
   produce a full `scoreboard.json` with AgentV assertions, after `npm ci`
   installed `@agentv/core`. This is not itself a positive *model* result —
   it only proves the environment fix worked — and it required no repo code
   change (pure session bootstrap), so it does not open a stack layer either.
2. Ship gates correctly reject both arms on a 3-example smoke fixture: this
   is expected fixture-scale behavior, not a harness defect.
3. `grammar_completion_bounds` (bounds arm) again shows **zero**
   `structural_similarity` delta vs control, consistent with cycle 2's
   (2026-07-30) prior finding that this lever doesn't move smoke quality at
   this scale.

## Classification

Non-positive (`SDLC_PHASE_A NON_POSITIVE`, `stack_layer=False`,
`action=no_stack_layer_non_positive`):
`fixture_insufficient_n` on both arms + `primary_metric_null_or_worse`
(delta 0.0). No stacked PR opened per `sdlc` autotrain-iteration-delivery.

## Next-run priorities (from driver)

1. **model:** test the distinct size-matched `component-plan` quality
   hypothesis next (rank 1, confidence 0.90).
2. **evaluation:** keep the matched control as the size-matched baseline
   every cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop — proceed to cycle 3.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c2-control/`,
  `.../runs/c20260802-continuous-openui-local-8c0b60dd-c2-bounds/`
- JSON twin: `continuous-openui-local-20260802-c2-results.json`
