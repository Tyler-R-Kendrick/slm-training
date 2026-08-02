# Autotrain continuous-openui-mpjoo9-20260802 cycle 2: component-plan wall-timeout

**Verdict:** no honest comparison is possible this cycle. Both arms trained to
completion (22/20 steps), but the matched control's `evaluate_model` call was
never reached before the campaign's per-arm wall budget
(`max_wall_minutes=1.1667`, derived from the repo's `MAX_RUN_MINUTES=3` cap
split across the campaign) ran out — it has **no** scoreboard at all, not even
base-suite scoring. The `component-plan` candidate did evaluate and scored
`structural_similarity=0.0964`, but with no control baseline this is not a
valid A/B comparison. Per `autotrain-iteration-delivery.md` this cycle is
**not positive** (`primary_metric_unavailable`); this is a soft failure and,
per `continuous.md`, soft failures never stop the loop.

## Recipe

- device: cpu (existing `.venv`, python3.12, `pip install -e
  ".[dev,torch,grammar]"`, torch 2.5.1+cu124 wheel, CPU execution only)
- train_version: `wf_smoke_v2`, eval_version: `e938_role_safe_all_targets_v2`,
  suite: `smoke`
- requested steps: 20; actual: 22 (`stopped_on=steps`) on both arms
- both arms size-matched at 1,755,764 trainable params (component-plan head
  prebuilt on both, per the campaign matrix)
- honesty mode: fixture / smoke (wiring-only, not a ship claim)
- upstream_commit `d2c3b97c1148bd503773a169890a2d5310ab3cdb` →
  integration_commit `a72a453e17bc3f8218ebc802d16cd3f4e9f18ee2` (branch
  fast-forward-merged from `origin/main` before this cycle, resolving a
  concurrent-session conflict in README/MODEL_CARD/versions.json from PR
  #1301 landing mid-loop)

## Result matrix

| Arm | Params | Steps | Last loss | Smoke n | Parse | Structure | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,755,764 | 22 | 14.3902 | — | — | — | — | trained; evaluation skipped (wall-budget exhausted) |
| component-plan candidate | 1,755,764 | 22 | 19.9160 | 3 | 1.0 | .0964 | 2,300.04 ms | evaluated; `--ship-gates` fails, AgentV unavailable |

Checkpoints: control
`417b3ce8…423c274cc`, candidate `68f1ce95…6bf903bdf` (both written, both
gitignored under `outputs/`, not synced/promoted).

## Honest gate and formal state

The control's `evaluate_model` invocation never ran — the events log shows
`experiment_finished` for the control's *training* stage only; no evaluation
command was executed for it before the campaign wall clock closed the cycle.
The candidate's `evaluate_model --ship-gates` completed its base suite scoring
(parse_rate, structural_similarity, etc.) but the AgentV publish step raised
the same `RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout
or set AGENTV_RUNNER` seen in cycle 1 (`mpjoo9-c1`) — no Node.js/npm toolchain
in this sandbox. Neither arm clears honest ship gates. Lean is
`not_applicable:screening`; no champion exists and no formal promotion claim
was attempted.

## SDLC Phase A classification

- `positive`: **false**
- `stack_layer`: **false**, `stack_action`: `no_stack_layer_non_positive`
- reasons: `wall_timeout:76ba3bbc916ec3e1e9d34efdfb6a6c59756a4805de6c2258b9d734a5e8dedc82`,
  `primary_metric_unavailable`
- Neither of the three positive gates (primary-metric win, ship-quality win,
  executable-unblock) applies when the primary metric has no control value to
  compare against.

## Next-run priorities (from `cycle_handoff.json`)

1. The `component-plan` hypothesis is exhausted (candidate scored, but the
   comparison is unusable) — try `component-edge` next (rank 1,
   `experiment_next`).
2. Keep the matched control as baseline every cycle (rank 2,
   `experiment_next`).
3. Rotate thrash recommendation across the lever bank, not bounds-only
   (rank 3, `monitor`).
4. Soft ship-gate/wall-timeout fails on fixture `n` never stop the continuous
   loop (rank 4, `monitor`).
5. Confirmed champions promote under cadence; thrash only screens (rank 5,
   `monitor`, speculative).

## Follow-up / blockers (deferred, not attempted this session)

1. **AgentV SDK / Node.js toolchain** — same recurring gap as cycle 1
   (`autotrain-mpjoo9-c1-bounds-screen.md`); `npm ci` / `AGENTV_RUNNER` not
   available in this sandbox, blocking any full honest ship-gate scoreboard
   here regardless of model quality.
2. **Per-arm wall-budget exhaustion** — a 2-arm campaign's
   train+evaluate-both-arms sequence does not reliably fit inside the derived
   `max_wall_minutes=1.1667` per-arm budget on CPU once one arm's evaluation
   (including grammar-constrained decode) is slow; the control lost the race
   this cycle. Not a code defect — a future cycle could lower `--steps`
   further or investigate reordering evaluation before the second arm trains,
   but this is left as an observation rather than a harness change, since a
   single occurrence does not yet establish a reproducible `HarnessSignalV1`.

Machine-readable evidence is in
[`autotrain-mpjoo9-c2-component-plan-timeout.json`](autotrain-mpjoo9-c2-component-plan-timeout.json).
Raw campaign artifacts:
`outputs/autoresearch/continuous-loop-20260802-continuous-openui-mpjoo9-2b5d9d52-c2/`
(gitignored, not committed).
