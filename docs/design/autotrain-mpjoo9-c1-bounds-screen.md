# Autotrain continuous-openui-mpjoo9-20260802 cycle 1: bounds screen

**Verdict:** exact quality null on the primary metric. The size-matched
`bounds` candidate ties the matched control on every smoke quality metric and
is 2.3% slower in fixture p50 latency, below any efficiency floor. The arm is
rejected; it is not checkpoint, promotion, or ship evidence. This cycle is
**not positive** per `autotrain-iteration-delivery.md` (fixture `insufficient_n`
+ null primary-metric delta).

## Recipe

- device: cpu (python3.12, `.venv`, `pip install -e ".[dev,torch,grammar]"`,
  torch 2.5.1+cu124 wheel with `torch.cuda.is_available() == False` on this
  sandbox — CPU execution only, no GPU requested)
- train_version: `wf_smoke_v2`, eval_version: `e938_role_safe_all_targets_v2`,
  suite: `smoke`, n=3
- requested steps: 20 (`--steps 20`); actual: 21 (`stopped_on=steps`)
- seed 100001, batch_size 2, lr 3e-4, trainable_parameters 1,608,962 (matched
  across both arms)
- honesty mode: fixture / smoke (wiring-only, not a ship claim)
- upstream_commit == integration_commit == `20ae71ff43051edc79d2475a51848a0bd5adb7f0`
  (branch was already level with `origin/main`; fast-forwarded from `c0762b4`
  before the cycle)

## Result matrix

| Arm | Params | n | Parse | Binder F1 | Meaningful | Structure | AST node/edge F1 | Placeholder fid. | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 3 | 1.0 | 0.6333 | 0 | .0575 | 0 / 0 | .5278 | 2,275.05 ms | gate-rejected; AgentV publish unavailable |
| bounds candidate | 1,608,962 | 3 | 1.0 | 0.6333 | 0 | .0575 | 0 / 0 | .5278 | 2,328.11 ms | exact quality null; 2.33% slower |

Both arms match on every quality metric (structural_similarity, binder F1,
component_type_recall, AST node/edge F1, placeholder fidelity, reward).
Training loss is identical (22.6219) between arms; training wall is
3.275 s (bounds) vs 3.495 s (control). `primary_metric` =
`smoke.structural_similarity`, control = candidate = 0.0575, improvement = 0.0.

## Honest gate and formal state

The base suite scoring (parse_rate, structural_similarity, meaningful_program,
AST metrics, decode_stats) completed for both arms via `evaluate_model`. The
subsequent `--ship-gates` AgentV publish step raised on both arms:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

This is a missing external toolchain in this sandbox (no Node.js/npm, no
`AGENTV_RUNNER`), not a canonical-harness code defect — no `HarnessSignalV1`
was opened. Smoke `n=3` is below the evidence floor and both arms miss
meaningful/structural/AST/reward gates regardless. Held-out, adversarial, OOD,
and `rico_held` suites were not run (screening cycle, fixture scale only).
Lean is `not_applicable:screening`; no champion exists and no formal promotion
claim was attempted.

## SDLC Phase A classification

- `positive`: **false**
- `stack_layer`: **false**, `stack_action`: `no_stack_layer_non_positive`
- reason: `primary_metric_null_or_worse:smoke.structural_similarity:control=0.0575:candidate=0.0575:improvement=0.0`
- This does not meet any of the three positive gates (primary-metric win,
  ship-quality win, executable-unblock) in
  `.claude/skills/sdlc/references/autotrain-iteration-delivery.md`.

## Next-run priorities (from `cycle_handoff.json`)

1. `component-plan` is the next unexhausted size-matched quality hypothesis
   (rank 1, `experiment_next`) — the completed `bounds` null should not be
   re-run.
2. Keep the matched control as baseline every cycle (rank 2, `experiment_next`).
3. Rotate thrash recommendation across the lever bank rather than bounds-only
   (rank 3, `monitor`).
4. Soft ship-gate fails on fixture `n` never stop the continuous loop
   (rank 4, `monitor`).
5. Confirmed champions promote under cadence; thrash only screens
   (rank 5, `monitor`, speculative).

## Follow-up / blocker (deferred, not attempted this session)

AgentV SDK (`npm ci` / Node.js toolchain, or `AGENTV_RUNNER`) is unavailable in
this sandbox, so `--ship-gates` publish always fails on this environment even
though base suite scoring is unaffected. This blocks a full honest ship-gate
scoreboard from ever completing here. Fixing this is an environment-setup task
(install Node/npm and run `npm ci`, or configure `AGENTV_RUNNER`), out of scope
for `improve-openui-harnesses` since no harness code is at fault. Recorded for
the next session/runner that has Node.js available.

Machine-readable evidence is in
[`autotrain-mpjoo9-c1-bounds-screen.json`](autotrain-mpjoo9-c1-bounds-screen.json).
Raw campaign artifacts:
`outputs/autoresearch/continuous-loop-20260802-continuous-openui-mpjoo9-2b5d9d52-c1/`
(gitignored, not committed).
