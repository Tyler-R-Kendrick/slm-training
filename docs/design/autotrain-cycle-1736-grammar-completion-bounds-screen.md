# Autotrain c1736: grammar-completion-bounds screen

**Verdict:** `grammar_completion_bounds` is an exactly size-matched quality null
on smoke `n=3`. It changes no measured quality metric and shaves `latency_ms_p50`
by only 1.46% (2,429.84 → 2,394.46 ms) — a real but fixture-scale delta that
cannot be credited as a positive latency win because `meaningful_program_rate`
is `0` on both arms, below the `≥ 1/3` floor policy requires before a
latency-only result counts. The arm is rejected and is not checkpoint,
promotion, or ship evidence.

## Result matrix

| Arm | n | Parse | Binder F1 | Meaningful | Structure | AST node / edge F1 | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 3 | 1.0 | .6333 | 0 | .0575 | 0 / 0 | 2,429.84 ms | complete fixture control; gates fail |
| `grammar_completion_bounds=True` | 3 | 1.0 | .6333 | 0 | .0575 | 0 / 0 | 2,394.46 ms | quality null; 1.46% faster, below mpr floor |

Both arms also match placeholder fidelity .5278, reward 0.0, 11 neural forwards,
31,865 unique completion states, 1,638 witness expansions, and 33,620 parser
forks. Training loss is identical (22.6219 both arms — CPU scratch, 21 steps,
seed 100001, batch 2, matched trainable parameters, no lever besides
`grammar_completion_bounds` differing between arms).

`--ship-gates` fail (fixture `n=3` is below the evidence floor); held-out,
adversarial, OOD, and `rico_held` were not run. Lean is `not_applicable:screening`;
promotion and RL remain locked.

## Harness signal and repair

The AgentV publish step failed on the first attempt in this session's fresh
checkout — the smoke suite scored both arms correctly, but
`scripts.evaluate_model --ship-gates` exited 2 because `node_modules` (and
therefore `@agentv/core`) was never installed:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

`npm ci` itself initially failed too, with `node: --import tsx is not allowed
in NODE_OPTIONS` — the container's `NODE_OPTIONS=--import tsx` environment
variable collides with npm's internal Node invocation. Unsetting `NODE_OPTIONS`
for the `npm ci` call installed `@agentv/core` cleanly, and the identical arms
were then replayed end-to-end. This is environment/session provisioning, not a
canonical harness code change, so it is not stack-eligible on its own — it is
recorded here so the next session in this checkout does not re-diagnose it.

This cycle also depended on a first-run driver failure worth recording: the
initial `git fetch origin main` in this session only fetched a stale
`origin/main` ref (`7bb77c9`, 2026-07-27) because an earlier combined
`git fetch origin <branch> main` failed outright when `<branch>` did not exist
remotely yet, short-circuiting before `main` updated. `git fetch --deepen=500
origin main` forced a fresh, correctly-updated `origin/main` (which turned out
to equal `HEAD`, `62f31556`), after which the supervised driver's
`git merge-base --is-ancestor` upstream check passed.

## Repaired next priority

Per the cross-cycle screening-arm bank, the completed `bounds` arm is now
exhausted for this loop's local campaign counter. The next supervised cycle
should test the standalone `component-plan` hypothesis (not jointly with
`component-edge`, which c1735 already rejected as a joint arm):

> Area `experiments`: `component_plan_loss_weight=1.0` improves smoke
> `structural_similarity` without lowering `parse_rate` or `binder_reference_f1`,
> versus the matched control.

Machine-readable evidence is in
[`autotrain-cycle-1736-grammar-completion-bounds-screen.json`](autotrain-cycle-1736-grammar-completion-bounds-screen.json).
