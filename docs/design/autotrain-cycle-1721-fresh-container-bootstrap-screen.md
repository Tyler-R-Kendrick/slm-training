# Autotrain c1721: fresh-container bootstrap and honest fixture screen

**Verdict:** fixture/scratch measurement complete for both arms; not a promotion
candidate. This cycle ran in a brand-new container with no prior `.venv`, no
`torch`, and no `npm ci` install, so the first attempt failed on environment
gaps rather than model or harness code. After bootstrapping a `python3.12`
venv (`pip install --no-deps -e .` + the pinned dev/runtime extras + CPU
`torch` from the PyTorch CPU index) and running `npm ci` at the repo root with
`NODE_OPTIONS` overridden (the ambient `NODE_OPTIONS="--import tsx" ...` is
rejected by this Node build, matching the precedent in
[`dsh5-10-replay-preference-rows.md`](dsh5-10-replay-preference-rows.md)),
both arms trained fresh (no reused checkpoint — this loop-id had no prior
local state) and completed honest smoke evaluation.

## Result matrix

| Arm | Training | Smoke result (n=3) | Gate outcome | Disposition |
| --- | --- | --- | --- | --- |
| control | fresh, 21 steps, 1,608,962 trainable params, seed 100001 | parse 1.0; binder F1 0.6333; meaningful 0.0; structural 0.0575; p50 2,084.47 ms | Complete AgentV run; honest gates fail on evidence volume + quality thresholds | Model evidence only; not ship |
| bounds (`--grammar-completion-bounds`) | fresh, 21 steps, same size/seed | parse 1.0; binder F1 0.6333; meaningful 0.0; structural 0.0575; p50 2,251.00 ms | Complete AgentV run; identical gate failures | Model evidence only; not ship |

Primary metric `smoke.binder_reference_f1` is identical between arms
(0.6333 vs 0.6333, improvement 0.0) — no candidate win. Both arms fail
`smoke:insufficient_n actual=3 need>=20` plus the `held_out` / `adversarial` /
`ood` / `rico_held` suites are absent (`missing_suite`), so this is a fixture
screening measurement, not a powered comparison. Full metrics, gate failure
lists, and version stamps are in the
[machine-readable record](autotrain-cycle-1721-fresh-container-bootstrap-screen.json).

## Diagnostic signal

Two environment gaps blocked the first run attempt in this fresh container,
both infrastructure, neither a harness defect:

1. `scripts.train_model` → `detect_device` raised
   `ModuleNotFoundError: No module named 'torch'` — the venv had installed
   `slm_training` with `--no-deps` per CI convention but never installed the
   `torch` extra. Fixed with
   `pip install "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu`
   (2.5.1+cpu), matching `pyproject.toml`'s `dev` extra pin.
2. `scripts.evaluate_model` → `publish_agentv_evaluation` raised
   `RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set
   AGENTV_RUNNER` — the repo-root `npm ci` had not been run in this container.
   `npm ci` itself failed first with `node: --import tsx is not allowed in
   NODE_OPTIONS` because the container's ambient `NODE_OPTIONS` includes
   `--import tsx`; overriding to `NODE_OPTIONS="--max-old-space-size=8192"`
   for the install (and for the eval subprocess) resolved it.

Neither gap touched repository code, so no harness repair or version bump is
required; this doc plus the venv/npm-ci recipe above is the record so the
next session in a fresh container does not re-diagnose the same two failures.

## Next-run priorities

Unchanged from the standing continuous-loop priority bank (screen the
`bounds` thrash arm against the matched `control`, keep the control as the
size-matched baseline, avoid single-lever thrash collapse, never let fixture
`insufficient_n` alone stop the loop). No promotion, no theorem optimum, and
no checkpoint worth carrying forward — c1721's checkpoints are local-only
fixture-scale artifacts under `outputs/` (gitignored) and are not referenced
by `docs/MODEL_CARD.md`.

Eval commit: `6f38011faff5913f564fbe7969b934b1c580320c`
(`model.twotower=v274`, unchanged from c1720's repair). No component version
bump — this cycle changed no versioned file.
