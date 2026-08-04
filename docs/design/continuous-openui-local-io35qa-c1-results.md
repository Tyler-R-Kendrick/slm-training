# Continuous autotrain: 2026-08-04 (session io35qa) cycle 1 — infra harness_failure, resolved (no repo code defect)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`
**Upstream / integration commit:** `f4949582` (clean `origin/main` tip; no local
divergence at cycle start)

## What happened

Both arms (`c20260804-continuous-openui-local-8c0b60dd-c1-control` and
`...-bounds`) exited `2` immediately, before any train/eval work ran:

```
ModuleNotFoundError: No module named 'torch'
RuntimeError: torch is not installed in this environment. Run
scripts/setup_dev_env.sh to install the pinned CPU wheel, or `pip install
--index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu` directly.
```

This is a fresh Claude Code on the web / remote-execution container with no
prior `.venv` and no dependency bootstrap — not a defect in
`src/slm_training/runtime/accel/__init__.py` or any `model_build` harness
code. The existing `RuntimeError` message already names the correct fix.

## SDLC Phase A

**Non-positive** (`no_stack_layer_non_positive` — `measurement_incomplete`,
`harness_failure` on both arms, `primary_metric_unavailable`). No stack layer
for this cycle.

## Repair (action index 0: `repair_harness`, owner `improve-openui-harnesses`)

No repo code changed for the failure itself — investigation confirmed the
guard behaves correctly. The unblocking action was environment provisioning,
exactly per `scripts/setup_dev_env.sh`:

1. `python3.12 -m venv` + `pip install -e . --no-deps`
2. `pip install pytest ruff numpy httpx fastapi lark openfeature-sdk pydantic
   PyYAML onnxruntime`
3. `pip install --index-url https://download.pytorch.org/whl/cpu
   torch==2.5.1+cpu`
4. `env -u NODE_OPTIONS npm ci` for the AgentV SDK (`scripts/run_agentv_eval.mjs`)
   — this container's global `NODE_OPTIONS=--import tsx` crashes plain `node`
   the same way earlier sessions found for the AgentV SDK (see PR #1351, #1363,
   #1391); `setup_dev_env.sh` already works around it with `env -u NODE_OPTIONS`.

**Durable fix, so this does not recur on the next fresh container:** added
`.claude/hooks/session-start.sh` (registered as a `SessionStart` hook in
`.claude/settings.json`) that runs the same bootstrap automatically for
remote/web sessions (`CLAUDE_CODE_REMOTE=true`), idempotently, so a future
scheduled autotrain session starts with `torch` and the AgentV SDK already
importable instead of burning cycle 1 on this exact `ModuleNotFoundError`.

## Verification (executable unblock)

The identical frozen arm pair was replayed in cycle 2 of this same session
after the fix — see
[`continuous-openui-local-io35qa-c2-results.md`](continuous-openui-local-io35qa-c2-results.md)
for the completed measurement.

Machine evidence:
[`continuous-openui-local-io35qa-c1-results.json`](continuous-openui-local-io35qa-c1-results.json).
