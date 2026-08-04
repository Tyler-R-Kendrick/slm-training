# Autotrain c2 (continuous-openui-local, 2026-08-04 session): AgentV SDK missing on a fresh container, not a model result

**Verdict:** infrastructure failure, not scoreable — a reproduction of
[`autotrain-cycle-c2-agentv-missing-infra-failure.json`](autotrain-cycle-c2-agentv-missing-infra-failure.json)
/ `.md` in a brand-new ephemeral remote-execution container, not a new defect.
Training completed for both the `control` and `canvas` `wf_smoke_v2` arms
(1,608,962 params, 22 steps, loss `14.3902` both, checkpoints
`1bc6370f...9286e` control / `9f73b7a8...053b1a4` canvas, local explicit
no-sync), but `evaluate_model.py --ship-gates` crashed before producing a
scoreboard for either arm: `RuntimeError: AgentV SDK is unavailable; run npm
ci in the checkout or set AGENTV_RUNNER`. Neither arm has smoke metrics; this
is not evidence about the model.

Root cause: this container's checkout was never bootstrapped —
no `.venv`, no torch, no repo-root `node_modules` (`@agentv/core` missing),
and no `node_modules` under `src/apps/openui_bridge` or
`src/apps/design_md_bridge`. The fix for this exact failure mode already
shipped in `scripts/setup_dev_env.sh` (commits `1faeff44`/`8da7b777`, #1360)
from the prior occurrence; it just hadn't been run yet in this fresh
container. No code change was needed this cycle — ran the bootstrap:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # pulls torch via the dev extra
env -u NODE_OPTIONS npm ci               # repo root: installs @agentv/core
(cd src/apps/openui_bridge && env -u NODE_OPTIONS npm ci)
(cd src/apps/design_md_bridge && env -u NODE_OPTIONS npm ci)
```

Verified `node_modules/@agentv/core/package.json` exists afterward.

No scoreboard, no smoke metrics, no ship-gate result exists for this cycle;
the checkpoints are local, explicit no-sync, and not reusable, promotable, or
ship evidence. Lean is `not_applicable:screening`.

Next: replay the identical frozen `canvas` arm (`retry_measurement`,
`frozen_manifest_sha256=209115b1ea6962ff702df035b514d57be16a737bda7e17ec50b9f53b4911d223`)
now that training, evaluation dependencies, and the Node bridges are sound in
this container. Done in c3/c4 — both replays hit a decode timeout on every
document; see
[`autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804.md`](autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804.md)
for that finding and the resulting
[`screening_decode_timeout_seconds` recalibration](autotrain-thrash-timing-pareto-20260804-recalibration.md).

Machine evidence:
[`autotrain-cycle-c2-agentv-missing-infra-failure-20260804.json`](autotrain-cycle-c2-agentv-missing-infra-failure-20260804.json).
