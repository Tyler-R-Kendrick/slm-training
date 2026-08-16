"""Environment playbook: the documented ``npm ci`` bridge/AgentV repair.

``.agents/skills/autotrain/references/continuous.md`` documents the exact
bounded local fix for the most common true harness crash (missing AgentV SDK
/ JS bridge deps on a fresh checkout or container): ``npm ci`` in
``src/apps/openui_bridge``, ``src/apps/design_md_bridge``, and the repo root
with ``NODE_OPTIONS`` cleared. Until this playbook existed the fix was
documented but never attempted without a human-opened agent session (FP2).

Verify probe (skeptic O5a): ``healed`` requires the original failing
resolution to pass — a ``node -e require.resolve`` probe for the missing
module named in the crash reason when parsable, else the AgentV SDK core
module — never ``npm ci`` exit 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.autoresearch.heal.classify import missing_js_module
from slm_training.autoresearch.heal.escalation import blocker_fingerprint
from slm_training.autoresearch.heal.schemas import (
    HealPlanV1,
    HealStepV1,
    HealVerifyV1,
)
from slm_training.levers import MAX_RUN_SECONDS

PLAYBOOK_ID = "npm_bridges/v1"

_INSTALL_ROOTS: tuple[str, ...] = (
    "src/apps/openui_bridge",
    "src/apps/design_md_bridge",
    "",  # repo root — AgentV SDK ship-gate eval dependency tree
)

_DEFAULT_PROBE_MODULE = "@agentv/core"

_STEP_TIMEOUT = int(MAX_RUN_SECONDS)


class _NpmBridgesPlaybook:
    playbook_id = PLAYBOOK_ID
    handles = frozenset({"environment"})

    def matches(self, blocker: dict) -> bool:
        return str(blocker.get("kind") or "") == "repair_harness"

    def plan(self, blocker: dict, *, cwd: Path) -> HealPlanV1 | None:
        reason = str(blocker.get("reason") or "")
        module = missing_js_module(reason) or _DEFAULT_PROBE_MODULE
        steps = []
        for index, rel in enumerate(_INSTALL_ROOTS):
            target = cwd / rel if rel else cwd
            if not (target / "package.json").is_file():
                continue
            steps.append(
                HealStepV1(
                    step_id=f"npm_ci_{index}_{rel or 'root'}".replace("/", "_"),
                    argv=("npm", "ci"),
                    cwd=rel,
                    # Some containers set a NODE_OPTIONS that breaks plain
                    # `npm ci` (documented in continuous.md).
                    env_clears=("NODE_OPTIONS",),
                    timeout_seconds=_STEP_TIMEOUT,
                    writes_allowed=(
                        f"{rel}/node_modules/" if rel else "node_modules/",
                        f"{rel}/package-lock.json" if rel else "package-lock.json",
                    ),
                )
            )
        if not steps:
            return None
        return HealPlanV1(
            playbook_id=self.playbook_id,
            blocker_fingerprint=blocker_fingerprint(
                str(blocker.get("kind") or ""), reason
            ),
            blocker_class="environment",
            steps=tuple(steps),
            # The probe runs from the repo root deliberately: the ship-gate
            # eval that crashed resolves from there, so root resolution IS the
            # original failing check. json.dumps hardens the module name
            # against quotes/backslashes in hostile crash text.
            verify=HealVerifyV1(
                argv=(
                    "node",
                    "-e",
                    f"require.resolve({json.dumps(module)})",
                ),
                cwd="",
                env_clears=("NODE_OPTIONS",),
                timeout_seconds=60,
            ),
        )


PLAYBOOK = _NpmBridgesPlaybook()
