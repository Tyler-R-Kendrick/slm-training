"""SLM-492: honest external-blocked prepared-package closeout for EXP-SR-11.

Reuses RSP-007 (``probe_pysr_environment`` / SRP-011 adapter isolation) without
claiming SOTA or scoring a benchmark loss when PySR/Julia are unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.symbolic_expr_pysr_adapter import DEFAULT_PYSR_MANIFEST
from slm_training.harnesses.experiments.rsp007_pysr_srbench import (
    CATALOGUE_ID,
    EVIDENCE_EXTERNAL_BLOCKED,
    PRIMARY_METRIC,
    Rsp007CampaignV1,
    probe_pysr_environment,
    run_campaign,
)
from slm_training.versioning import build_version_stamp, git_commit

LINEAR_ISSUE = "SLM-492"
DESIGN_STEM = "docs/design/iter-slm492-exp-sr-11-external-blocked-closeout-20260810"
VERSION_COMPONENT = "harness.experiments.slm492_external_blocked_closeout"
RESULT_SCHEMA = "slm492_external_blocked_closeout/v1"

PREPARED_REPLAY = "python -m scripts.run_rsp007_pysr_srbench --mode fixture --seed 0"

__all__ = [
    "DESIGN_STEM",
    "LINEAR_ISSUE",
    "PREPARED_REPLAY",
    "RESULT_SCHEMA",
    "VERSION_COMPONENT",
    "probe_environment",
    "render_markdown",
    "run_closeout",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def probe_environment(
    manifest: Any = DEFAULT_PYSR_MANIFEST,
) -> dict[str, Any]:
    """Record PySR import and Julia binary availability — never installs anything."""
    pysr_probe = probe_pysr_environment(manifest)
    julia_path = shutil.which("julia")
    julia_available = julia_path is not None
    pysr_spec = importlib.util.find_spec("pysr")
    return {
        **pysr_probe,
        "pysr_import_available": pysr_spec is not None,
        "live_execution_ready": pysr_spec is not None and julia_available,
        "julia_binary_available": julia_available,
        "julia_path": julia_path,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }


def run_closeout(*, root: Path, seed: int = 0) -> dict[str, Any]:
    """Execute the prepared external-blocked package and return durable evidence."""
    env_probe = probe_environment()
    campaign = Rsp007CampaignV1(seed=seed)
    rsp007 = run_campaign(campaign, root=root / "rsp007_fixture")

    return {
        "kind": RESULT_SCHEMA,
        "linear_issue": LINEAR_ISSUE,
        "catalogue_id": CATALOGUE_ID,
        "primary_metric": PRIMARY_METRIC,
        "status": "external_blocked",
        "disposition": "external_blocked",
        "incomplete": True,
        "external_blocked": True,
        "promotion": False,
        "no_sota_claims": True,
        "claim_class": "diagnostic",
        "srbench_matched_score_gap": None,
        "env_probe": env_probe,
        "prepared_package": {
            "rsp007_harness_replay": PREPARED_REPLAY,
            "rsp007_owner": "src/slm_training/harnesses/experiments/rsp007_pysr_srbench.py",
            "srp011_adapter_owner": "src/slm_training/dsl/symbolic_expr_pysr_adapter.py",
            "prior_fixture_evidence": "docs/design/iter-slm486-rsp-007-pysr-srbench-20260810.json",
        },
        "rsp007_result": rsp007,
        "scope_disclaimer": (
            "SLM-492 closes EXP-SR-11 follow-up as an honest external-blocked prepared "
            "package: RSP-007 harness and SRP-011 adapter isolation are replayable, but "
            "PySR/Julia absence yields incomplete evidence — never a benchmark loss or "
            "SOTA claim. srbench_matched_score_gap stays null."
        ),
        "timestamp": _now(),
        "recipe": {
            "command": ["python", "-m", "scripts.run_slm492_external_blocked_closeout"],
            "docs_path": f"{DESIGN_STEM}.json",
            "device": "cpu",
            "honesty_mode": "external_blocked_prepared_package",
        },
        "version_stamp": build_version_stamp(VERSION_COMPONENT),
        "evidence_cutoff_commit": git_commit(),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    env = payload.get("env_probe", {})
    prepared = payload.get("prepared_package", {})
    rsp007 = payload.get("rsp007_result", {})
    pysr_arm = rsp007.get("pysr_adapter_matched", {})
    lines = [
        f"# {LINEAR_ISSUE}: EXP-SR-11 external-blocked prepared-package closeout",
        "",
        f"**Status:** `{payload.get('status')}` / incomplete — not a benchmark loss.",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id')}`",
        "",
        f"**Primary metric (`{PRIMARY_METRIC}`):** `{rsp007.get(PRIMARY_METRIC)}`",
        "",
        f"**No SOTA claims.** `{payload.get('no_sota_claims')}`",
        "",
        "## Environment probe",
        "",
        f"- PySR import available: `{env.get('pysr_import_available')}`",
        f"- Julia binary available: `{env.get('julia_binary_available')}`",
        f"- Live execution ready: `{env.get('live_execution_ready')}`",
        f"- Manifest fingerprint: `{str(env.get('manifest_fingerprint', ''))[:16]}…`",
        f"- Tool version (pinned): `{env.get('tool_version')}`",
        "",
        "## Prepared package pointers",
        "",
        f"- RSP-007 replay: `{prepared.get('rsp007_harness_replay')}`",
        f"- RSP-007 owner: `{prepared.get('rsp007_owner')}`",
        f"- SRP-011 adapter: `{prepared.get('srp011_adapter_owner')}`",
        f"- Prior SLM-486 evidence: `{prepared.get('prior_fixture_evidence')}`",
        "",
        "## RSP-007 external-blocked snapshot",
        "",
        f"- evidence: `{rsp007.get('evidence')}`",
        f"- complete: `{rsp007.get('complete')}`",
        f"- pysr_adapter status: `{pysr_arm.get('status')}`",
        f"- pysr validation_loss: `{pysr_arm.get('validation_loss')}`",
        "",
        "## Scope",
        "",
        payload.get("scope_disclaimer", ""),
        "",
        "Command: `python -m scripts.run_slm492_external_blocked_closeout`",
        "",
    ]
    return "\n".join(lines)


def load_closeout_summary(path: Path | None = None) -> dict[str, Any]:
    """Load SLM-492 closeout JSON for disposition supersession wiring."""
    target = path or Path(f"{DESIGN_STEM}.json")
    if not target.is_file():
        return {"present": False, "path": str(target)}
    data = json.loads(target.read_text(encoding="utf-8"))
    rsp007 = data.get("rsp007_result", {})
    return {
        "present": True,
        "path": str(target),
        "status": data.get("status"),
        "external_blocked": data.get("external_blocked"),
        "evidence": rsp007.get("evidence", EVIDENCE_EXTERNAL_BLOCKED),
        "complete": rsp007.get("complete", False),
        "srbench_matched_score_gap": data.get("srbench_matched_score_gap"),
        "evidence_cutoff_commit": data.get("evidence_cutoff_commit"),
    }
