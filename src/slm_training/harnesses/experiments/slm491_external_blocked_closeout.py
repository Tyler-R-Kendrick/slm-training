"""SLM-491: honest external-blocked prepared-package closeout for EXP-SR-3.

Reuses SIE-006 (``run_external_blocked_campaign``) and the frozen SIE-005 packet
without fabricating human ratings or external-judge calibration metrics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.sie006_construct_validity import (
    CATALOGUE_ID,
    PRIMARY_METRIC,
    SIE005_FROZEN_DOC,
    Sie006CampaignV1,
    run_external_blocked_campaign,
)
from slm_training.versioning import build_version_stamp, git_commit

LINEAR_ISSUE = "SLM-491"
DESIGN_STEM = "docs/design/iter-slm491-exp-sr-3-external-blocked-closeout-20260810"
VERSION_COMPONENT = "harness.experiments.slm491_external_blocked_closeout"
RESULT_SCHEMA = "slm491_external_blocked_closeout/v1"

PREPARED_REPLAY = (
    "python -m scripts.run_sie006_construct_validity --mode external-blocked --seed 0"
)

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


def probe_environment() -> dict[str, Any]:
    """Record whether real external-judge transport or human raters are available."""
    external_judge_api_key_set = bool(os.environ.get("EXTERNAL_JUDGE_API_KEY"))
    human_rater_corpus_available = False
    return {
        "external_judge_api_key_set": external_judge_api_key_set,
        "real_external_judge_transport": external_judge_api_key_set,
        "real_human_raters_available": human_rater_corpus_available,
        "live_calibration_ready": external_judge_api_key_set and human_rater_corpus_available,
    }


def run_closeout(*, root: Path, seed: int = 0) -> dict[str, Any]:
    """Execute the prepared external-blocked package and return durable evidence."""
    env_probe = probe_environment()
    campaign = Sie006CampaignV1(seed=seed, sie005_frozen_doc=str(SIE005_FROZEN_DOC))
    sie006 = run_external_blocked_campaign(campaign, root=root / "sie006_external_blocked")

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
        "claim_class": "diagnostic",
        "evaluator_human_agreement_kappa": None,
        "external_human_calibration_error": None,
        "env_probe": env_probe,
        "prepared_package": {
            "sie005_frozen_doc": str(SIE005_FROZEN_DOC),
            "sie006_construct_validity_replay": PREPARED_REPLAY,
            "owner_module": "src/slm_training/evals/evaluator_calibration_protocol.py",
            "campaign_owner": "src/slm_training/harnesses/experiments/sie006_calibration_campaign.py",
        },
        "sie006_result": sie006,
        "scope_disclaimer": (
            "SLM-491 closes EXP-SR-3 follow-up as an honest external-blocked prepared "
            "package: verifies the frozen SIE-005 pin and SIE-006 replay surface, but "
            "does not claim construct-validity pass/fail on real humans. Kappa and "
            "calibration metrics remain null — never mocked-as-real."
        ),
        "timestamp": _now(),
        "recipe": {
            "command": ["python", "-m", "scripts.run_slm491_external_blocked_closeout"],
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
    sie006 = payload.get("sie006_result", {})
    lines = [
        f"# {LINEAR_ISSUE}: EXP-SR-3 external-blocked prepared-package closeout",
        "",
        f"**Status:** `{payload.get('status')}` / incomplete — not a construct-validity claim.",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id')}`",
        "",
        f"**Primary metric (`{PRIMARY_METRIC}`):** `{payload.get('primary_metric')}` = "
        f"`{payload.get('evaluator_human_agreement_kappa')}`",
        "",
        "## Environment probe",
        "",
        f"- EXTERNAL_JUDGE_API_KEY set: `{env.get('external_judge_api_key_set')}`",
        f"- Real human raters available: `{env.get('real_human_raters_available')}`",
        f"- Live calibration ready: `{env.get('live_calibration_ready')}`",
        "",
        "## Prepared package pointers",
        "",
        f"- SIE-005 frozen packet: `{prepared.get('sie005_frozen_doc')}`",
        f"- SIE-006 replay: `{prepared.get('sie006_construct_validity_replay')}`",
        f"- VCE-010 owner: `{prepared.get('owner_module')}`",
        "",
        "## SIE-006 external-blocked snapshot",
        "",
        f"- pair_n: `{sie006.get('pair_n')}`",
        f"- external_blocked: `{sie006.get('external_blocked')}`",
        f"- evaluator_human_agreement_kappa: `{sie006.get('evaluator_human_agreement_kappa')}`",
        "",
        "## Scope",
        "",
        payload.get("scope_disclaimer", ""),
        "",
        "Command: `python -m scripts.run_slm491_external_blocked_closeout`",
        "",
    ]
    return "\n".join(lines)


def load_closeout_summary(path: Path | None = None) -> dict[str, Any]:
    """Load SLM-491 closeout JSON for disposition supersession wiring."""
    target = path or Path(f"{DESIGN_STEM}.json")
    if not target.is_file():
        return {"present": False, "path": str(target)}
    data = json.loads(target.read_text(encoding="utf-8"))
    return {
        "present": True,
        "path": str(target),
        "status": data.get("status"),
        "external_blocked": data.get("external_blocked"),
        "evaluator_human_agreement_kappa": data.get("evaluator_human_agreement_kappa"),
        "evidence_cutoff_commit": data.get("evidence_cutoff_commit"),
    }
