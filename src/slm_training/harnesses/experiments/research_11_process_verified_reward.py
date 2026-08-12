"""RESEARCH-11 / SLM-574: process-verified reward shaping pilot campaign."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    CampaignLockV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.formal.process_verified_reward import (
    CORPUS_RELATIVE,
    paired_reward_report,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-11"
LINEAR_ID = "SLM-574"
CAMPAIGN_ID = "research-11-process-verified-reward"
EXPERIMENT_ID = "research_11_process_verified_reward"
EVIDENCE_MD = "docs/design/iter-revmath-research-11-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-11-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_11_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_11"
PRIMARY_METRIC = "verified_process_reward_correlation_with_final_proof_success"
HARNESS_COMPONENT = "harness.experiments.slm574_research11_process_reward"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_commit(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
        if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
            return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        pass
    return "0" * 40, True


def _config_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_campaign_manifest(
    *,
    root: Path | None = None,
    created_at: str = "1970-01-01T00:00:00Z",
    source_commit: str | None = None,
    source_dirty: bool | None = None,
) -> ExperimentCampaignV1:
    base = root or _repo_root()
    commit, dirty = _source_commit(base)
    if source_commit is not None:
        commit = source_commit
    if source_dirty is not None:
        dirty = source_dirty
    control_cfg = {
        "arm": "terminal_only_reward",
        "corpus_relpath": CORPUS_RELATIVE,
    }
    treatment_cfg = {
        "arm": "process_verified_reward",
        "corpus_relpath": CORPUS_RELATIVE,
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "Process-verified reward shaping improves correlation with final "
            "proof success beyond terminal-only rewards without rewarding "
            "unverifiable intermediate states."
        ),
        "decision": (
            "Accept iff process reward correlation exceeds terminal-only "
            "with zero leakage violations, zero reward hacking, and zero "
            "unverifiable intermediate inflation; else reject/retire."
        ),
        "endpoints": [
            {
                "endpoint_id": "process_correlation",
                "metric": PRIMARY_METRIC,
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "unverifiable_rate",
                "metric": "unverifiable_intermediate_reward_rate",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
        ],
        "arms": [
            {
                "arm_id": "terminal_control",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "process_verified_treatment",
                "role": "candidate",
                "config_sha256": _config_sha(treatment_cfg),
            },
        ],
        "seeds": [0],
        "budget": {
            "max_experiments": 1,
            "max_wall_minutes": 30.0,
            "max_gpu_hours": 0.0,
        },
        "stopping_rules": ["single fixture corpus stop after paired report"],
        "controls": [
            {
                "control_id": "terminal_baseline",
                "description": "Terminal-only proof success reward.",
                "kind": "positive",
            },
            {
                "control_id": "leakage_guard",
                "description": "Label metadata must not influence process rewards.",
                "kind": "negative",
            },
        ],
        "negative_controls": ["leakage_guard"],
        "multiplicity_families": [
            {
                "family_id": "research11_primary",
                "hypothesis_ids": ["process_correlation"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "correlation_gain",
                "endpoint_id": "process_correlation",
                "operator": "gt",
                "threshold": 0.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "unverifiable_rate",
                "endpoint_id": "unverifiable_rate",
                "operator": "gt",
                "threshold": 0.0,
            }
        ],
        "artifact_requirements": [
            {"kind": "version_stamp", "minimum_count": 1},
            {"kind": "paired_examples", "minimum_count": 1},
        ],
        "claim_class": "fixture",
        "source_commit": commit,
        "source_dirty": dirty,
        "author": "research-11",
        "created_at": created_at,
    }
    return ExperimentCampaignV1.model_validate(payload)


def lock_campaign(*, root: Path | None = None) -> CampaignLockV1:
    manifest = build_campaign_manifest(
        root=root, source_commit="0" * 40, source_dirty=False
    )
    return CampaignLockV1(
        manifest_sha256=campaign_manifest_sha256(manifest), manifest=manifest
    )


def write_campaign_lock(path: Path, *, root: Path | None = None) -> CampaignLockV1:
    lock = lock_campaign(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock


def load_campaign_lock(path: Path) -> CampaignLockV1:
    return CampaignLockV1.model_validate_json(path.read_text(encoding="utf-8"))


def run_experiment(*, root: Path | None = None, enabled: bool = False) -> dict[str, Any]:
    base = root or _repo_root()
    if not enabled:
        return {
            "schema": "research_11_experiment_result/v1",
            "experiment_key": EXPERIMENT_KEY,
            "linear_id": LINEAR_ID,
            "default_off": True,
            "executed": False,
            "decision": "skipped_default_off",
            "reason": f"pass enabled=True or set {ENABLE_FLAG}=1",
        }

    row = experiment_by_key(EXPERIMENT_KEY, root=base)
    assert_execution_allowed(row)
    lock_path = base / LOCK_RELPATH
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing campaign lock: {lock_path}")
    lock = load_campaign_lock(lock_path)
    if row.get("campaign_lock_sha256") != lock.manifest_sha256:
        raise RuntimeError(
            f"{EXPERIMENT_KEY}: registry campaign_lock_sha256 mismatch vs {LOCK_RELPATH}"
        )

    report = paired_reward_report(root=base)
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        HARNESS_COMPONENT,
    )
    return {
        "schema": "research_11_experiment_result/v1",
        "experiment_key": EXPERIMENT_KEY,
        "linear_id": LINEAR_ID,
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "default_off": True,
        "research_only": True,
        "executed": True,
        "campaign_lock_sha256": lock.manifest_sha256,
        "claim_class": "fixture",
        "decision": report["decision"],
        "reason": report["reason"],
        "primary_metric": PRIMARY_METRIC,
        "primary_value": report[PRIMARY_METRIC],
        "terminal_reward_correlation": report["terminal_reward_correlation"],
        "correlation_gain": report["correlation_gain"],
        "unverifiable_intermediate_reward_rate": report["unverifiable_intermediate_reward_rate"],
        "reward_hacking_incidents": report["reward_hacking_incidents"],
        "paired_report": report,
        "toolchain": pinned_toolchain(),
        "evidence_path": EVIDENCE_MD,
        "version_stamp": stamp,
    }


def render_evidence_markdown(result: dict[str, Any]) -> str:
    primary = result.get("primary_value")
    primary_s = "n/a" if primary is None else f"{primary:.6g}"
    decision = result.get("decision")
    return f"""# RESEARCH-11 — process-verified reward shaping (SLM-574)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Verifier-grounded process rewards from checked tactic prefixes and failure
localization improve learning signal correlation vs terminal-only rewards.

## Contract

| Arm | Role |
| --- | --- |
| Terminal-only proof success reward | control |
| Process-verified dense shaping | treatment |

| Gate | Result |
| --- | --- |
| Process reward correlation | {primary_s} |
| Correlation gain vs terminal | {result.get("correlation_gain")} |
| Unverifiable intermediate rate | {result.get("unverifiable_intermediate_reward_rate")} |
| Reward hacking incidents | {result.get("reward_hacking_incidents")} |
| Decision | **{decision}** |

Reason: `{result.get("reason")}`.

## Corpus

- Frozen proof attempts: `{CORPUS_RELATIVE}`

## Campaign lock

- Manifest sha256: `{result.get("campaign_lock_sha256")}`
- Lock artifact: `{LOCK_RELPATH}`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_11_process_reward
SLM_ENABLE_RESEARCH_11=1 PYTHONPATH=src uv run python -m scripts.run_research_11_process_reward --write
```

## Authority note

Fixture reward simulation only. Filing is not production readiness.
"""


def write_evidence(result: dict[str, Any], *, root: Path | None = None) -> None:
    base = root or _repo_root()
    (base / EVIDENCE_JSON).write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (base / EVIDENCE_MD).write_text(render_evidence_markdown(result), encoding="utf-8")


def patch_preregistry(*, root: Path | None = None, lock_sha: str, disposition: str) -> None:
    base = root or _repo_root()
    path = base / "src/slm_training/resources/research_experiment_preregistry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["experiments"]:
        if row.get("experiment_key") != EXPERIMENT_KEY:
            continue
        row["campaign_lock_sha256"] = lock_sha
        row["disposition"] = disposition
        break
    else:
        raise RuntimeError(f"{EXPERIMENT_KEY} missing from preregistry")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
