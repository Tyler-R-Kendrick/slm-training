"""RESEARCH-14 / SLM-572: checker diversity pilot campaign.

Default-off / research-only. Execution requires RESEARCH-02
``assert_execution_allowed`` (blockers complete + campaign lock digest).
"""

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
from slm_training.formal.checker_diversity_backend import (
    CONTROL_CHECKERS,
    CORPUS_RELATIVE,
    TREATMENT_CHECKERS,
    paired_diversity_report,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-14"
LINEAR_ID = "SLM-572"
CAMPAIGN_ID = "research-14-checker-diversity"
EXPERIMENT_ID = "research_14_checker_diversity"
EVIDENCE_MD = "docs/design/iter-revmath-research-14-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-14-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_14_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_14"
PRIMARY_METRIC = "unique_defect_detection_gain_vs_single_checker"
HARNESS_COMPONENT = "harness.experiments.slm572_research14_checker_diversity"


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
        "arm": "single_checker",
        "checkers": list(CONTROL_CHECKERS),
        "corpus_relpath": CORPUS_RELATIVE,
    }
    treatment_cfg = {
        "arm": "diverse_checkers",
        "checkers": list(TREATMENT_CHECKERS),
        "corpus_relpath": CORPUS_RELATIVE,
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "Checker diversity across independent fault domains increases "
            "defect detection versus a single checker family on frozen "
            "certificate corpora."
        ),
        "decision": (
            "Accept iff diverse checkers detect additional independent fault "
            "classes without false-alarm blow-up; else reject/retire."
        ),
        "endpoints": [
            {
                "endpoint_id": "unique_defect_gain",
                "metric": PRIMARY_METRIC,
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "false_alarm_rate",
                "metric": "false_alarm_rate",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "shared_blind_spot_residual",
                "metric": "shared_blind_spot_residual",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
        ],
        "arms": [
            {
                "arm_id": "single_checker_control",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "diverse_checkers_treatment",
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
                "control_id": "single_checker_family",
                "description": "One structural-family checker only.",
                "kind": "positive",
            },
            {
                "control_id": "false_alarm_fails_closed",
                "description": "Honest positives must not trigger false alarms.",
                "kind": "negative",
            },
        ],
        "negative_controls": ["false_alarm_fails_closed"],
        "multiplicity_families": [
            {
                "family_id": "research14_primary",
                "hypothesis_ids": ["unique_defect_gain"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "unique_gain",
                "endpoint_id": "unique_defect_gain",
                "operator": "gt",
                "threshold": 0.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "false_alarm",
                "endpoint_id": "false_alarm_rate",
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
        "author": "research-14",
        "created_at": created_at,
    }
    return ExperimentCampaignV1.model_validate(payload)


def lock_campaign(*, root: Path | None = None) -> CampaignLockV1:
    manifest = build_campaign_manifest(
        root=root, source_commit="0" * 40, source_dirty=False
    )
    digest = campaign_manifest_sha256(manifest)
    return CampaignLockV1(manifest_sha256=digest, manifest=manifest)


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
            "schema": "research_14_experiment_result/v1",
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

    report = paired_diversity_report(root=base)
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        HARNESS_COMPONENT,
    )
    return {
        "schema": "research_14_experiment_result/v1",
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
        "unique_fault_class_gain_count": report["unique_fault_class_gain_count"],
        "false_alarm_rate": report["false_alarm_rate"],
        "shared_blind_spot_residual": report["shared_blind_spot_residual"],
        "timeout_unknown_rate": report["timeout_unknown_rate"],
        "paired_report": report,
        "toolchain": pinned_toolchain(),
        "evidence_path": EVIDENCE_MD,
        "version_stamp": stamp,
    }


def render_evidence_markdown(result: dict[str, Any]) -> str:
    primary = result.get("primary_value")
    primary_s = "n/a" if primary is None else f"{primary:.6g}"
    decision = result.get("decision")
    report = result.get("paired_report") or {}
    return f"""# RESEARCH-14 — checker diversity (SLM-572)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Adding genuinely distinct checker implementations across independent EVID-08
trust domains increases seeded fault-class detection versus a single checker
family without false-alarm blow-up.

## Contract

| Arm | Role |
| --- | --- |
| Single structural-family checker | control |
| Structural + Lean kernel + encoding bridge | treatment |

| Gate | Result |
| --- | --- |
| Unique defect detection gain | {primary_s} |
| Unique fault-class gain count | {result.get("unique_fault_class_gain_count")} |
| Control detected classes | {report.get("control_detected_classes")} |
| Treatment detected classes | {report.get("treatment_detected_classes")} |
| False alarm rate | {result.get("false_alarm_rate")} |
| Shared blind-spot residual | {result.get("shared_blind_spot_residual")} |
| Decision | **{decision}** |

Reason: `{result.get("reason")}`.

## Corpus

- Frozen cases: `{CORPUS_RELATIVE}`
- EVID-08 trust domains: structural, lean4_kernel, encoding_bridge
- EVID-11-aligned seeded fault families (fixture simulation)

## Campaign lock

- Manifest sha256: `{result.get("campaign_lock_sha256")}`
- Lock artifact: `{LOCK_RELPATH}`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_14_checker_diversity
SLM_ENABLE_RESEARCH_14=1 PYTHONPATH=src uv run python -m scripts.run_research_14_checker_diversity --write
```

## Authority note

Fixture trust-domain simulation only — no mandatory external checker dependency.
Filing is not production readiness.
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


__all__ = [
    "CAMPAIGN_ID",
    "ENABLE_FLAG",
    "EVIDENCE_JSON",
    "EVIDENCE_MD",
    "EXPERIMENT_ID",
    "EXPERIMENT_KEY",
    "HARNESS_COMPONENT",
    "LINEAR_ID",
    "LOCK_RELPATH",
    "PRIMARY_METRIC",
    "build_campaign_manifest",
    "load_campaign_lock",
    "lock_campaign",
    "patch_preregistry",
    "render_evidence_markdown",
    "run_experiment",
    "write_campaign_lock",
    "write_evidence",
]
