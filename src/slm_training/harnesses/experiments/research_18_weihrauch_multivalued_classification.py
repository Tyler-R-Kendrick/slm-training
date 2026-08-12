"""RESEARCH-18 / SLM-551: anytime-valid promotion pilot campaign."""

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
from slm_training.formal.weihrauch_multivalued_classification import (
    CORPUS_RELATIVE,
    paired_classification_report,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-18"
LINEAR_ID = "SLM-551"
CAMPAIGN_ID = "research-18-weihrauch-multivalued-classification"
EXPERIMENT_ID = "research_18_weihrauch_multivalued_classification"
EVIDENCE_MD = "docs/design/iter-revmath-research-18-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-18-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_18_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_18"
PRIMARY_METRIC = "faithful_weihrauch_class_assignment_rate_on_frozen_task_set"
HARNESS_COMPONENT = "harness.experiments.slm551_research18_weihrauch"


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
        "arm": "control_total_computable",
        "corpus_relpath": CORPUS_RELATIVE,
        "default_off": True,
    }
    treatment_cfg = {
        "arm": "treatment_weihrauch",
        "corpus_relpath": CORPUS_RELATIVE,
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "Anytime-valid e-process evidence can govern adaptive autoresearch "
            "promotion without invalidating type-I control under optional stopping."
        ),
        "decision": (
            "Accept iff treatment type-I rate stays within alpha tolerance under "
            "null replay and planted-stream power remains usable vs fixed-n oracle; "
            "else reject/retire."
        ),
        "endpoints": [
            {
                "endpoint_id": "faithful_classification",
                "metric": PRIMARY_METRIC,
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 1.0,
            },
            {
                "endpoint_id": "big_five_inflation",
                "metric": "big_five_label_inflation",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
        ],
        "arms": [
            {
                "arm_id": "control_total_computable",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "treatment_weihrauch",
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
        "stopping_rules": [
            "frozen multivalued task corpus stops after paired classification"
        ],
        "controls": [
            {
                "control_id": "control_total_computable",
                "description": "Unclassified multi-valued tasks treated as ordinary total computable work.",
                "kind": "positive",
            },
            {
                "control_id": "big_five_inflation_fails_closed",
                "description": "Weihrauch labels must not auto-upgrade to Big-Five production claims.",
                "kind": "negative",
            },
        ],
        "negative_controls": ["big_five_inflation_fails_closed"],
        "multiplicity_families": [
            {
                "family_id": "research18_primary",
                "hypothesis_ids": ["faithful_classification"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "faithful_classification",
                "endpoint_id": "faithful_classification",
                "operator": "ge",
                "threshold": 1.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "big_five_inflation",
                "endpoint_id": "big_five_inflation",
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
        "author": "research-18",
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
            "schema": "research_18_experiment_result/v1",
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

    report = paired_classification_report(root=base)
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        HARNESS_COMPONENT,
    )
    return {
        "schema": "research_18_experiment_result/v1",
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
        "big_five_label_inflation": report["big_five_label_inflation"],
        "oracle_relative_smuggling": report["oracle_relative_smuggling"],
        "unknown_collapse_rate": report["unknown_collapse_rate"],
        "control_misclassifies_oracle_tasks": report["control_misclassifies_oracle_tasks"],
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
    return f"""# RESEARCH-18 — anytime-valid adaptive promotion (SLM-551)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Weihrauch/oracle classification of multi-valued proof/completion tasks
separates computable cores from oracle-relative remainders without Big-Five inflation.

## Contract

| Arm | Role |
| --- | --- |
| Total-computable collapse (control) | control |
| Weihrauch + practical classification | treatment |

| Gate | Result |
| --- | --- |
| Faithful Weihrauch assignment rate | {primary_s} |
| Big-Five label inflation count | {result.get("big_five_label_inflation")} |
| Oracle-relative smuggling count | {result.get("oracle_relative_smuggling")} |
| Unknown collapse rate | {result.get("unknown_collapse_rate")} |
| Control misclassifies oracle tasks | {result.get("control_misclassifies_oracle_tasks")} |
| Decision | **{decision}** |

Reason: `{result.get("reason")}`.

## Corpus

- Frozen simulation spec: `{CORPUS_RELATIVE}`
- Eval treatment tasks: {report.get("eval_treatment_count")}

## Campaign lock

- Manifest sha256: `{result.get("campaign_lock_sha256")}`
- Lock artifact: `{LOCK_RELPATH}`
- Claim class: `fixture` (research pilot; no production authority)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_18_weihrauch_multivalued_classification
SLM_ENABLE_RESEARCH_18=1 PYTHONPATH=src uv run python -m scripts.run_research_18_weihrauch_multivalued_classification --write
```

## Authority note

Fixture Weihrauch/oracle taxonomy only. Filing is not production authority.
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
