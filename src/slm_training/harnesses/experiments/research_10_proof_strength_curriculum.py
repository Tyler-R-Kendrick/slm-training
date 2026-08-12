"""RESEARCH-10 / SLM-566: proof-strength curriculum pilot campaign.

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
from slm_training.formal.proof_strength_curriculum import (
    CORPUS_RELATIVE,
    paired_curriculum_report,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-10"
LINEAR_ID = "SLM-566"
CAMPAIGN_ID = "research-10-proof-strength-curriculum"
EXPERIMENT_ID = "research_10_proof_strength_curriculum"
EVIDENCE_MD = "docs/design/iter-revmath-research-10-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-10-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_10_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_10"
PRIMARY_METRIC = "sample_efficiency_to_target_obligation_pass_rate"
HARNESS_COMPONENT = "harness.experiments.slm566_research10_proof_strength"


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
        "arm": "unordered_mixed_strength",
        "ordering": "unordered",
        "corpus_relpath": CORPUS_RELATIVE,
    }
    treatment_cfg = {
        "arm": "obligation_difficulty_curriculum",
        "ordering": "obligation_difficulty",
        "corpus_relpath": CORPUS_RELATIVE,
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "A proof-strength curriculum ordered by formal obligation difficulty "
            "improves Lean training sample efficiency versus unordered mixing."
        ),
        "decision": (
            "Accept iff treatment eval pass rate exceeds control under fixed "
            "budget with zero train/eval root leakage and no easy-task regression; "
            "else reject/retire."
        ),
        "endpoints": [
            {
                "endpoint_id": "sample_efficiency",
                "metric": PRIMARY_METRIC,
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 1.0,
            },
            {
                "endpoint_id": "easy_task_regression",
                "metric": "easy_task_regression",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
        ],
        "arms": [
            {
                "arm_id": "unordered_control",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "curriculum_treatment",
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
                "control_id": "unordered_matched_budget",
                "description": "Unordered mixed-strength schedule with matched step budget.",
                "kind": "positive",
            },
            {
                "control_id": "easy_task_regression_fails_closed",
                "description": (
                    "Curriculum must not regress easy obligations (difficulty <= 2) "
                    "that gold-pass under the fixture budget."
                ),
                "kind": "negative",
            },
        ],
        "negative_controls": [
            "easy_task_regression_fails_closed",
        ],
        "multiplicity_families": [
            {
                "family_id": "research10_primary",
                "hypothesis_ids": ["sample_efficiency"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "efficiency_gain",
                "endpoint_id": "sample_efficiency",
                "operator": "gt",
                "threshold": 1.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "easy_regression",
                "endpoint_id": "easy_task_regression",
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
        "author": "research-10",
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
            "schema": "research_10_experiment_result/v1",
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

    report = paired_curriculum_report(root=base)
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        HARNESS_COMPONENT,
    )
    return {
        "schema": "research_10_experiment_result/v1",
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
        "control_eval_pass_rate": report["control_eval_pass_rate"],
        "treatment_eval_pass_rate": report["treatment_eval_pass_rate"],
        "easy_task_regression": report["easy_task_regression"],
        "train_eval_root_leakage": report["train_eval_root_leakage"],
        "paired_report": report,
        "toolchain": pinned_toolchain(),
        "evidence_path": EVIDENCE_MD,
        "version_stamp": stamp,
    }


def render_evidence_markdown(result: dict[str, Any]) -> str:
    primary = result.get("primary_value")
    primary_s = "n/a" if primary is None else f"{primary:.6g}"
    decision = result.get("decision")
    return f"""# RESEARCH-10 — proof-strength curriculum (SLM-566)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

A proof-strength curriculum ordered by formal obligation difficulty improves
sample efficiency versus unordered mixing on dependency-disjoint eval groups.

## Contract

| Arm | Role |
| --- | --- |
| Unordered mixed-strength schedule (matched budget) | control |
| Obligation-difficulty curriculum | treatment |

| Gate | Result |
| --- | --- |
| Sample efficiency ratio (treatment/control) | {primary_s} |
| Control eval pass rate | {result.get("control_eval_pass_rate")} |
| Treatment eval pass rate | {result.get("treatment_eval_pass_rate")} |
| Train/eval root leakage | {result.get("train_eval_root_leakage")} |
| Easy-task regression | {result.get("easy_task_regression")} |
| Decision | **{decision}** |

Reason: `{result.get("reason")}`.

## Corpus

- Frozen records: `{CORPUS_RELATIVE}`
- Obligation kinds: original, ablation, reversal, constructivization, counterexample, bound
- HARN-11-style root-family isolation between train and eval

## Campaign lock

- Manifest sha256: `{result.get("campaign_lock_sha256")}`
- Lock artifact: `{LOCK_RELPATH}`
- Claim class: `fixture` (research pilot; no promotion)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_10_proof_strength
SLM_ENABLE_RESEARCH_10=1 PYTHONPATH=src uv run python -m scripts.run_research_10_proof_strength --write
```

## Authority note

Fixture schedule proxy only — no trainer invoked. Filing is not production readiness.
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
