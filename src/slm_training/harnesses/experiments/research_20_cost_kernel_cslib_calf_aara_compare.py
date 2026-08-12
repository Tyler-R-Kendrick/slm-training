"""RESEARCH-20 / SLM-552: local cost kernel vs CSLib/Calf/AARA compare pilot."""

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
from slm_training.formal.cost_kernel_cslib_calf_aara_compare import (
    CORPUS_RELATIVE,
    paired_compare_report,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-20"
LINEAR_ID = "SLM-552"
CAMPAIGN_ID = "research-20-cost-kernel-cslib-calf-aara-compare"
EXPERIMENT_ID = "research_20_cost_kernel_cslib_calf_aara_compare"
EVIDENCE_MD = "docs/design/iter-revmath-research-20-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-20-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_20_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_20"
PRIMARY_METRIC = "transferability_disagreement_rate_across_cost_formalisms"
HARNESS_COMPONENT = "harness.experiments.slm552_research20_cost_compare"


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
        "arm": "control_local_cost_kernel",
        "corpus_relpath": CORPUS_RELATIVE,
        "default_off": True,
    }
    treatment_cfg = {
        "arm": "treatment_external_formalisms",
        "corpus_relpath": CORPUS_RELATIVE,
        "formalisms": ["cslib", "calf", "aara"],
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "Comparing the local cost kernel against CSLib/Calf/AARA clarifies "
            "which cost claims transfer and which remain repository-local heuristics."
        ),
        "decision": (
            "Accept a comparison report only if external formalisms reduce LOC or "
            "enable new bound classes with explicit transfer theorems; else reject."
        ),
        "endpoints": [
            {
                "endpoint_id": "transferability",
                "metric": PRIMARY_METRIC,
                "role": "primary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "maintenance_cost",
                "metric": "maintenance_cost_loc_burden_delta",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "silent_equivalence",
                "metric": "silent_equivalence_claims",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
        ],
        "arms": [
            {
                "arm_id": "control_local_cost_kernel",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "treatment_external_formalisms",
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
            "frozen cost fixture corpus stops after paired formalism comparison"
        ],
        "controls": [
            {
                "control_id": "control_local_cost_kernel",
                "description": "Event-trace local cost kernel without external formalism arms.",
                "kind": "positive",
            },
            {
                "control_id": "silent_equivalence_fails_closed",
                "description": "External formalisms must not claim equivalence without transfer theorems.",
                "kind": "negative",
            },
        ],
        "negative_controls": ["silent_equivalence_fails_closed"],
        "multiplicity_families": [
            {
                "family_id": "research20_primary",
                "hypothesis_ids": ["transferability"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "transferability",
                "endpoint_id": "transferability",
                "operator": "le",
                "threshold": 0.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "silent_equivalence",
                "endpoint_id": "silent_equivalence",
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
        "author": "research-20",
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
            "schema": "research_20_experiment_result/v1",
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

    report = paired_compare_report(root=base)
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        HARNESS_COMPONENT,
    )
    return {
        "schema": "research_20_experiment_result/v1",
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
        "silent_equivalence_claims": report["silent_equivalence_claims"],
        "incomparable_metric_smuggling": report["incomparable_metric_smuggling"],
        "maintenance_cost_loc_burden_delta": report["maintenance_cost_loc_burden_delta"],
        "new_bound_classes_enabled": report["new_bound_classes_enabled"],
        "loc_reduced_vs_local_kernel": report["loc_reduced_vs_local_kernel"],
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
    return f"""# RESEARCH-20 — local cost kernel vs CSLib/Calf/AARA (SLM-552)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Comparing the local event-trace cost kernel against CSLib/Calf/AARA clarifies
which cost claims transfer and which remain repository-local heuristics.

## Contract

| Arm | Role |
| --- | --- |
| Local event-trace cost kernel | control |
| Hermetic CSLib/Calf/AARA adapters | treatment |

| Gate | Result |
| --- | --- |
| Transferability disagreement rate | {primary_s} |
| Silent equivalence claims | {result.get("silent_equivalence_claims")} |
| Incomparable metric smuggling | {result.get("incomparable_metric_smuggling")} |
| LOC burden delta vs local kernel | {result.get("maintenance_cost_loc_burden_delta")} |
| New bound classes enabled | {result.get("new_bound_classes_enabled")} |
| LOC reduced vs local | {result.get("loc_reduced_vs_local_kernel")} |
| Decision | **{decision}** |

Reason: `{result.get("reason")}`.

## Corpus

- Frozen simulation spec: `{CORPUS_RELATIVE}`
- Eval fixtures: {report.get("eval_fixture_count")}

## Campaign lock

- Manifest sha256: `{result.get("campaign_lock_sha256")}`
- Lock artifact: `{LOCK_RELPATH}`
- Claim class: `fixture` (research pilot; no merge without transfer theorems)

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_20_cost_compare
SLM_ENABLE_RESEARCH_20=1 PYTHONPATH=src uv run python -m scripts.run_research_20_cost_compare --write
```

## Authority note

Hermetic formalism comparison only. External CSLib/Calf/AARA are not imported.
This rejection closes the **external-formalism merge** approach on frozen cost
fixtures: dependency/proof burden rises without LOC reduction or new bound classes.
Local event-trace kernel remains authoritative for represented counters.
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
