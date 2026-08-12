"""RESEARCH-05 / SLM-563: preregistered VSS LRAT SAT pilot campaign.

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
from slm_training.formal.vss_lrat_backend import (
    default_unsat_suite,
    paired_warm_trials,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-05"
LINEAR_ID = "SLM-563"
CAMPAIGN_ID = "research-05-vss-lrat"
EXPERIMENT_ID = "research_05_vss_lrat_sat"
EVIDENCE_MD = "docs/design/iter-revmath-research-05-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-05-preregistered-results.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_05_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_05"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_commit(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
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
    """Preregistered fixture/diagnostic campaign (locked before outcomes)."""

    base = root or _repo_root()
    commit, dirty = _source_commit(base)
    if source_commit is not None:
        commit = source_commit
    if source_dirty is not None:
        dirty = source_dirty
    control_cfg = {"arm": "exhaustive_vss_replay", "backend": "python_enumerative"}
    treatment_cfg = {
        "arm": "vss_lrat_sat",
        "backend": "lrat_pilot",
        "toolchain": pinned_toolchain(),
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "For a declared bounded VSS subset, verified VSS-to-CNF plus checked "
            "LRAT preserves exact agreement while making warm refutation cheaper "
            "than exhaustive replay."
        ),
        "decision": (
            "Accept iff zero witness disagreements, 100% mutation rejection, and "
            "median paired warm LRAT/exhaustive ratio < 1.0; else reject/retire."
        ),
        "endpoints": [
            {
                "endpoint_id": "warm_lrat_over_exhaustive_ratio",
                "metric": "median_paired_warm_lrat_check_over_exhaustive_replay_time_ratio",
                "role": "primary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "witness_disagreement_count",
                "metric": "witness_disagreement_count",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "mutation_rejection_rate",
                "metric": "mutation_rejection_rate",
                "role": "secondary",
                "direction": "increase",
                "minimum_effect": 0.0,
            },
        ],
        "arms": [
            {
                "arm_id": "exhaustive_replay",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "lrat_pilot",
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
            "single fixture suite stop after paired warm/exhaustive report",
        ],
        "controls": [
            {
                "control_id": "exhaustive_replay",
                "description": "Canonical exhaustive VSS replay on identical instances.",
                "kind": "positive",
            },
            {
                "control_id": "encoding_flip_literal",
                "description": "Encoding polarity mutation must not authorize unsat.",
                "kind": "negative",
            },
            {
                "control_id": "certificate_hint_corrupt",
                "description": "Corrupt LRAT hints must not authorize unsat.",
                "kind": "negative",
            },
        ],
        "negative_controls": ["encoding_flip_literal", "certificate_hint_corrupt"],
        "multiplicity_families": [
            {
                "family_id": "research05_primary",
                "hypothesis_ids": ["warm_lrat_over_exhaustive_ratio"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "ratio_lt_one",
                "endpoint_id": "warm_lrat_over_exhaustive_ratio",
                "operator": "lt",
                "threshold": 1.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "any_disagreement",
                "endpoint_id": "witness_disagreement_count",
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
        "author": "research-05",
        "created_at": created_at,
    }
    return ExperimentCampaignV1.model_validate(payload)


def lock_campaign(*, root: Path | None = None) -> CampaignLockV1:
    # Fixture lock uses a stable source identity so the digest is independent of
    # transient worktree dirtiness; claim_class remains fixture/research-only.
    manifest = build_campaign_manifest(
        root=root,
        source_commit="0" * 40,
        source_dirty=False,
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


def run_experiment(
    *,
    root: Path | None = None,
    enabled: bool = False,
    warm_repeats: int = 5,
) -> dict[str, Any]:
    """Run RESEARCH-05. Refuses unless explicitly enabled (default-off)."""

    base = root or _repo_root()
    if not enabled:
        return {
            "schema": "research_05_experiment_result/v1",
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

    report = paired_warm_trials(default_unsat_suite(), warm_repeats=warm_repeats)
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        "formal.objects",
    )
    payload = {
        "schema": "research_05_experiment_result/v1",
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
        "primary_metric": "median_paired_warm_lrat_check_over_exhaustive_replay_time_ratio",
        "primary_value": report[
            "median_paired_warm_lrat_check_over_exhaustive_replay_time_ratio"
        ],
        "correctness_ok": report["correctness_ok"],
        "witness_disagreement_count": report["witness_disagreement_count"],
        "mutation_rejection_rate": report["mutation_rejection_rate"],
        "supported_subset_coverage": report["supported_subset_coverage"],
        "paired_report": report,
        "toolchain": pinned_toolchain(),
        "evidence_path": EVIDENCE_MD,
        "version_stamp": stamp,
    }
    return payload


def render_evidence_markdown(result: dict[str, Any]) -> str:
    ratio = result.get("primary_value")
    ratio_s = "n/a" if ratio is None else f"{ratio:.6g}"
    decision = result.get("decision")
    return f"""# RESEARCH-05 — VSS SAT backend with LRAT (SLM-563)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

For a declared bounded VSS subset, verified VSS→CNF + checked LRAT preserves
exact semantic agreement / mutation rejection while making warm LRAT-check
cheaper than exhaustive replay.

## Contract

| Arm | Role |
| --- | --- |
| Exhaustive VSS/CNF replay | matched control |
| Deterministic CNF + LRAT via EVID-10 `lrat_pilot` | treatment |

| Gate | Result |
| --- | --- |
| Witness disagreements | {result.get("witness_disagreement_count")} |
| Mutation rejection rate | {result.get("mutation_rejection_rate")} |
| Supported-subset coverage | {result.get("supported_subset_coverage")} |
| Median warm LRAT / exhaustive ratio | {ratio_s} |
| Decision | **{decision}** |

Decision rule: correctness gates pass **and** ratio < 1.0; else reject/retire.
Timeout / tool failure / unsupported → `unknown` (candidates preserved).

## Campaign lock

- Manifest sha256: `{result.get("campaign_lock_sha256")}`
- Lock artifact: `{LOCK_RELPATH}`
- Claim class: `fixture` (research pilot; no promotion)

## Toolchain pin

```json
{json.dumps(result.get("toolchain") or pinned_toolchain(), indent=2, sort_keys=True)}
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Results JSON | [`iter-revmath-research-05-preregistered-results.json`](iter-revmath-research-05-preregistered-results.json) |
| Adapter | `src/slm_training/formal/vss_lrat_backend.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_05_vss_lrat.py` |
| Encoding bridge | `src/slm_training/formal/encoding_adapter.py` (EVID-10) |

## Run

```bash
# default-off: refuses without enable
PYTHONPATH=src uv run python -m scripts.run_research_05_vss_lrat
SLM_ENABLE_RESEARCH_05=1 PYTHONPATH=src uv run python -m scripts.run_research_05_vss_lrat --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Warm LRAT checking remains research-only; production refutation authority
stays EVID-09 exhaustive replay / checked certificate policy.
"""


def write_evidence(result: dict[str, Any], *, root: Path | None = None) -> None:
    base = root or _repo_root()
    json_path = base / EVIDENCE_JSON
    md_path = base / EVIDENCE_MD
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_evidence_markdown(result), encoding="utf-8")


__all__ = [
    "CAMPAIGN_ID",
    "ENABLE_FLAG",
    "EVIDENCE_JSON",
    "EVIDENCE_MD",
    "EXPERIMENT_ID",
    "EXPERIMENT_KEY",
    "LINEAR_ID",
    "LOCK_RELPATH",
    "build_campaign_manifest",
    "load_campaign_lock",
    "lock_campaign",
    "render_evidence_markdown",
    "run_experiment",
    "write_campaign_lock",
    "write_evidence",
]
