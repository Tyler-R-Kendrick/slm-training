"""RESEARCH-09 / SLM-541: preregistered treewidth-aware VSS campaign.

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
from slm_training.formal.vss_treewidth_backend import (
    PROOF_FORMAT_VERSION,
    default_param_suite,
    paired_param_report,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-09"
LINEAR_ID = "SLM-541"
CAMPAIGN_ID = "research-09-vss-treewidth"
EXPERIMENT_ID = "research_09_vss_treewidth"
EVIDENCE_MD = "docs/design/iter-revmath-research-09-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-09-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_09_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_09"
PRIMARY_METRIC = "rank_correlation_of_predicted_vs_observed_solver_cost"
HARNESS_COMPONENT = "harness.experiments.slm541_research09_vss_treewidth"


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
    control_cfg = {
        "arm": "flat_size_proxy_plus_enumerative",
        "backend": "encoding_adapter_enumerative",
        "proxy": "product_domain_x_clauses",
    }
    treatment_cfg = {
        "arm": "vss_treewidth_param",
        "backend": "treewidth_proxy_bag_dp",
        "toolchain": pinned_toolchain(),
        "proof_format_version": PROOF_FORMAT_VERSION,
        "feature": "treewidth_proxy",
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "Parameterized/treewidth-aware VSS complexity bounds predict "
            "practical solver cost better than flat instance size on a "
            "frozen suite."
        ),
        "decision": (
            "Accept iff treewidth-proxy Spearman correlation exceeds the flat "
            "size proxy on the frozen suite, witness disagreements stay 0, "
            "timeout-as-refutation stays 0, and mutation rejection is 1.0; "
            "else reject/retire (honest null if params do not help)."
        ),
        "endpoints": [
            {
                "endpoint_id": "rank_correlation",
                "metric": PRIMARY_METRIC,
                "role": "primary",
                "direction": "increase",
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
                "endpoint_id": "timeout_as_refutation_count",
                "metric": "timeout_as_refutation_count",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "parameter_estimation_failures",
                "metric": "parameter_estimation_failures",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
        ],
        "arms": [
            {
                "arm_id": "flat_size_proxy",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "treewidth_proxy",
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
            "single fixture suite stop after paired param report",
        ],
        "controls": [
            {
                "control_id": "flat_size_proxy",
                "description": (
                    "Flat product-domain × clause-count cost proxy paired with "
                    "production enumerative outcomes on the identical suite."
                ),
                "kind": "positive",
            },
            {
                "control_id": "unsupported_stays_unknown",
                "description": (
                    "Outside the preregistered tw/n support caps, treatment "
                    "must stay unknown (never invent refutation)."
                ),
                "kind": "negative",
            },
            {
                "control_id": "clause_polarity_flip",
                "description": (
                    "Mutated clause polarity must re-solve; no stale unsat "
                    "authority from the original instance."
                ),
                "kind": "negative",
            },
            {
                "control_id": "empty_problem_unknown",
                "description": "Empty problem must stay unknown, never unsat.",
                "kind": "negative",
            },
        ],
        "negative_controls": [
            "unsupported_stays_unknown",
            "clause_polarity_flip",
            "empty_problem_unknown",
        ],
        "multiplicity_families": [
            {
                "family_id": "research09_primary",
                "hypothesis_ids": ["rank_correlation"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "correlation_improves",
                "endpoint_id": "rank_correlation",
                "operator": "gt",
                "threshold": 0.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "any_disagreement",
                "endpoint_id": "witness_disagreement_count",
                "operator": "gt",
                "threshold": 0.0,
            },
            {
                "gate_id": "any_fake_refutation",
                "endpoint_id": "timeout_as_refutation_count",
                "operator": "gt",
                "threshold": 0.0,
            },
        ],
        "artifact_requirements": [
            {"kind": "version_stamp", "minimum_count": 1},
            {"kind": "paired_examples", "minimum_count": 1},
        ],
        "claim_class": "fixture",
        "source_commit": commit,
        "source_dirty": dirty,
        "author": "research-09",
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
) -> dict[str, Any]:
    """Run RESEARCH-09. Refuses unless explicitly enabled (default-off)."""

    base = root or _repo_root()
    if not enabled:
        return {
            "schema": "research_09_experiment_result/v1",
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

    report = paired_param_report(default_param_suite())
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        HARNESS_COMPONENT,
    )
    payload = {
        "schema": "research_09_experiment_result/v1",
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
        "primary_metric": PRIMARY_METRIC,
        "primary_value": report[PRIMARY_METRIC],
        "correctness_ok": report["correctness_ok"],
        "correlation_improves_vs_flat": report["correlation_improves_vs_flat"],
        "rank_correlation_flat_size_proxy": report["rank_correlation_flat_size_proxy"],
        "rank_correlation_treewidth_proxy": report["rank_correlation_treewidth_proxy"],
        "witness_disagreement_count": report["witness_disagreement_count"],
        "timeout_as_refutation_count": report["timeout_as_refutation_count"],
        "parameter_estimation_failures": report["parameter_estimation_failures"],
        "miscalibration_rate": report["miscalibration_rate"],
        "mutation_rejection_rate": report["mutation_rejection_rate"],
        "proof_format_version": report["proof_format_version"],
        "paired_report": report,
        "toolchain": pinned_toolchain(),
        "evidence_path": EVIDENCE_MD,
        "version_stamp": stamp,
    }
    return payload


def render_evidence_markdown(result: dict[str, Any]) -> str:
    rate = result.get("primary_value")
    rate_s = "n/a" if rate is None else f"{rate:.6g}"
    flat = result.get("rank_correlation_flat_size_proxy")
    flat_s = "n/a" if flat is None else f"{flat:.6g}"
    decision = result.get("decision")
    return f"""# RESEARCH-09 — parameterized / treewidth-aware VSS (SLM-541)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Parameterized/treewidth-aware VSS complexity bounds predict practical solver
cost better than flat instance size on a frozen suite.

## Contract

| Arm | Role |
| --- | --- |
| Flat size / clause-count cost proxy + enumerative solver | matched control |
| Treewidth / binder / ambiguity / residual-class instrumentation + bag DP on supported subset | treatment |

| Gate | Result |
| --- | --- |
| Treewidth-proxy Spearman ρ (primary) | {rate_s} |
| Flat-size-proxy Spearman ρ | {flat_s} |
| Correlation improves vs flat | {result.get("correlation_improves_vs_flat")} |
| Witness disagreements | {result.get("witness_disagreement_count")} |
| Timeout→fake refutation count | {result.get("timeout_as_refutation_count")} |
| Parameter estimation failures | {result.get("parameter_estimation_failures")} |
| Mutation rejection rate | {result.get("mutation_rejection_rate")} |
| Decision | **{decision}** |

Decision rule: treewidth-proxy correlation **strictly exceeds** flat-size
proxy, with zero witness disagreements, zero timeout-as-refutation, and
mutation rejection == 1.0; else **reject** (honest null if params do not help).
Unsupported / incomplete → `unknown` (never `refuted`/`unsat`).

## Declared support + proof format

- Supported subset: exact treewidth ≤ 2 and n ≤ 8 (bool domains)
- Proof / instrumentation format: `{result.get("proof_format_version") or PROOF_FORMAT_VERSION}`
- Production enumerative solver unchanged (`encoding_adapter.exists_satisfying_assignment`)

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
| Results JSON | [`iter-revmath-research-09-preregistered.json`](iter-revmath-research-09-preregistered.json) |
| Adapter | `src/slm_training/formal/vss_treewidth_backend.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_09_vss_treewidth.py` |

## Run

```bash
# default-off: refuses without enable
PYTHONPATH=src uv run python -m scripts.run_research_09_vss_treewidth
SLM_ENABLE_RESEARCH_09=1 PYTHONPATH=src uv run python -m scripts.run_research_09_vss_treewidth --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Treewidth instrumentation remains research-only; production refutation
authority stays EVID-09 exhaustive replay / checked certificate policy.
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
    "HARNESS_COMPONENT",
    "LINEAR_ID",
    "LOCK_RELPATH",
    "PRIMARY_METRIC",
    "build_campaign_manifest",
    "load_campaign_lock",
    "lock_campaign",
    "render_evidence_markdown",
    "run_experiment",
    "write_campaign_lock",
    "write_evidence",
]
