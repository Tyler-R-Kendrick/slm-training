"""RESEARCH-07 / SLM-565: preregistered VSS Alethe/SMT pilot campaign.

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
from slm_training.formal.vss_alethe_backend import (
    PROOF_FORMAT_VERSION,
    SUPPORTED_SMT_THEORIES,
    default_theory_suite,
    paired_agreement_report,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-07"
LINEAR_ID = "SLM-565"
CAMPAIGN_ID = "research-07-vss-alethe-smt"
EXPERIMENT_ID = "research_07_vss_alethe_smt"
EVIDENCE_MD = "docs/design/iter-revmath-research-07-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-07-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_07_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_07"
PRIMARY_METRIC = "exact_agreement_rate_on_supported_theory_subset"


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
        "arm": "exhaustive_theory_replay",
        "backend": "python_enumerative",
        "sat_overlap": True,
    }
    treatment_cfg = {
        "arm": "vss_alethe_smt",
        "backend": "alethe_pilot",
        "toolchain": pinned_toolchain(),
        "supported_smt_theories": sorted(SUPPORTED_SMT_THEORIES),
        "proof_format_version": PROOF_FORMAT_VERSION,
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "Alethe/SMT proof reconstruction for theory-rich VSS constraints "
            "preserves exact agreement on a declared theory subset without "
            "widening unknown into refutation."
        ),
        "decision": (
            "Accept iff exact agreement rate on the supported theory subset is "
            "1.0, mutation rejection is 1.0, and reconstruction failures stay "
            "unknown (never counted as refutation); else reject/retire."
        ),
        "endpoints": [
            {
                "endpoint_id": "theory_agreement_rate",
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
                "endpoint_id": "mutation_rejection_rate",
                "metric": "mutation_rejection_rate",
                "role": "secondary",
                "direction": "increase",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "fake_refutation_count",
                "metric": "reconstruction_failure_as_refutation_count",
                "role": "secondary",
                "direction": "decrease",
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
                "arm_id": "alethe_pilot",
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
            "single fixture suite stop after theory agreement report",
        ],
        "controls": [
            {
                "control_id": "exhaustive_replay",
                "description": (
                    "Canonical exhaustive finite-domain replay on identical "
                    "theory instances (SAT/CNF overlap where applicable)."
                ),
                "kind": "positive",
            },
            {
                "control_id": "certificate_premise_corrupt",
                "description": "Corrupt Alethe premises must not authorize unsat.",
                "kind": "negative",
            },
            {
                "control_id": "theory_metadata_corrupt",
                "description": "Altered theory metadata must not authorize unsat.",
                "kind": "negative",
            },
            {
                "control_id": "incomplete_reconstruction_unknown",
                "description": (
                    "Incomplete reconstruction / unsupported theory must stay "
                    "unknown, never refuted."
                ),
                "kind": "negative",
            },
        ],
        "negative_controls": [
            "certificate_premise_corrupt",
            "theory_metadata_corrupt",
            "incomplete_reconstruction_unknown",
        ],
        "multiplicity_families": [
            {
                "family_id": "research07_primary",
                "hypothesis_ids": ["theory_agreement_rate"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "agreement_eq_one",
                "endpoint_id": "theory_agreement_rate",
                "operator": "eq",
                "threshold": 1.0,
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
                "endpoint_id": "fake_refutation_count",
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
        "author": "research-07",
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
    """Run RESEARCH-07. Refuses unless explicitly enabled (default-off)."""

    base = root or _repo_root()
    if not enabled:
        return {
            "schema": "research_07_experiment_result/v1",
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

    report = paired_agreement_report(default_theory_suite())
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        "harness.experiments.slm565_research07_vss_alethe",
    )
    payload = {
        "schema": "research_07_experiment_result/v1",
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
        "witness_disagreement_count": report["witness_disagreement_count"],
        "mutation_rejection_rate": report["mutation_rejection_rate"],
        "reconstruction_failure_as_refutation_count": report[
            "reconstruction_failure_as_refutation_count"
        ],
        "unknown_preservation_rate": report["unknown_preservation_rate"],
        "sat_overlap_agreement_rate": report["sat_overlap_agreement_rate"],
        "supported_smt_theories": report["supported_smt_theories"],
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
    decision = result.get("decision")
    theories = result.get("supported_smt_theories") or sorted(SUPPORTED_SMT_THEORIES)
    return f"""# RESEARCH-07 — Alethe/SMT proof reconstruction (SLM-565)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

Alethe/SMT proof reconstruction for theory-rich VSS constraints preserves exact
agreement on a declared theory subset without widening unknown into refutation.

## Contract

| Arm | Role |
| --- | --- |
| Exhaustive finite-domain replay (+ SAT/CNF overlap) | matched control |
| VSS→SMT + checked/reconstructed Alethe via EVID-10 `alethe_pilot` | treatment |

| Gate | Result |
| --- | --- |
| Exact agreement rate (supported subset) | {rate_s} |
| Witness disagreements | {result.get("witness_disagreement_count")} |
| Mutation rejection rate | {result.get("mutation_rejection_rate")} |
| Reconstruction→fake refutation count | {result.get("reconstruction_failure_as_refutation_count")} |
| Decision | **{decision}** |

Decision rule: agreement rate == 1.0, mutation rejection == 1.0, and zero
fake refutations; else reject/retire. Unsupported theory / incomplete
reconstruction → `unknown` (never `refuted`/`unsat`).

## Declared theories + proof format

- Supported SMT theories: `{", ".join(theories)}`
- Proof format version: `{result.get("proof_format_version") or PROOF_FORMAT_VERSION}`
- Solver: fixture-backed stub when cvc5 unavailable (fail closed, default off)

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
| Results JSON | [`iter-revmath-research-07-preregistered.json`](iter-revmath-research-07-preregistered.json) |
| Adapter | `src/slm_training/formal/vss_alethe_backend.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_07_vss_alethe.py` |
| Encoding bridge | `src/slm_training/formal/encoding_adapter.py` (EVID-10) |

## Run

```bash
# default-off: refuses without enable
PYTHONPATH=src uv run python -m scripts.run_research_07_vss_alethe
SLM_ENABLE_RESEARCH_07=1 PYTHONPATH=src uv run python -m scripts.run_research_07_vss_alethe --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Alethe reconstruction remains research-only; production refutation authority
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
    "PRIMARY_METRIC",
    "build_campaign_manifest",
    "load_campaign_lock",
    "lock_campaign",
    "render_evidence_markdown",
    "run_experiment",
    "write_campaign_lock",
    "write_evidence",
]
