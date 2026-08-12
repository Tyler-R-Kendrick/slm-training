"""RESEARCH-04 / SLM-550: Reverse Mathematics Zoo classification benchmark.

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
from slm_training.formal.rm_zoo_benchmark import (
    CORPUS_RELATIVE,
    paired_benchmark_report,
    pinned_toolchain,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-04"
LINEAR_ID = "SLM-550"
CAMPAIGN_ID = "research-04-rm-zoo-benchmark"
EXPERIMENT_ID = "research_04_rm_zoo_benchmark"
EVIDENCE_MD = "docs/design/iter-revmath-research-04-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-04-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_04_campaign_lock.v1.json"
)
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_04"
PRIMARY_METRIC = "exact_classification_reversal_success_dependency_disjoint"
HARNESS_COMPONENT = "harness.experiments.slm550_research04_rm_zoo"


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
        "arm": "big_five_only_benchmark",
        "corpus_filter": "control_big_five",
        "split": "train",
    }
    treatment_cfg = {
        "arm": "rm_zoo_mixed_benchmark",
        "corpus_relpath": CORPUS_RELATIVE,
        "include_non_big_five": True,
        "split_eval": "eval",
        "default_off": True,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "A non-Big-Five Reverse Mathematics Zoo benchmark exposes whether "
            "the revmath harness learns assumption/reversal reasoning rather "
            "than five-label mapping."
        ),
        "decision": (
            "Accept iff dependency-disjoint eval exact rate == 1.0, "
            "overclaim_rate == 0, unknown calibration == 1.0, and zoo "
            "discriminates from Big-Five-only retrieval; else reject/retire."
        ),
        "endpoints": [
            {
                "endpoint_id": "eval_exact_rate",
                "metric": PRIMARY_METRIC,
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 1.0,
            },
            {
                "endpoint_id": "overclaim_rate",
                "metric": "overclaim_rate",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "unknown_calibration",
                "metric": "unknown_calibration",
                "role": "secondary",
                "direction": "increase",
                "minimum_effect": 1.0,
            },
        ],
        "arms": [
            {
                "arm_id": "big_five_only",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "rm_zoo_mixed",
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
            "single fixture zoo corpus stop after paired benchmark report",
        ],
        "controls": [
            {
                "control_id": "big_five_only_train",
                "description": (
                    "Big-Five-only label claims of comparable difficulty on "
                    "train split (WKL0 reversal + overclaim negatives)."
                ),
                "kind": "positive",
            },
            {
                "control_id": "zoo_overclaim_rejected",
                "description": (
                    "Zoo principles must not inherit Big-Five equivalence "
                    "from print_axioms/compile alone."
                ),
                "kind": "negative",
            },
        ],
        "negative_controls": [
            "zoo_overclaim_rejected",
        ],
        "multiplicity_families": [
            {
                "family_id": "research04_primary",
                "hypothesis_ids": ["eval_exact_rate"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "eval_exact",
                "endpoint_id": "eval_exact_rate",
                "operator": "eq",
                "threshold": 1.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "any_overclaim",
                "endpoint_id": "overclaim_rate",
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
        "author": "research-04",
        "created_at": created_at,
    }
    return ExperimentCampaignV1.model_validate(payload)


def lock_campaign(*, root: Path | None = None) -> CampaignLockV1:
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
    """Run RESEARCH-04. Refuses unless explicitly enabled (default-off)."""

    base = root or _repo_root()
    if not enabled:
        return {
            "schema": "research_04_experiment_result/v1",
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

    report = paired_benchmark_report(root=base)
    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        HARNESS_COMPONENT,
    )
    return {
        "schema": "research_04_experiment_result/v1",
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
        "correctness_ok": report["correctness_ok"],
        "train_control_exact_rate": report["train_control_exact_rate"],
        "overclaim_rate": report["overclaim_rate"],
        "unknown_calibration": report["unknown_calibration"],
        "library_retrieval_only": report["library_retrieval_only"],
        "zoo_discriminates": report["zoo_discriminates"],
        "eval_entry_count": report["eval_entry_count"],
        "paired_report": report,
        "toolchain": pinned_toolchain(),
        "evidence_path": EVIDENCE_MD,
        "version_stamp": stamp,
    }


def render_evidence_markdown(result: dict[str, Any]) -> str:
    rate = result.get("primary_value")
    rate_s = "n/a" if rate is None else f"{rate:.6g}"
    decision = result.get("decision")
    return f"""# RESEARCH-04 — Reverse Mathematics Zoo benchmark (SLM-550)

**Status:** preregistered evidence ({decision})  
**Experiment key:** `{EXPERIMENT_KEY}`  
**Linear:** [{LINEAR_ID}](https://linear.app/quickdeploy-ai/issue/{LINEAR_ID})  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

A non-Big-Five Reverse Mathematics Zoo benchmark exposes whether the revmath
harness learns assumption/reversal reasoning rather than five-label mapping.

## Contract

| Arm | Role |
| --- | --- |
| Big-Five-only train benchmark | matched control |
| Zoo + Big-Five mixed eval benchmark | treatment |

| Gate | Result |
| --- | --- |
| Eval exact classification/reversal (dependency-disjoint) | {rate_s} |
| Overclaim rate | {result.get("overclaim_rate")} |
| Unknown calibration | {result.get("unknown_calibration")} |
| Library-retrieval-only falsifier | {result.get("library_retrieval_only")} |
| Zoo discriminates | {result.get("zoo_discriminates")} |
| Decision | **{decision}** |

Reason: `{result.get("reason")}`.

Decision rule: eval exact rate == 1.0, overclaim == 0, unknown calibration == 1.0,
and zoo corpus discriminates from Big-Five-only retrieval; else reject/retire.

## Corpus

- Frozen corpus: `{CORPUS_RELATIVE}`
- Citations: arxiv-2212.00489 (RM Zoo), simpson-sosoa
- Zoo principles: `zoo:DNR`, `zoo:COH`, `zoo:FIP`, `zoo:RT22`, `zoo:CADS`

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
| Results JSON | [`iter-revmath-research-04-preregistered.json`](iter-revmath-research-04-preregistered.json) |
| Backend | `src/slm_training/formal/rm_zoo_benchmark.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_04_rm_zoo_benchmark.py` |

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_04_rm_zoo
SLM_ENABLE_RESEARCH_04=1 PYTHONPATH=src uv run python -m scripts.run_research_04_rm_zoo --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Zoo classifications remain research-only; production RM labels stay HARN-08
conservative labeling over the finite kernel.
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
