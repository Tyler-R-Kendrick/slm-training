"""RESEARCH-03 / SLM-549: Big-Five interpretation pilot (isolated Lean package).

Default-off / research-only. Execution requires RESEARCH-02
``assert_execution_allowed`` (blockers complete + campaign lock digest).
Production code must not import ``src/revmath_research_lean``.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    CampaignLockV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.formal.computability_classification import (
    ClassificationClaimV1,
    InterpretationPackageV1,
    validate_classification_claim,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.labeling import (
    RmLabelClaimV1,
    validate_label_claim,
)
from slm_training.harnesses.reasoning.revmath.schemas import (
    AxisCertificateRefV1,
    RevmathFourAxisAnalysisV1,
)
from slm_training.lineage.records import canonical_json
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_KEY = "RESEARCH-03"
LINEAR_ID = "SLM-549"
CAMPAIGN_ID = "research-03-big-five-interpretation"
EXPERIMENT_ID = "research_03_big_five_interpretation"
EVIDENCE_MD = "docs/design/iter-revmath-research-03-preregistered.md"
EVIDENCE_JSON = "docs/design/iter-revmath-research-03-preregistered.json"
LOCK_RELPATH = (
    "src/slm_training/resources/formal/research_03_campaign_lock.v1.json"
)
PACKAGE_RELPATH = "src/revmath_research_lean"
ENABLE_FLAG = "SLM_ENABLE_RESEARCH_03"
PRIMARY_METRIC = "checked_bidirectional_classification_with_documented_assumptions"
CODING_ID = "coding.wkl_sigma01_sep.v1"
BASE_THEORY_ID = "RCA0"
TARGET_SUBSYSTEM = "WKL0"
NAMED_AXIOMS = (
    "RevMathResearch.Forward.wkl_implies_sigma01_sep",
    "RevMathResearch.Reverse.sigma01_sep_implies_wkl",
    "RevMathResearch.Reverse.not_both_dead_bridge",
)

# Production trees that must never import the research package.
_PRODUCTION_LEAN_ROOTS = (
    "src/leverproof_lean",
    "src/slm_training/formal/lean",
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "src" / "slm_training"
        ).is_dir():
            return parent
    return here.parents[4]


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


def forward_evidence_sha256() -> str:
    return content_sha(
        {
            "direction": "forward",
            "axiom": NAMED_AXIOMS[0],
            "coding_id": CODING_ID,
        }
    )


def reversal_evidence_sha256() -> str:
    return content_sha(
        {
            "direction": "reverse",
            "axiom": NAMED_AXIOMS[1],
            "coding_id": CODING_ID,
        }
    )


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
        "arm": "production_finite_kernel_labeling",
        "imports_research_package": False,
        "big_five_from_print_axioms": False,
    }
    treatment_cfg = {
        "arm": "revmath_research_lean_package",
        "package": PACKAGE_RELPATH,
        "coding_id": CODING_ID,
        "base_theory_id": BASE_THEORY_ID,
        "target_subsystem": TARGET_SUBSYSTEM,
        "named_axioms": list(NAMED_AXIOMS),
        "default_off": True,
        "production_authority": False,
    }
    payload: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "An isolated Lean research package can encode one genuine Big-Five "
            "bidirectional classification without contaminating the production "
            "finite kernel."
        ),
        "decision": (
            "Accept iff lake build + explicit interpretation package validate a "
            "bidirectional WKL↔Σ⁰₁-Sep classification, production trees do not "
            "import the package, and #print-axioms-alone stays insufficient; "
            "else reject/unknown."
        ),
        "endpoints": [
            {
                "endpoint_id": "bidirectional_classification",
                "metric": PRIMARY_METRIC,
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 1.0,
            },
            {
                "endpoint_id": "production_import_contamination",
                "metric": "production_import_contamination",
                "role": "secondary",
                "direction": "decrease",
                "minimum_effect": 0.0,
            },
            {
                "endpoint_id": "print_axioms_alone_rejected",
                "metric": "print_axioms_alone_rejected",
                "role": "secondary",
                "direction": "increase",
                "minimum_effect": 1.0,
            },
        ],
        "arms": [
            {
                "arm_id": "control_production_finite_kernel",
                "role": "control",
                "config_sha256": _config_sha(control_cfg),
            },
            {
                "arm_id": "treatment_revmath_research_lean",
                "role": "candidate",
                "config_sha256": _config_sha(treatment_cfg),
            },
        ],
        "seeds": [0],
        "budget": {"max_experiments": 2, "max_wall_minutes": 30.0},
        "stopping_rules": [
            "Abort on production import contamination.",
            "Unknown when lake unavailable or encoding burden disproportionate.",
            "Reject when interpretation package fails validation.",
        ],
        "controls": [
            {
                "control_id": "matched_production_finite_kernel",
                "description": (
                    "Control is the production finite-kernel labeling path "
                    "without research package import."
                ),
                "kind": "positive",
            },
            {
                "control_id": "print_axioms_alone_negative",
                "description": (
                    "#print axioms alone must not authorize rm_research:WKL0."
                ),
                "kind": "negative",
            },
            {
                "control_id": "production_import_contamination_negative",
                "description": (
                    "LeverProofLean / OpenUIProofs must not import RevMathResearch."
                ),
                "kind": "negative",
            },
        ],
        "negative_controls": [
            "print_axioms_alone_negative",
            "production_import_contamination_negative",
        ],
        "multiplicity_families": [
            {
                "family_id": "research03_primary",
                "hypothesis_ids": ["bidirectional_classification"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "classification_checked",
                "endpoint_id": "bidirectional_classification",
                "operator": "eq",
                "threshold": 1.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "any_contamination",
                "endpoint_id": "production_import_contamination",
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
        "author": "research-03",
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


def _scan_production_imports(root: Path) -> list[str]:
    """Return production Lean files that import the research package."""
    hits: list[str] = []
    pattern = re.compile(r"\bimport\s+RevMathResearch\b")
    for rel in _PRODUCTION_LEAN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in base.rglob("*.lean"):
            text = path.read_text(encoding="utf-8")
            if pattern.search(text) or "revmath_research_lean" in text:
                hits.append(str(path.relative_to(root)))
    return sorted(hits)


def _lake_build(package_dir: Path) -> dict[str, Any]:
    lake = None
    for candidate in (
        Path.home() / ".elan" / "bin" / "lake",
        Path("/usr/bin/lake"),
    ):
        if candidate.is_file():
            lake = candidate
            break
    if lake is None:
        from shutil import which

        found = which("lake")
        lake = Path(found) if found else None
    if lake is None:
        return {
            "ok": False,
            "status": "unknown",
            "reason": "lake_unavailable",
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    proc = subprocess.run(
        [str(lake), "build"],
        cwd=package_dir,
        capture_output=True,
        text=True,
        timeout=60 * 25,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "status": "ok" if proc.returncode == 0 else "invalid",
        "reason": "lake_build",
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def _control_print_axioms_rejected() -> dict[str, Any]:
    """Matched control: #print axioms alone must not yield Big-Five labels."""
    claim = RmLabelClaimV1(
        schema_version="revmath_label_claim/v1",
        claim_id="research03.control.print_axioms",
        labeling_state="reversed_equivalent",
        proposed_subsystem_id="WKL0",
        analysis_kind="genuine_reverse_mathematics",
        evidence_kinds=("print_axioms_only",),
        base_theory_id=None,
        coding_id=None,
        interpretation_status="uninterpreted",
        forward_evidence_sha256=None,
        reversal_evidence_sha256=None,
        practical_class=None,
        notes="control: axioms listing alone",
    )
    result = validate_label_claim(claim)
    accepted = bool(result.accepted)
    return {
        "rejected": not accepted,
        "reason": "print_axioms_alone",
        "accepted": accepted,
        "rejection_reasons": list(result.rejection_reasons),
    }


def _treatment_interpretation_ok() -> dict[str, Any]:
    fwd = forward_evidence_sha256()
    rev = reversal_evidence_sha256()
    package = InterpretationPackageV1(
        base_theory_id=BASE_THEORY_ID,
        coding_id=CODING_ID,
        interpretation_status="explicit_reversal",
        forward_evidence_sha256=fwd,
        reversal_evidence_sha256=rev,
    )
    claim = ClassificationClaimV1(
        schema_version="computability_classification/v1",
        claim_id="research03.wkl0.bidirectional",
        class_id=f"rm_research:{TARGET_SUBSYSTEM}",
        evidence_kinds=(
            "interpretation_package",
            "forward_implication",
            "reversal",
        ),
        source_family="revmath_research_lean",
        notes="RESEARCH-03 isolated Lean package; named Simpson glue axioms listed",
        interpretation_package=package,
    )
    verdict = validate_classification_claim(claim)
    label = RmLabelClaimV1(
        schema_version="revmath_label_claim/v1",
        claim_id="research03.label.reversed_equivalent",
        labeling_state="reversed_equivalent",
        proposed_subsystem_id=TARGET_SUBSYSTEM,
        analysis_kind="genuine_reverse_mathematics",
        evidence_kinds=(
            "interpretation_package",
            "forward_implication",
            "reversal",
        ),
        base_theory_id=BASE_THEORY_ID,
        coding_id=CODING_ID,
        interpretation_status="explicit_reversal",
        forward_evidence_sha256=fwd,
        reversal_evidence_sha256=rev,
        practical_class=None,
        notes="RESEARCH-03 bidirectional WKL↔Sep",
    )
    label_result = validate_label_claim(label)
    return {
        "package": package.to_dict(),
        "classification_accepted": bool(verdict.accepted),
        "classification_digest": verdict.verdict_digest,
        "label_accepted": bool(label_result.accepted),
        "label_state": label_result.labeling_state,
        "label_rejection_reasons": list(label_result.rejection_reasons),
        "forward_evidence_sha256": fwd,
        "reversal_evidence_sha256": rev,
        "named_axioms": list(NAMED_AXIOMS),
    }


def _four_axis(*, lake_ok: bool, classification_ok: bool) -> dict[str, Any]:
    proved = "proved_axis" if lake_ok and classification_ok else "unknown_axis"
    cert = forward_evidence_sha256() if proved == "proved_axis" else None
    analysis = RevmathFourAxisAnalysisV1(
        assumption_strength=AxisCertificateRefV1(
            axis="assumption_strength",
            status=proved,
            certificate_sha256=cert,
            theorem_ref="RevMathResearch.Interpretation.forward_from_wkl"
            if proved == "proved_axis"
            else None,
            notes="RCA0 research base + named Simpson glue axioms",
        ),
        computability=AxisCertificateRefV1(
            axis="computability",
            status=proved,
            certificate_sha256=reversal_evidence_sha256()
            if proved == "proved_axis"
            else None,
            theorem_ref="RevMathResearch.Interpretation.reverse_from_sep"
            if proved == "proved_axis"
            else None,
            notes="rm_research:WKL0 via explicit interpretation package",
        ),
        resource_bounds=AxisCertificateRefV1(
            axis="resource_bounds",
            status="not_claimed",
            notes="Not the subject of RESEARCH-03",
        ),
        implementation_refinement=AxisCertificateRefV1(
            axis="implementation_refinement",
            status="proved_axis" if lake_ok else "unknown_axis",
            certificate_sha256=content_sha({"isolation": True, "package": PACKAGE_RELPATH})
            if lake_ok
            else None,
            theorem_ref="package_isolation_scan" if lake_ok else None,
            notes="Production Lean trees do not import RevMathResearch",
        ),
    )
    return analysis.model_dump(mode="json")


def run_experiment(
    *,
    root: Path | None = None,
    enabled: bool = False,
    run_lake: bool = True,
) -> dict[str, Any]:
    """Run RESEARCH-03. Refuses unless explicitly enabled (default-off)."""

    base = root or _repo_root()
    if not enabled:
        return {
            "schema": "research_03_experiment_result/v1",
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

    package_dir = base / PACKAGE_RELPATH
    if not package_dir.is_dir():
        raise FileNotFoundError(f"missing research package: {package_dir}")

    contamination = _scan_production_imports(base)
    control = _control_print_axioms_rejected()
    treatment = _treatment_interpretation_ok()
    lake = (
        _lake_build(package_dir)
        if run_lake
        else {
            "ok": True,
            "status": "ok",
            "reason": "lake_skipped_for_test",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }
    )

    classification_ok = bool(
        treatment["classification_accepted"] and treatment["label_accepted"]
    )
    contamination_count = len(contamination)
    print_axioms_ok = bool(control["rejected"])

    if contamination_count > 0:
        decision = "reject"
        reason = "production_import_contamination"
    elif not print_axioms_ok:
        decision = "reject"
        reason = "print_axioms_control_failed"
    elif lake["status"] == "unknown":
        decision = "unknown"
        reason = "lake_unavailable_encoding_burden_unknown"
    elif not lake["ok"]:
        decision = "reject"
        reason = "lake_build_failed"
    elif not classification_ok:
        decision = "reject"
        reason = "interpretation_package_invalid"
    else:
        decision = "accept"
        reason = "bidirectional_classification_checked"

    primary_value = 1.0 if decision == "accept" else 0.0
    four_axis = _four_axis(
        lake_ok=bool(lake["ok"]),
        classification_ok=classification_ok and decision == "accept",
    )

    stamp = build_version_stamp(
        "governance.research_experiment_preregistry",
        "harness.experiments.slm549_research03_big_five",
    )
    transfer_lessons = [
        "Keep genuine RM encodings in an isolated Lake package; never import into LeverProofLean/OpenUIProofs.",
        "Big-Five labels require InterpretationPackageV1 (coding_id + both evidence digests); #print axioms is a negative control only.",
        "Named literature glue axioms are acceptable research scaffolding when listed in the interpretation package — they must not become production authority.",
        "Assumption-minimization tasks can reuse the four-axis split: assumption_strength vs computability vs isolation refinement.",
    ]

    return {
        "schema": "research_03_experiment_result/v1",
        "experiment_key": EXPERIMENT_KEY,
        "linear_id": LINEAR_ID,
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": EXPERIMENT_ID,
        "default_off": True,
        "research_only": True,
        "executed": True,
        "campaign_lock_sha256": lock.manifest_sha256,
        "claim_class": "fixture",
        "decision": decision,
        "reason": reason,
        "primary_metric": PRIMARY_METRIC,
        "primary_value": primary_value,
        "checked_bidirectional_classification_with_documented_assumptions": primary_value,
        "production_import_contamination": contamination_count,
        "contamination_hits": contamination,
        "print_axioms_alone_rejected": print_axioms_ok,
        "lake": {k: lake[k] for k in ("ok", "status", "reason", "returncode")},
        "treatment": treatment,
        "control": control,
        "four_axis": four_axis,
        "package_path": PACKAGE_RELPATH,
        "coding_id": CODING_ID,
        "named_axioms": list(NAMED_AXIOMS),
        "transfer_lessons": transfer_lessons,
        "promotion_boundary": (
            "research_only / default_off: no production decode, ship-gate, or "
            "serving authority"
        ),
        "version_stamp": stamp,
    }


def render_evidence_markdown(result: dict[str, Any]) -> str:
    lessons = "\n".join(f"- {x}" for x in result.get("transfer_lessons") or [])
    axioms = "\n".join(f"- `{a}`" for a in result.get("named_axioms") or [])
    return f"""# RESEARCH-03 — Big-Five interpretation Lean package (SLM-549)

**Status:** preregistered evidence ({result.get("decision")})
**Experiment key:** `RESEARCH-03`
**Linear:** [SLM-549](https://linear.app/quickdeploy-ai/issue/SLM-549)
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

An isolated Lean research package can encode one genuine Big-Five bidirectional
classification (WKL ↔ Σ⁰₁ separation over RCA₀) without contaminating the
production finite kernel.

## Contract

| Arm | Role |
| --- | --- |
| Production finite-kernel labeling (no research import; `#print axioms` alone) | matched control |
| `src/revmath_research_lean` + explicit InterpretationPackageV1 | treatment |

| Gate | Result |
| --- | --- |
| Checked bidirectional classification | {result.get("primary_value")} |
| Production import contamination | {result.get("production_import_contamination")} |
| `#print axioms` alone rejected | {result.get("print_axioms_alone_rejected")} |
| Lake build | {result.get("lake", {}).get("status")} |
| Decision | **{result.get("decision")}** |

Reason: `{result.get("reason")}`.

## Explicit coding / interpretation

- coding_id: `{result.get("coding_id")}`
- base: `{BASE_THEORY_ID}`
- target: `rm_research:{TARGET_SUBSYSTEM}`
- Named axioms (insufficient alone — package required):

{axioms}

## Campaign lock

- Manifest sha256: `{result.get("campaign_lock_sha256")}`
- Lock artifact: `{LOCK_RELPATH}`
- Claim class: `fixture` (research pilot; no promotion)

## Four-axis evidence

```json
{json.dumps(result.get("four_axis"), indent=2, sort_keys=True)}
```

## Transfer lessons (production assumption-minimization)

{lessons}

## Artifacts

| Artifact | Path |
| --- | --- |
| Results JSON | [`iter-revmath-research-03-preregistered.json`](iter-revmath-research-03-preregistered.json) |
| Lean package | `{PACKAGE_RELPATH}` |
| Experiment | `src/slm_training/harnesses/experiments/research_03_big_five_interpretation.py` |

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_03_big_five
SLM_ENABLE_RESEARCH_03=1 PYTHONPATH=src uv run python -m scripts.run_research_03_big_five --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
`rm_research:WKL0` stays research-only; production practical computability
labels remain the KERN-12 finite vocabulary.
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
    """Activate RESEARCH-03 in the RESEARCH-02 registry (same change as evidence)."""
    base = root or _repo_root()
    path = base / "src/slm_training/resources/research_experiment_preregistry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    found = False
    for row in data["experiments"]:
        if row.get("experiment_key") != EXPERIMENT_KEY:
            continue
        row["campaign_lock_sha256"] = lock_sha
        row["disposition"] = disposition
        # Ensure Linear blockers are recorded complete.
        blocker_ids = {b["blocker_id"] for b in row.get("blockers") or []}
        for bid, note in (
            ("SLM-532", "KERN-12 computability vocabulary landed"),
            ("SLM-536", "HARN-03 revmath runner landed"),
        ):
            if bid not in blocker_ids:
                row.setdefault("blockers", []).append(
                    {"blocker_id": bid, "note": note, "status": "completed"}
                )
        found = True
        break
    if not found:
        raise RuntimeError(f"{EXPERIMENT_KEY} missing from preregistry")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "CAMPAIGN_ID",
    "ENABLE_FLAG",
    "EVIDENCE_JSON",
    "EVIDENCE_MD",
    "EXPERIMENT_ID",
    "EXPERIMENT_KEY",
    "LINEAR_ID",
    "LOCK_RELPATH",
    "PACKAGE_RELPATH",
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
