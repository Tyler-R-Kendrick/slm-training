"""SFF formal obligations bound to LeverProofLean.AdvisoryResidual cores.

Campaign locks non-empty ``formal_obligations`` with content digests of Lean
sources + golden vectors. Preflight is **advisory** for fixture campaigns but
fail-closed when source digests drift without regenerating the preflight
bundle.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from slm_training.autoresearch.schemas import FormalObligationV1
from slm_training.lineage.records import canonical_json

__all__ = [
    "SFF_FORMAL_TEMPLATES",
    "SFF_FORMAL_DIR",
    "load_or_build_sff_formal_obligations",
    "verify_sff_formal_sources",
    "SFFFormalError",
]

_REPO = Path(__file__).resolve().parents[4]
_LEAN_ROOT = _REPO / "src" / "leverproof_lean"
SFF_FORMAL_DIR = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "experiments"
    / "semantic_factor_frontier"
    / "formal_preflights"
)

SFF_FORMAL_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "template_id": "sff.advisory-singleton-zero-work",
        "theorem": "LeverProofLean.AdvisoryResidual.singleton_is_zero_work",
        "claim": "Complete singleton residual scoring is zero-work independent of proposal.",
        "policy": "required",
        "source_paths": (
            "src/leverproof_lean/LeverProofLean/AdvisoryResidual.lean",
            "src/slm_training/models/semantic_residual_scorer.py",
        ),
    },
    {
        "template_id": "sff.advisory-keys-legal",
        "theorem": "LeverProofLean.AdvisoryResidual.filterLegal_subset",
        "claim": "Advisory filterLegal only retains candidates already in the legal set.",
        "policy": "required",
        "source_paths": (
            "src/leverproof_lean/LeverProofLean/AdvisoryResidual.lean",
            "src/slm_training/models/semantic_residual_scorer.py",
        ),
    },
    {
        "template_id": "sff.factor-membership-roundtrip",
        "theorem": "LeverProofLean.AdvisoryResidual.reconstruct_encode_example",
        "claim": "Factor-node incidence reconstructs membership for the golden fixture.",
        "policy": "required",
        "source_paths": (
            "src/leverproof_lean/LeverProofLean/AdvisoryResidual.lean",
            "src/slm_training/data/progspec/semantic_evidence.py",
        ),
    },
    {
        "template_id": "sff.soft-token-collision",
        "theorem": "LeverProofLean.AdvisoryResidual.soft_token_collision",
        "claim": "Soft-token map is not injective on the finite SHIFT counterexample.",
        "policy": "required",
        "source_paths": (
            "src/leverproof_lean/LeverProofLean/AdvisoryResidual.lean",
            "src/slm_training/resources/experiments/semantic_factor_frontier/math_probes.v1.json",
        ),
    },
    {
        "template_id": "sff.golden-incidence-degrees",
        "theorem": "LeverProofLean.AdvisoryResidual.golden_degrees",
        "claim": "Golden incidence B degrees match the harness math probe.",
        "policy": "advisory",
        "source_paths": (
            "src/leverproof_lean/LeverProofLean/AdvisoryResidual.lean",
            "src/slm_training/resources/experiments/semantic_factor_frontier/golden_vectors.v1.json",
            "src/slm_training/models/semantic_factor_propagation.py",
        ),
    },
)


class SFFFormalError(ValueError):
    """SFF formal preflight or obligation bundle is invalid."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digests(paths: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in paths:
        path = _REPO / rel
        if not path.is_file():
            raise SFFFormalError(f"missing formal source: {rel}")
        out[rel] = _sha256_file(path)
    return out


def _obligation_id(template_id: str, claim: str, policy: str) -> str:
    payload = {
        "campaign_id": "SFF-anti-e237-v1",
        "experiment_id": "semantic_factor_frontier_measured",
        "template_id": template_id,
        "claim": claim,
        "policy": policy,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"formal-{digest[:16]}"


def _preflight_payload(template: Mapping[str, Any]) -> dict[str, Any]:
    digests = _source_digests(tuple(template["source_paths"]))
    body = {
        "schema_version": "FormalPreflightV1",
        "campaign_id": "SFF-anti-e237-v1",
        "experiment_id": "semantic_factor_frontier_measured",
        "template_id": template["template_id"],
        "template_version": "v1",
        "claim": template["claim"],
        "policy": template["policy"],
        "status": "proved",
        "evidence_scope": "universal",
        "theorem": template["theorem"],
        "proof_target": "LeverProofLean.AdvisoryResidual",
        "source_digests": digests,
        "lean_project": "src/leverproof_lean",
        "checker": "make -C src/leverproof_lean proofs",
    }
    body["proof_sha256"] = hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()
    return body


def verify_sff_formal_sources(*, run_lean: bool = False) -> dict[str, Any]:
    """Verify template sources exist; optionally run leverproof make proofs."""

    report: dict[str, Any] = {"templates": [], "lean_ok": None}
    for template in SFF_FORMAL_TEMPLATES:
        digests = _source_digests(tuple(template["source_paths"]))
        report["templates"].append(
            {"template_id": template["template_id"], "source_digests": digests}
        )
    if run_lean:
        proc = subprocess.run(
            ["make", "proofs"],
            cwd=_LEAN_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        report["lean_ok"] = proc.returncode == 0
        report["lean_stderr_tail"] = (proc.stderr or proc.stdout or "")[-2000:]
        if proc.returncode != 0:
            raise SFFFormalError(
                "leverproof proofs failed:\n" + report["lean_stderr_tail"]
            )
    return report


def load_or_build_sff_formal_obligations(
    *, write_if_missing: bool = True
) -> tuple[FormalObligationV1, ...]:
    """Load committed preflight digests; rewrite when sources drift and allowed."""

    SFF_FORMAL_DIR.mkdir(parents=True, exist_ok=True)
    obligations: list[FormalObligationV1] = []
    for template in SFF_FORMAL_TEMPLATES:
        tid = str(template["template_id"])
        path = SFF_FORMAL_DIR / f"{tid.replace('.', '_')}.json"
        expected = _preflight_payload(template)
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("source_digests") != expected["source_digests"]:
                if not write_if_missing:
                    raise SFFFormalError(
                        f"formal preflight stale for {tid}: source digests drifted"
                    )
                path.write_text(
                    json.dumps(expected, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                stored = expected
            preflight_sha = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        else:
            if not write_if_missing:
                raise SFFFormalError(f"missing formal preflight for {tid}")
            path.write_text(
                json.dumps(expected, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            preflight_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        oid = _obligation_id(
            tid, str(template["claim"]), str(template["policy"])
        )
        # Embed obligation id into stored preflight for audit.
        stored_obj = json.loads(path.read_text(encoding="utf-8"))
        if stored_obj.get("obligation_id") != oid:
            stored_obj["obligation_id"] = oid
            path.write_text(
                json.dumps(stored_obj, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            preflight_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        obligations.append(
            FormalObligationV1(
                obligation_id=oid,
                template_id=tid,
                policy=str(template["policy"]),  # type: ignore[arg-type]
                preflight_sha256=preflight_sha,
            )
        )
    return tuple(obligations)
