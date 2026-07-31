"""SFF formal obligations bound to LeverProofLean.AdvisoryResidual cores.

Campaign locks non-empty ``formal_obligations`` with content digests of Lean
sources + golden vectors. **Never** rewrites a committed preflight as
``status: proved`` without a successful Lean proof run.
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
    "load_sff_formal_obligations",
    "regenerate_sff_formal_preflights",
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
        "theorem": "LeverProofLean.AdvisoryResidual.golden_S_column_masses_from_B",
        "claim": "Golden incidence B yields column-stochastic S numerators (derived from B).",
        "policy": "advisory",
        "source_paths": (
            "src/leverproof_lean/LeverProofLean/AdvisoryResidual.lean",
            "src/slm_training/resources/experiments/semantic_factor_frontier/golden_vectors.v1.json",
            "src/slm_training/models/semantic_factor_propagation.py",
        ),
    },
    {
        "template_id": "sff.role-shuffle-membership",
        "theorem": "LeverProofLean.AdvisoryResidual.role_shuffle_preserves_membership",
        "claim": "Role rotation preserves factor membership node lists.",
        "policy": "required",
        "source_paths": (
            "src/leverproof_lean/LeverProofLean/AdvisoryResidual.lean",
            "src/slm_training/data/progspec/semantic_evidence.py",
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


def _preflight_path(template_id: str) -> Path:
    return SFF_FORMAL_DIR / f"{template_id.replace('.', '_')}.json"


def _preflight_payload(
    template: Mapping[str, Any],
    *,
    lean_ok: bool,
    lean_log_tail: str = "",
) -> dict[str, Any]:
    digests = _source_digests(tuple(template["source_paths"]))
    oid = _obligation_id(
        str(template["template_id"]),
        str(template["claim"]),
        str(template["policy"]),
    )
    body: dict[str, Any] = {
        "schema_version": "FormalPreflightV1",
        "campaign_id": "SFF-anti-e237-v1",
        "experiment_id": "semantic_factor_frontier_measured",
        "obligation_id": oid,
        "template_id": template["template_id"],
        "template_version": "v1",
        "claim": template["claim"],
        "policy": template["policy"],
        # Only "proved" when Lean checker succeeded in this regeneration.
        "status": "proved" if lean_ok else "unknown",
        "evidence_scope": "universal",
        "theorem": template["theorem"],
        "proof_target": "LeverProofLean.AdvisoryResidual",
        "source_digests": digests,
        "lean_project": "src/leverproof_lean",
        "checker": "make -C src/leverproof_lean proofs",
        "lean_ok": lean_ok,
    }
    if lean_log_tail:
        body["lean_log_tail"] = lean_log_tail[-1500:]
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


def regenerate_sff_formal_preflights(*, require_lean: bool = True) -> list[Path]:
    """Rewrite preflight JSON only after a successful Lean proof run.

    Raises if Lean fails and ``require_lean`` is true. Never stamps
    ``status: proved`` without ``lean_ok``.
    """

    SFF_FORMAL_DIR.mkdir(parents=True, exist_ok=True)
    lean_ok = False
    lean_log = ""
    proc = subprocess.run(
        ["make", "proofs"],
        cwd=_LEAN_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    lean_ok = proc.returncode == 0
    lean_log = (proc.stderr or proc.stdout or "")[-2000:]
    if require_lean and not lean_ok:
        raise SFFFormalError(
            "cannot regenerate formal preflights: Lean proofs failed:\n" + lean_log
        )
    written: list[Path] = []
    for template in SFF_FORMAL_TEMPLATES:
        path = _preflight_path(str(template["template_id"]))
        payload = _preflight_payload(
            template, lean_ok=lean_ok, lean_log_tail=lean_log
        )
        if payload["status"] != "proved":
            raise SFFFormalError(
                f"refusing to write non-proved preflight for {template['template_id']}"
            )
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(path)
    return written


def load_sff_formal_obligations() -> tuple[FormalObligationV1, ...]:
    """Load committed preflights; **fail closed** on missing or stale digests.

    Does not rewrite files. Use :func:`regenerate_sff_formal_preflights` after
    Lean source changes (requires a green ``make proofs``).
    """

    obligations: list[FormalObligationV1] = []
    for template in SFF_FORMAL_TEMPLATES:
        tid = str(template["template_id"])
        path = _preflight_path(tid)
        if not path.is_file():
            raise SFFFormalError(
                f"missing formal preflight for {tid}; run "
                "regenerate_sff_formal_preflights() after Lean proofs pass"
            )
        stored = json.loads(path.read_text(encoding="utf-8"))
        expected_digests = _source_digests(tuple(template["source_paths"]))
        if stored.get("source_digests") != expected_digests:
            raise SFFFormalError(
                f"formal preflight stale for {tid}: source digests drifted; "
                "run regenerate_sff_formal_preflights() after make proofs"
            )
        if stored.get("status") != "proved" or stored.get("lean_ok") is not True:
            raise SFFFormalError(
                f"formal preflight for {tid} is not a proved Lean result "
                f"(status={stored.get('status')!r}, lean_ok={stored.get('lean_ok')!r})"
            )
        if stored.get("theorem") != template["theorem"]:
            raise SFFFormalError(
                f"formal preflight theorem mismatch for {tid}: "
                f"{stored.get('theorem')!r} != {template['theorem']!r}"
            )
        oid = _obligation_id(tid, str(template["claim"]), str(template["policy"]))
        if stored.get("obligation_id") != oid:
            raise SFFFormalError(
                f"formal preflight obligation_id mismatch for {tid}"
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


# Back-compat alias used by older call sites during transition.
def load_or_build_sff_formal_obligations(
    *, write_if_missing: bool = False
) -> tuple[FormalObligationV1, ...]:
    """Deprecated: prefer ``load_sff_formal_obligations`` (fail-closed).

    ``write_if_missing`` is ignored for safety — regeneration requires an
    explicit :func:`regenerate_sff_formal_preflights` call with Lean green.
    """

    del write_if_missing
    return load_sff_formal_obligations()
