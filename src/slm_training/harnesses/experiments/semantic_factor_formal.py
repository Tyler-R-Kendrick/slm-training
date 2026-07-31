"""SFF formal obligations bound to LeverProofLean.AdvisoryResidual cores.

Campaign locks non-empty ``formal_obligations`` with content digests of Lean
sources + golden vectors. Committed preflights must validate as
``FormalPreflightV1``. Regeneration requires a successful Lean proof run and
never stamps ``status: proved`` without ``lean_ok``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from slm_training.autoresearch.schemas import FormalObligationV1, FormalPreflightV1
from slm_training.lineage.records import canonical_json

__all__ = [
    "SFF_FORMAL_TEMPLATES",
    "SFF_FORMAL_DIR",
    "load_sff_formal_obligations",
    "regenerate_sff_formal_preflights",
    "verify_sff_formal_sources",
    "validate_committed_preflights",
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _lean_version() -> str:
    proc = subprocess.run(
        ["lake", "env", "lean", "--version"],
        cwd=_LEAN_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or proc.stderr or "").strip()
    return text or "unknown"


def _proof_sha256(template: Mapping[str, Any], digests: Mapping[str, str]) -> str:
    payload = {
        "theorem": template["theorem"],
        "template_id": template["template_id"],
        "source_digests": dict(digests),
        "lean_project": "src/leverproof_lean",
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _build_preflight_model(
    template: Mapping[str, Any],
    *,
    lean_ok: bool,
    lean_version: str,
    build_output: str,
    duration_seconds: float,
) -> FormalPreflightV1:
    digests = _source_digests(tuple(template["source_paths"]))
    oid = _obligation_id(
        str(template["template_id"]),
        str(template["claim"]),
        str(template["policy"]),
    )
    status = "proved" if lean_ok else "unknown"
    return FormalPreflightV1(
        campaign_id="SFF-anti-e237-v1",
        experiment_id="semantic_factor_frontier_measured",
        obligation_id=oid,
        template_id=str(template["template_id"]),
        template_version="v1",
        claim=str(template["claim"]),
        policy=str(template["policy"]),  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        evidence_scope="universal",
        theorem=str(template["theorem"]),
        proof_target="LeverProofLean.AdvisoryResidual",
        checker_contract="make -C src/leverproof_lean proofs",
        assumptions=(
            "LeverProofLean Mathlib-free closed development",
            "AdvisoryResidual theorems certified via make proofs",
        ),
        open_assumptions=(),
        source_digests=dict(digests),
        proof_sha256=_proof_sha256(template, digests),
        lean_version=lean_version,
        mathlib_version="none",  # leverproof project is Mathlib-free
        build_output_sha256=_sha256_text(build_output),
        counterexample=None,
        duration_seconds=float(duration_seconds),
    )


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

    Emits ``FormalPreflightV1``-valid artifacts. Raises if Lean fails and
    ``require_lean`` is true. Never stamps ``status: proved`` without Lean OK.
    """

    SFF_FORMAL_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = subprocess.run(
        ["make", "proofs"],
        cwd=_LEAN_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    duration = time.monotonic() - started
    lean_ok = proc.returncode == 0
    build_output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    if require_lean and not lean_ok:
        raise SFFFormalError(
            "cannot regenerate formal preflights: Lean proofs failed:\n"
            + build_output[-2000:]
        )
    lean_version = _lean_version()
    written: list[Path] = []
    for template in SFF_FORMAL_TEMPLATES:
        preflight = _build_preflight_model(
            template,
            lean_ok=lean_ok,
            lean_version=lean_version,
            build_output=build_output,
            duration_seconds=duration,
        )
        if preflight.status != "proved":
            raise SFFFormalError(
                f"refusing to write non-proved preflight for {template['template_id']}"
            )
        # Round-trip validate before write.
        FormalPreflightV1.model_validate(preflight.model_dump())
        path = _preflight_path(str(template["template_id"]))
        path.write_text(
            json.dumps(preflight.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def validate_committed_preflights() -> list[FormalPreflightV1]:
    """Load and schema-validate every committed preflight (fail closed)."""

    out: list[FormalPreflightV1] = []
    for template in SFF_FORMAL_TEMPLATES:
        tid = str(template["template_id"])
        path = _preflight_path(tid)
        if not path.is_file():
            raise SFFFormalError(f"missing formal preflight for {tid}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            preflight = FormalPreflightV1.model_validate(raw)
        except Exception as exc:
            raise SFFFormalError(
                f"formal preflight for {tid} is not FormalPreflightV1: {exc}"
            ) from exc
        expected = _source_digests(tuple(template["source_paths"]))
        if dict(preflight.source_digests) != expected:
            raise SFFFormalError(
                f"formal preflight stale for {tid}: source digests drifted; "
                "run regenerate_sff_formal_preflights() after make proofs"
            )
        if preflight.status != "proved":
            raise SFFFormalError(
                f"formal preflight for {tid} status is {preflight.status!r}, not proved"
            )
        if preflight.theorem != template["theorem"]:
            raise SFFFormalError(
                f"formal preflight theorem mismatch for {tid}: "
                f"{preflight.theorem!r} != {template['theorem']!r}"
            )
        oid = _obligation_id(tid, str(template["claim"]), str(template["policy"]))
        if preflight.obligation_id != oid:
            raise SFFFormalError(
                f"formal preflight obligation_id mismatch for {tid}"
            )
        out.append(preflight)
    return out


def load_sff_formal_obligations() -> tuple[FormalObligationV1, ...]:
    """Load committed preflights; **fail closed** on missing/stale/invalid.

    Does not rewrite files. Use :func:`regenerate_sff_formal_preflights` after
    Lean source changes (requires a green ``make proofs``).
    """

    preflights = validate_committed_preflights()
    obligations: list[FormalObligationV1] = []
    for template, preflight in zip(SFF_FORMAL_TEMPLATES, preflights, strict=True):
        path = _preflight_path(str(template["template_id"]))
        preflight_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        obligations.append(
            FormalObligationV1(
                obligation_id=preflight.obligation_id,
                template_id=preflight.template_id,
                policy=preflight.policy,
                preflight_sha256=preflight_sha,
            )
        )
    return tuple(obligations)


def load_or_build_sff_formal_obligations(
    *, write_if_missing: bool = False
) -> tuple[FormalObligationV1, ...]:
    """Deprecated alias: always fail-closed load (never auto-writes)."""

    del write_if_missing
    return load_sff_formal_obligations()
