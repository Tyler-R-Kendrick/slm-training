"""Mixture manifests and online family-weighted sampling (P1b)."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.catalog import (
    KNOWN_FAMILIES,
    classify_source_family,
)


@dataclass(frozen=True)
class MixtureManifest:
    mixture_id: str
    weights: dict[str, float]
    notes: str = ""
    version: int = 1

    def normalized(self) -> MixtureManifest:
        clean = {k: float(v) for k, v in self.weights.items() if float(v) > 0}
        total = sum(clean.values())
        if total <= 0:
            raise ValueError("mixture weights must sum to a positive value")
        return MixtureManifest(
            mixture_id=self.mixture_id,
            weights={k: v / total for k, v in sorted(clean.items())},
            notes=self.notes,
            version=self.version,
        )


def load_mixture_manifest(path: Path | str) -> MixtureManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MixtureManifest(
        mixture_id=str(data.get("mixture_id") or Path(path).stem),
        weights={str(k): float(v) for k, v in (data.get("weights") or {}).items()},
        notes=str(data.get("notes") or ""),
        version=int(data.get("version") or 1),
    ).normalized()


def write_mixture_manifest(path: Path | str, manifest: MixtureManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest.normalized())
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def mixture_hash(manifest: MixtureManifest) -> str:
    payload = json.dumps(asdict(manifest.normalized()), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_family_pools(
    records: Iterable[ExampleRecord],
) -> dict[str, list[ExampleRecord]]:
    pools: dict[str, list[ExampleRecord]] = {}
    for record in records:
        family = str(
            (record.meta or {}).get("source_family") or classify_source_family(record)
        )
        pools.setdefault(family, []).append(record)
    return pools


def sample_mixture_batch(
    records: list[ExampleRecord],
    *,
    weights: dict[str, float],
    batch_size: int,
    rng: random.Random,
    pools: dict[str, list[ExampleRecord]] | None = None,
) -> list[ExampleRecord]:
    """Online weighted per-family draw (uses loop RNG for bit-exact resume)."""
    if batch_size <= 0:
        return []
    pools = pools or index_family_pools(records)
    manifest = MixtureManifest(mixture_id="runtime", weights=weights).normalized()
    usable = {
        family: members
        for family, members in pools.items()
        if members and family in manifest.weights
    }
    if not usable:
        # Fall back to uniform over all records when no weight matches.
        return [records[rng.randrange(len(records))] for _ in range(batch_size)]

    families = sorted(usable)
    family_weights = [manifest.weights[f] for f in families]
    total = sum(family_weights)
    family_weights = [w / total for w in family_weights]

    out: list[ExampleRecord] = []
    for _ in range(batch_size):
        family = rng.choices(families, weights=family_weights, k=1)[0]
        members = usable[family]
        out.append(members[rng.randrange(len(members))])
    return out


def local_probe_candidates(
    base: dict[str, float],
    *,
    vary: tuple[str, ...] = (
        "prompt_paraphrase",
        "layout_augment",
        "namespace_augment",
        "stress_adversarial",
    ),
    scales: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
) -> list[MixtureManifest]:
    """Vary synth/quality-tier weights while holding organic totals fixed."""
    organic = {
        k: v
        for k, v in base.items()
        if k
        in {
            "rico_real",
            "awwwards_real",
            "human_curated",
            "human_feedback",
        }
    }
    probes: list[MixtureManifest] = []
    for family in vary:
        if family not in base and family not in KNOWN_FAMILIES:
            continue
        for scale in scales:
            weights = dict(base)
            weights.update(organic)
            weights[family] = max(0.01, float(base.get(family, 0.05)) * scale)
            probes.append(
                MixtureManifest(
                    mixture_id=f"local_{family}_{scale:g}",
                    weights=weights,
                    notes=f"local probe: {family}×{scale:g}",
                ).normalized()
            )
    return probes


def global_probe_candidates(
    local_composition: dict[str, float],
    *,
    organic_totals: tuple[float, ...] = (0.4, 0.55, 0.7),
    feedback_totals: tuple[float, ...] = (0.05, 0.1, 0.15),
) -> list[MixtureManifest]:
    """Vary organic vs synthetic vs feedback totals with local mix frozen."""
    synth_keys = [
        k
        for k in local_composition
        if k
        in {
            "prompt_paraphrase",
            "layout_augment",
            "namespace_augment",
            "stress_adversarial",
            "self_distilled_success",
            "self_distilled_repair",
        }
    ]
    probes: list[MixtureManifest] = []
    for organic in organic_totals:
        for feedback in feedback_totals:
            synth = max(0.05, 1.0 - organic - feedback)
            weights: dict[str, float] = {}
            # Split organic evenly across present organic families.
            org_present = ["rico_real", "awwwards_real", "human_curated"]
            for k in org_present:
                weights[k] = organic / len(org_present)
            weights["human_feedback"] = feedback
            if synth_keys:
                local_sum = sum(local_composition.get(k, 0.0) for k in synth_keys) or 1.0
                for k in synth_keys:
                    weights[k] = synth * (local_composition.get(k, 0.0) / local_sum)
            else:
                weights["prompt_paraphrase"] = synth
            probes.append(
                MixtureManifest(
                    mixture_id=f"global_o{organic:g}_f{feedback:g}",
                    weights=weights,
                    notes=f"global probe organic={organic} feedback={feedback}",
                ).normalized()
            )
    return probes


def fit_weight_regression(
    rows: list[dict[str, Any]],
    *,
    families: list[str] | None = None,
) -> dict[str, Any]:
    """Linear regression NLL ← family weights (ordinary least squares).

    ``rows`` entries need ``weights`` (dict) and ``weighted_nll`` (float).
    """
    if not rows:
        raise ValueError("no rows for regression")
    families = families or sorted({k for row in rows for k in (row.get("weights") or {})})
    # Design matrix with bias column.
    x_rows: list[list[float]] = []
    y: list[float] = []
    for row in rows:
        nll = row.get("weighted_nll")
        if nll is None:
            continue
        w = row.get("weights") or {}
        x_rows.append([1.0] + [float(w.get(f, 0.0)) for f in families])
        y.append(float(nll))
    if len(x_rows) < 2:
        raise ValueError("need at least 2 scored mixtures for regression")

    # Solve (XᵀX)β = Xᵀy via Gaussian elimination (no numpy required).
    n_feat = len(families) + 1
    xtx = [[0.0] * n_feat for _ in range(n_feat)]
    xty = [0.0] * n_feat
    for xs, yi in zip(x_rows, y):
        for i in range(n_feat):
            xty[i] += xs[i] * yi
            for j in range(n_feat):
                xtx[i][j] += xs[i] * xs[j]
    beta = _solve_linear(xtx, xty)
    coefs = {families[i]: beta[i + 1] for i in range(len(families))}
    return {
        "intercept": beta[0],
        "coefficients": coefs,
        "families": families,
        "n": len(y),
    }


def propose_from_fit(
    fit: dict[str, Any],
    *,
    base: dict[str, float],
    n: int = 3,
    step: float = 0.05,
) -> list[MixtureManifest]:
    """Propose candidates that down-weight families with positive NLL coefficients."""
    coefs: dict[str, float] = dict(fit.get("coefficients") or {})
    ranked = sorted(coefs.items(), key=lambda kv: kv[1], reverse=True)
    proposals: list[MixtureManifest] = []
    for i in range(max(1, n)):
        weights = dict(base)
        for family, coef in ranked[: i + 1]:
            if coef <= 0:
                continue
            weights[family] = max(0.01, float(weights.get(family, 0.05)) - step * (i + 1))
        # Boost the most negative (helpful) families.
        helpful = sorted(coefs.items(), key=lambda kv: kv[1])
        for family, coef in helpful[: i + 1]:
            if coef >= 0:
                continue
            weights[family] = float(weights.get(family, 0.05)) + step * (i + 1)
        proposals.append(
            MixtureManifest(
                mixture_id=f"propose_{i + 1}",
                weights=weights,
                notes="regression-proposed",
            ).normalized()
        )
    return proposals


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            # Singular — ridge the diagonal and retry once.
            for i in range(n):
                a[i][i] += 1e-6
            return _solve_linear(a, b) if col == 0 else [0.0] * n
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            for j in range(col, n + 1):
                m[row][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def default_base_weights() -> dict[str, float]:
    return {
        "rico_real": 0.45,
        "human_curated": 0.15,
        "prompt_paraphrase": 0.10,
        "layout_augment": 0.10,
        "namespace_augment": 0.05,
        "human_feedback": 0.10,
        "stress_adversarial": 0.05,
    }
