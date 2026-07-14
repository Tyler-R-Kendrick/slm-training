"""Mixture manifests and online family-weighted sampling (P1b)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.catalog import (
    KNOWN_FAMILIES,
    classify_source_family,
)

TASK_GROUPS = {
    "generation": ("generation",),
    "repair_completion_inpaint": ("repair", "completion", "inpaint"),
    "patch_edit": ("patch", "edit"),
    "state_behavior": ("state", "behavior"),
    "noop_adversarial": ("noop", "adversarial"),
}
DEFAULT_TASK_WEIGHTS = {group: 0.2 for group in TASK_GROUPS}
NEW_FAMILIES = (
    "programspec_generated",
    "language_contract",
    "corruption_repair",
    "edit_trajectory",
    "frontier_described",
    "frontier_semantic",
    "frontier_product",
    "frontier_user",
    "frontier_simplified",
    "abstraction_ladder",
    "renderer_visual",
    "web_distilled",
    "diffusion_corruption",
)
ORGANIC_FAMILIES = (
    "rico_real",
    "awwwards_real",
    "human_curated",
)
FEEDBACK_FAMILIES = ("human_feedback",)
_COMPONENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")


@dataclass(frozen=True)
class MixtureManifest:
    mixture_id: str
    weights: dict[str, float]
    task_weights: dict[str, float] | None = None
    notes: str = ""
    version: int = 2

    def normalized(self) -> MixtureManifest:
        clean = {k: float(v) for k, v in self.weights.items() if float(v) > 0}
        total = sum(clean.values())
        if total <= 0:
            raise ValueError("mixture weights must sum to a positive value")
        task_weights = None
        if self.task_weights:
            unknown = sorted(set(self.task_weights) - set(TASK_GROUPS))
            if unknown:
                raise ValueError(f"unknown task groups: {unknown}")
            positive_tasks = {
                key: float(value)
                for key, value in self.task_weights.items()
                if float(value) > 0
            }
            task_total = sum(positive_tasks.values())
            if task_total <= 0:
                raise ValueError("task weights must sum to a positive value")
            task_weights = {
                key: float(value) / task_total
                for key, value in sorted(positive_tasks.items())
            }
        return MixtureManifest(
            mixture_id=self.mixture_id,
            weights={k: v / total for k, v in sorted(clean.items())},
            task_weights=task_weights,
            notes=self.notes,
            version=self.version,
        )


def load_mixture_manifest(path: Path | str) -> MixtureManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MixtureManifest(
        mixture_id=str(data.get("mixture_id") or Path(path).stem),
        weights={str(k): float(v) for k, v in (data.get("weights") or {}).items()},
        task_weights=(
            {str(k): float(v) for k, v in data["task_weights"].items()}
            if data.get("task_weights")
            else None
        ),
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
    payload = json.dumps(
        asdict(manifest.normalized()), sort_keys=True, separators=(",", ":")
    )
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


def task_group(task: str | None) -> str | None:
    for group, tasks in TASK_GROUPS.items():
        if task in tasks:
            return group
    return None


def index_task_family_pools(
    records: Iterable[ExampleRecord],
) -> dict[str, dict[str, list[ExampleRecord]]]:
    pools: dict[str, dict[str, list[ExampleRecord]]] = {}
    for record in records:
        group = task_group(str((record.meta or {}).get("task") or ""))
        if group is None:
            continue
        family = str(
            (record.meta or {}).get("source_family") or classify_source_family(record)
        )
        pools.setdefault(group, {}).setdefault(family, []).append(record)
    return pools


def sample_mixture_batch(
    records: list[ExampleRecord],
    *,
    weights: dict[str, float],
    batch_size: int,
    rng: random.Random,
    pools: dict[str, list[ExampleRecord]] | None = None,
    task_weights: dict[str, float] | None = None,
    task_pools: dict[str, dict[str, list[ExampleRecord]]] | None = None,
) -> list[ExampleRecord]:
    """Online weighted per-family draw (uses loop RNG for bit-exact resume)."""
    if batch_size <= 0:
        return []
    pools = pools or index_family_pools(records)
    manifest = MixtureManifest(mixture_id="runtime", weights=weights).normalized()
    if task_weights:
        task_manifest = MixtureManifest(
            mixture_id="runtime",
            weights=weights,
            task_weights=task_weights,
        ).normalized()
        task_pools = task_pools or index_task_family_pools(records)
        usable_tasks = {
            group: family_pools
            for group, family_pools in task_pools.items()
            if family_pools and group in (task_manifest.task_weights or {})
        }
        if usable_tasks:
            groups = sorted(usable_tasks)
            group_weights = [task_manifest.task_weights[group] for group in groups]
            out: list[ExampleRecord] = []
            for _ in range(batch_size):
                group = rng.choices(groups, weights=group_weights, k=1)[0]
                family_pools = usable_tasks[group]
                families = sorted(family_pools)
                family_weights = [
                    manifest.weights.get(family, 0.0) for family in families
                ]
                if not any(family_weights):
                    family_weights = [1.0] * len(families)
                family = rng.choices(families, weights=family_weights, k=1)[0]
                members = family_pools[family]
                out.append(members[rng.randrange(len(members))])
            return out
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
    vary: tuple[str, ...] | None = None,
    scales: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    task_weights: dict[str, float] | None = None,
) -> list[MixtureManifest]:
    """Vary synth/quality-tier weights while holding organic totals fixed."""
    vary = vary or tuple(sorted(base))
    probes: list[MixtureManifest] = []
    for family in vary:
        if family not in base and family not in KNOWN_FAMILIES:
            continue
        for scale in scales:
            if scale == 1.0:
                continue
            weights = dict(base)
            weights[family] = max(0.01, float(base.get(family, 0.05)) * scale)
            probes.append(
                MixtureManifest(
                    mixture_id=f"local_{family}_{scale:g}",
                    weights=weights,
                    task_weights=task_weights,
                    notes=f"local probe: {family}×{scale:g}",
                ).normalized()
            )
    return probes


def global_probe_candidates(
    local_composition: dict[str, float],
    *,
    organic_totals: tuple[float, ...] = (0.4, 0.55, 0.7),
    feedback_totals: tuple[float, ...] = (0.05, 0.1, 0.15),
    task_weights: dict[str, float] | None = None,
) -> list[MixtureManifest]:
    """Vary organic vs synthetic vs feedback totals with local mix frozen."""
    organic_keys = [key for key in ORGANIC_FAMILIES if key in local_composition]
    feedback_keys = [key for key in FEEDBACK_FAMILIES if key in local_composition]
    synth_keys = [
        key
        for key in local_composition
        if key not in set(organic_keys) | set(feedback_keys)
    ]
    probes: list[MixtureManifest] = []
    for organic in organic_totals:
        for feedback in feedback_totals:
            synth = max(0.05, 1.0 - organic - feedback)
            weights: dict[str, float] = {}
            # Split organic evenly across present organic families.
            for key in organic_keys:
                denom = sum(local_composition[k] for k in organic_keys) or 1.0
                weights[key] = organic * local_composition[key] / denom
            for key in feedback_keys:
                denom = sum(local_composition[k] for k in feedback_keys) or 1.0
                weights[key] = feedback * local_composition[key] / denom
            if synth_keys:
                local_sum = (
                    sum(local_composition.get(k, 0.0) for k in synth_keys) or 1.0
                )
                for k in synth_keys:
                    weights[k] = synth * (local_composition.get(k, 0.0) / local_sum)
            else:
                weights["prompt_paraphrase"] = synth
            probes.append(
                MixtureManifest(
                    mixture_id=f"global_o{organic:g}_f{feedback:g}",
                    weights=weights,
                    task_weights=task_weights,
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
    families = families or sorted(
        {k for row in rows for k in (row.get("weights") or {})}
    )
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
    for index in range(1, n_feat):
        xtx[index][index] += 1e-6
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
    task_weights: dict[str, float] | None = None,
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
            weights[family] = max(
                0.01, float(weights.get(family, 0.05)) - step * (i + 1)
            )
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
                task_weights=task_weights,
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
        "rico_real": 0.12,
        "awwwards_real": 0.04,
        "human_curated": 0.08,
        "human_feedback": 0.06,
        "programspec_generated": 0.12,
        "language_contract": 0.08,
        "corruption_repair": 0.06,
        "edit_trajectory": 0.06,
        "frontier_described": 0.01,
        "frontier_semantic": 0.01,
        "frontier_product": 0.01,
        "frontier_user": 0.01,
        "frontier_simplified": 0.01,
        "abstraction_ladder": 0.06,
        "renderer_visual": 0.05,
        "web_distilled": 0.03,
        "diffusion_corruption": 0.02,
        "prompt_paraphrase": 0.03,
        "layout_augment": 0.025,
        "namespace_augment": 0.015,
        "stress_adversarial": 0.025,
        "self_distilled_success": 0.02,
        "self_distilled_repair": 0.02,
        "gold_correction": 0.015,
    }


def corpus_diagnostics(
    records: Iterable[ExampleRecord],
    *,
    configured_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows = list(records)
    family_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    program_families: set[str] = set()
    structures: set[str] = set()
    components: Counter[str] = Counter()
    unclassified = 0
    from slm_training.data.leakage import fingerprint_openui_structure

    for record in rows:
        meta = record.meta or {}
        family = str(meta.get("source_family") or classify_source_family(record))
        family_counts[family] += 1
        task = str(meta.get("task") or "")
        group = task_group(task)
        if group is None:
            unclassified += 1
        else:
            task_counts[task] += 1
            group_counts[group] += 1
        if meta.get("program_family_id"):
            program_families.add(str(meta["program_family_id"]))
        structures.add(fingerprint_openui_structure(record.openui))
        components.update(_COMPONENT_RE.findall(record.openui))

    configured = set(configured_weights or {})
    present = set(family_counts)
    return {
        "records": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "task_group_counts": dict(sorted(group_counts.items())),
        "unclassified_tasks": unclassified,
        "unique_program_families": len(program_families),
        "unique_structural_families": len(structures),
        "observed_component_counts": dict(sorted(components.items())),
        "configured_but_absent_families": sorted(configured - present),
        "present_but_unweighted_families": sorted(present - configured),
    }
