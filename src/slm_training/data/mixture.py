"""Mixture manifests and online family-weighted sampling (P1b)."""

from __future__ import annotations

import hashlib
import json
import math
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
    # Scope-graded identity anchors (echo pairs) sample as their own task
    # axis so diffusion/mixture training can dial memorization separately.
    "identity_echo": ("identity",),
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
    "scope_contract",
    "scope_identity_document",
    "scope_identity_statement",
    "scope_identity_expression",
    "scope_identity_lexical",
    "scope_canonical_document",
    "scope_canonical_statement",
    "scope_canonical_expression",
    "scope_canonical_lexical",
    "scope_repair_statement",
    "scope_repair_expression",
    "scope_repair_lexical",
    "lexical_typed_map",
)
ORGANIC_FAMILIES = (
    "rico_real",
    "awwwards_real",
    "human_curated",
)
FEEDBACK_FAMILIES = ("human_feedback",)
SAMPLING_POLICIES = frozenset(
    {"with_replacement", "capacity_aware", "quota_capacity_aware", "exposure_targeted"}
)
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
    if isinstance(data.get("manifest"), dict):
        data = data["manifest"]
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


def record_task(record: ExampleRecord) -> str:
    """Return the explicit task or the output contract's generation default."""
    task = str((record.meta or {}).get("task") or "")
    return task or "generation"


def index_task_family_pools(
    records: Iterable[ExampleRecord],
) -> dict[str, dict[str, list[ExampleRecord]]]:
    pools: dict[str, dict[str, list[ExampleRecord]]] = {}
    for record in records:
        group = task_group(record_task(record))
        if group is None:
            continue
        family = str(
            (record.meta or {}).get("source_family") or classify_source_family(record)
        )
        pools.setdefault(group, {}).setdefault(family, []).append(record)
    return pools


def record_action_counts(record: ExampleRecord) -> Counter[str]:
    """Extract component names and placeholders from a record's OpenUI target."""
    counts: Counter[str] = Counter(_COMPONENT_RE.findall(record.openui))
    counts.update(record.placeholders or ())
    return counts


def build_exposure_ledger(
    records: list[ExampleRecord],
    selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build per-action and aggregate exposure statistics for a corpus or selection.

    The ledger is audit-only: it reports raw counts and diversity caps without
    changing the sampler.  ``selected_ids`` restricts the audit to a sampled
    subset (e.g. the records chosen by ``_sample_exposure_targeted``).
    """
    selected_ids = selected_ids or {r.id for r in records}
    action_records: dict[str, list[ExampleRecord]] = {}
    action_counts: Counter[str] = Counter()
    roots: dict[str, set[str | None]] = {}
    templates: dict[str, set[str | None]] = {}
    total_decisions = 0

    for record in records:
        counts = record_action_counts(record)
        if not counts:
            continue
        total_decisions += sum(counts.values())
        meta = record.meta or {}
        root_id = str(meta.get("root_parent_id") or meta.get("parent_id") or record.id)
        template_key = str(meta.get("parent_id") or meta.get("source_family") or record.source)
        for action, count in counts.items():
            action_records.setdefault(action, []).append(record)
            action_counts[action] += count
            roots.setdefault(action, set()).add(root_id)
            templates.setdefault(action, set()).add(template_key)

    actions = {
        action: {
            "raw_record_count": len(
                [r for r in action_records.get(action, []) if r.id in selected_ids]
            ),
            "raw_target_decision_count": sum(
                record_action_counts(r).get(action, 0)
                for r in action_records.get(action, [])
                if r.id in selected_ids
            ),
            "unique_root_count": len(roots.get(action, set())),
            "unique_prompt_template_count": len(templates.get(action, set())),
        }
        for action in sorted(action_counts)
    }

    selected_records = [r for r in records if r.id in selected_ids]
    selected_counts: Counter[str] = Counter()
    for record in selected_records:
        selected_counts.update(record_action_counts(record))

    return {
        "actions": actions,
        "observed_decisions_per_run": dict(sorted(selected_counts.items())),
        "aggregate": {
            "total_records": len(records),
            "selected_records": len(selected_records),
            "total_decisions": total_decisions,
            "selected_decisions": sum(selected_counts.values()),
            "unique_actions": len(action_counts),
        },
    }


def _sample_exposure_targeted(
    records: list[ExampleRecord],
    weights: dict[str, float],
    action_targets: dict[str, float],
    total_decision_budget: int,
    per_root_cap: int | None,
    per_template_cap: int | None,
    max_importance_weight: float,
    rng: random.Random,
) -> list[ExampleRecord]:
    """Greedy rare-action sampler with bounded importance weights and diversity caps.

    Each selected record contributes its action counts toward the per-action
    targets.  Selection stops when the total record budget is exhausted, all
    targets are met, or no remaining record can be added without violating a cap.
    """
    if total_decision_budget <= 0 or not records:
        return []

    record_counts = [record_action_counts(r) for r in records]
    max_iw = max(1.0, float(max_importance_weight))
    no_cap = len(records)
    per_root_cap = no_cap if per_root_cap is None else max(1, int(per_root_cap))
    per_template_cap = no_cap if per_template_cap is None else max(1, int(per_template_cap))

    current: Counter[str] = Counter()
    root_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    selected: list[ExampleRecord] = []
    selected_set: set[int] = set()

    targets = {a: float(t) for a, t in action_targets.items() if t > 0}
    if not targets:
        # No targets supplied: fall back to uniform random within the budget.
        return [
            records[rng.randrange(len(records))]
            for _ in range(min(total_decision_budget, len(records)))
        ]

    def _gain(index: int) -> tuple[float, int]:
        counts = record_counts[index]
        if not counts:
            return (0.0, index)
        gain = 0.0
        for action, count in counts.items():
            target = targets.get(action, 0.0)
            deficit = max(0.0, target - current[action])
            # Importance weight is target-normalized and capped.
            importance = min(max_iw, target / max(1.0, current[action]))
            gain += deficit * importance * count
        return (gain, index)

    while len(selected) < total_decision_budget:
        best_gain = -1.0
        best_index = -1
        for index, record in enumerate(records):
            if index in selected_set:
                continue
            meta = record.meta or {}
            root_id = str(meta.get("root_parent_id") or meta.get("parent_id") or record.id)
            template_key = str(
                meta.get("parent_id") or meta.get("source_family") or record.source
            )
            if root_counts[root_id] >= per_root_cap:
                continue
            if template_counts[template_key] >= per_template_cap:
                continue
            gain, _ = _gain(index)
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index < 0 or best_gain <= 0:
            break
        selected_set.add(best_index)
        selected.append(records[best_index])
        current.update(record_counts[best_index])
        meta = records[best_index].meta or {}
        root_id = str(meta.get("root_parent_id") or meta.get("parent_id") or records[best_index].id)
        template_key = str(
            meta.get("parent_id") or meta.get("source_family") or records[best_index].source
        )
        root_counts[root_id] += 1
        template_counts[template_key] += 1

    return selected


def _build_default_action_targets(
    records: list[ExampleRecord],
    weights: dict[str, float],
    total_decision_budget: int,
    max_importance_weight: float,
) -> dict[str, float]:
    """Sqrt-inverse-frequency capped targets derived from family weights."""
    if not records:
        return {}
    family_pools = index_family_pools(records)
    action_family_counts: dict[str, Counter[str]] = {}
    for family, members in family_pools.items():
        family_weight = float(weights.get(family, 0.0))
        if family_weight <= 0:
            continue
        for record in members:
            for action, count in record_action_counts(record).items():
                action_family_counts.setdefault(action, Counter())[family] += (
                    count * family_weight
                )

    if not action_family_counts:
        # No weighted actions: fall back to raw corpus frequencies.
        for record in records:
            for action, count in record_action_counts(record).items():
                action_family_counts.setdefault(action, Counter())["_corpus"] += count

    max_iw = max(1.0, float(max_importance_weight))
    raw: dict[str, float] = {}
    for action, family_counter in action_family_counts.items():
        weighted_freq = sum(family_counter.values())
        raw[action] = min(max_iw, 1.0 / math.sqrt(max(1.0, weighted_freq)))

    total = sum(raw.values()) or 1.0
    return {a: total_decision_budget * v / total for a, v in raw.items()}


def sample_mixture_batch(
    records: list[ExampleRecord],
    *,
    weights: dict[str, float],
    batch_size: int,
    rng: random.Random,
    pools: dict[str, list[ExampleRecord]] | None = None,
    task_weights: dict[str, float] | None = None,
    task_pools: dict[str, dict[str, list[ExampleRecord]]] | None = None,
    sampling_policy: str = "with_replacement",
    **kwargs: Any,
) -> list[ExampleRecord]:
    """Online weighted per-family draw (uses loop RNG for bit-exact resume)."""
    if batch_size <= 0:
        return []
    if sampling_policy not in SAMPLING_POLICIES:
        raise ValueError(f"unknown mixture sampling policy: {sampling_policy!r}")
    if sampling_policy == "exposure_targeted":
        action_targets = kwargs.get("action_targets")
        total_decision_budget = int(
            kwargs.get("total_decision_budget") or batch_size
        )
        per_root_cap = kwargs.get("per_root_cap")
        per_template_cap = kwargs.get("per_template_cap")
        max_importance_weight = float(kwargs.get("max_importance_weight") or 10.0)
        if not action_targets:
            action_targets = _build_default_action_targets(
                records,
                weights=weights,
                total_decision_budget=total_decision_budget,
                max_importance_weight=max_importance_weight,
            )
        if not action_targets:
            # Nothing to target: retain the legacy default sampler.
            return sample_mixture_batch(
                records,
                weights=weights,
                batch_size=batch_size,
                rng=rng,
                pools=pools,
                task_weights=task_weights,
                task_pools=task_pools,
                sampling_policy="with_replacement",
            )
        selected = _sample_exposure_targeted(
            records,
            weights=weights,
            action_targets=action_targets,
            total_decision_budget=total_decision_budget,
            per_root_cap=per_root_cap,
            per_template_cap=per_template_cap,
            max_importance_weight=max_importance_weight,
            rng=rng,
        )
        # Pad only if the greedy selection under-ran the requested batch size,
        # and only with records that do not violate the diversity caps.
        if len(selected) < batch_size:
            shortfall = batch_size - len(selected)
            selected_ids = {r.id for r in selected}
            root_counts: Counter[str] = Counter()
            template_counts: Counter[str] = Counter()
            for record in selected:
                meta = record.meta or {}
                root_counts[
                    str(meta.get("root_parent_id") or meta.get("parent_id") or record.id)
                ] += 1
                template_counts[
                    str(
                        meta.get("parent_id")
                        or meta.get("source_family")
                        or record.source
                    )
                ] += 1
            per_root_cap = (
                len(records) if kwargs.get("per_root_cap") is None else int(kwargs["per_root_cap"])
            )
            per_template_cap = (
                len(records) if kwargs.get("per_template_cap") is None else int(kwargs["per_template_cap"])
            )
            candidates: list[ExampleRecord] = []
            for record in records:
                if record.id in selected_ids:
                    continue
                meta = record.meta or {}
                root_key = str(
                    meta.get("root_parent_id") or meta.get("parent_id") or record.id
                )
                template_key = str(
                    meta.get("parent_id")
                    or meta.get("source_family")
                    or record.source
                )
                if (
                    root_counts[root_key] < per_root_cap
                    and template_counts[template_key] < per_template_cap
                ):
                    candidates.append(record)
            rng.shuffle(candidates)
            selected.extend(candidates[:shortfall])
        return selected
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
            if sampling_policy == "quota_capacity_aware":
                return _sample_quota_capacity_aware_tasks(
                    usable_tasks,
                    group_weights=dict(zip(groups, group_weights, strict=True)),
                    family_weights=manifest.weights,
                    batch_size=batch_size,
                    rng=rng,
                )
            if sampling_policy == "capacity_aware":
                weighted_records: list[tuple[ExampleRecord, float]] = []
                for group, group_weight in zip(groups, group_weights, strict=True):
                    family_pools = usable_tasks[group]
                    families = sorted(family_pools)
                    family_weights = [
                        manifest.weights.get(family, 0.0) for family in families
                    ]
                    if not any(family_weights):
                        family_weights = [1.0] * len(families)
                    family_total = sum(family_weights)
                    for family, family_weight in zip(
                        families, family_weights, strict=True
                    ):
                        members = family_pools[family]
                        row_weight = (
                            group_weight * family_weight / family_total / len(members)
                        )
                        weighted_records.extend(
                            (member, row_weight) for member in members
                        )
                return _sample_capacity_aware(
                    weighted_records, batch_size=batch_size, rng=rng
                )
            # Family order/weights per group are draw-invariant; hoist them out
            # of the per-draw loop (mirrors the non-task path below). Consumes
            # no RNG, so the draw sequence is unchanged.
            group_families: dict[str, tuple[list[str], list[float]]] = {}
            for group in groups:
                families = sorted(usable_tasks[group])
                family_weights = [
                    manifest.weights.get(family, 0.0) for family in families
                ]
                if not any(family_weights):
                    family_weights = [1.0] * len(families)
                group_families[group] = (families, family_weights)
            out: list[ExampleRecord] = []
            for _ in range(batch_size):
                group = rng.choices(groups, weights=group_weights, k=1)[0]
                family_pools = usable_tasks[group]
                families, family_weights = group_families[group]
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
        if sampling_policy == "capacity_aware":
            return _sample_capacity_aware(
                [(record, 1.0) for record in records],
                batch_size=batch_size,
                rng=rng,
            )
        return [records[rng.randrange(len(records))] for _ in range(batch_size)]

    families = sorted(usable)
    family_weights = [manifest.weights[f] for f in families]
    total = sum(family_weights)
    family_weights = [w / total for w in family_weights]

    if sampling_policy == "capacity_aware":
        weighted_records = [
            (member, family_weight / len(usable[family]))
            for family, family_weight in zip(families, family_weights, strict=True)
            for member in usable[family]
        ]
        return _sample_capacity_aware(
            weighted_records, batch_size=batch_size, rng=rng
        )
    if sampling_policy == "quota_capacity_aware":
        return _sample_quota_capacity_aware_families(
            usable,
            family_weights=dict(zip(families, family_weights, strict=True)),
            batch_size=batch_size,
            rng=rng,
        )

    out: list[ExampleRecord] = []
    for _ in range(batch_size):
        family = rng.choices(families, weights=family_weights, k=1)[0]
        members = usable[family]
        out.append(members[rng.randrange(len(members))])
    return out


def _sample_capacity_aware(
    weighted_records: list[tuple[ExampleRecord, float]],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[ExampleRecord]:
    """Weighted draws without replacement, restarting only after pool exhaustion."""
    if not weighted_records:
        return []
    out: list[ExampleRecord] = []
    while len(out) < batch_size:
        remaining = list(weighted_records)
        # Parallel weight list popped in lockstep with `remaining`: same
        # per-draw weight sequence (bit-exact RNG draws) without rebuilding
        # the O(n) list on every draw.
        remaining_weights = [weight for _, weight in remaining]
        cycle_size = min(batch_size - len(out), len(remaining))
        for _ in range(cycle_size):
            index = rng.choices(
                range(len(remaining)),
                weights=remaining_weights,
                k=1,
            )[0]
            record, _ = remaining.pop(index)
            remaining_weights.pop(index)
            out.append(record)
    return out


def _apportion_with_capacity(
    total: int,
    *,
    weights: dict[str, float],
    capacities: dict[str, int],
) -> dict[str, int]:
    """Largest-remainder quotas with deterministic capacity redistribution."""
    quotas = {key: 0 for key in sorted(capacities)}
    while sum(quotas.values()) < total:
        active = [key for key in quotas if quotas[key] < capacities[key]]
        if not active:
            break
        remaining = total - sum(quotas.values())
        active_total = sum(max(0.0, weights.get(key, 0.0)) for key in active)
        effective = {
            key: (
                max(0.0, weights.get(key, 0.0))
                if active_total > 0
                else 1.0
            )
            for key in active
        }
        effective_total = sum(effective.values())
        ideals = {
            key: remaining * effective[key] / effective_total for key in active
        }
        for key in active:
            add = min(capacities[key] - quotas[key], int(ideals[key]))
            if add:
                quotas[key] += add
        if sum(quotas.values()) >= total:
            break
        candidates = sorted(
            (key for key in active if quotas[key] < capacities[key]),
            key=lambda item: (-(ideals[item] % 1), -effective[item], item),
        )
        if not candidates:
            break
        for key in candidates[: total - sum(quotas.values())]:
            quotas[key] += 1
    return quotas


def _sample_quota_capacity_aware_tasks(
    pools: dict[str, dict[str, list[ExampleRecord]]],
    *,
    group_weights: dict[str, float],
    family_weights: dict[str, float],
    batch_size: int,
    rng: random.Random,
) -> list[ExampleRecord]:
    out: list[ExampleRecord] = []
    while len(out) < batch_size:
        cycle_size = min(
            batch_size - len(out),
            sum(len(rows) for families in pools.values() for rows in families.values()),
        )
        group_quotas = _apportion_with_capacity(
            cycle_size,
            weights=group_weights,
            capacities={
                group: sum(len(rows) for rows in families.values())
                for group, families in pools.items()
            },
        )
        cycle: list[ExampleRecord] = []
        for group in sorted(group_quotas):
            families = pools[group]
            family_quotas = _apportion_with_capacity(
                group_quotas[group],
                weights=family_weights,
                capacities={family: len(rows) for family, rows in families.items()},
            )
            for family in sorted(family_quotas):
                rows = list(families[family])
                rng.shuffle(rows)
                cycle.extend(rows[: family_quotas[family]])
        rng.shuffle(cycle)
        out.extend(cycle)
    return out


def _sample_quota_capacity_aware_families(
    pools: dict[str, list[ExampleRecord]],
    *,
    family_weights: dict[str, float],
    batch_size: int,
    rng: random.Random,
) -> list[ExampleRecord]:
    out: list[ExampleRecord] = []
    while len(out) < batch_size:
        cycle_size = min(batch_size - len(out), sum(map(len, pools.values())))
        quotas = _apportion_with_capacity(
            cycle_size,
            weights=family_weights,
            capacities={family: len(rows) for family, rows in pools.items()},
        )
        cycle: list[ExampleRecord] = []
        for family in sorted(quotas):
            rows = list(pools[family])
            rng.shuffle(rows)
            cycle.extend(rows[: quotas[family]])
        rng.shuffle(cycle)
        out.extend(cycle)
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
        "scope_contract": 0.02,
        "prompt_paraphrase": 0.03,
        "layout_augment": 0.025,
        "stress_adversarial": 0.025,
        "self_distilled_success": 0.02,
        "self_distilled_repair": 0.02,
        "gold_correction": 0.015,
        # Scope-graded families: canonical outputs carry the highest new
        # weights (deliberate ranking bias toward tree-optimized targets).
        "scope_identity_document": 0.02,
        "scope_identity_statement": 0.02,
        "scope_identity_expression": 0.02,
        "scope_identity_lexical": 0.02,
        "scope_canonical_document": 0.03,
        "scope_canonical_statement": 0.03,
        "scope_canonical_expression": 0.03,
        "scope_canonical_lexical": 0.03,
        "scope_repair_statement": 0.02,
        "scope_repair_expression": 0.02,
        "scope_repair_lexical": 0.02,
        "lexical_typed_map": 0.02,
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
        task = record_task(record)
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
