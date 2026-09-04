"""Lever knobs and the fingerprints that identify a treatment.

One responsibility: reducing a set of experiment knobs to a stable identity --
the fingerprint two arms are compared by, the short token a slug carries, and
the lever categories a thrash bank may vary.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autotrain_io import (
    read_json,
)

LEVER_KNOB_KEYS = (
    "ltr_prefix_loss_weight",
    "component_token_loss_weight",
    "component_edge_token_loss_weight",
    "compiler_decision_token_loss_weight",
    "structure_token_loss_weight",
    "typed_family_balance_loss_weight",
    "ltr_tail_loss_weight",
    "compiler_alignment_loss_weight",
    "compiler_alignment_margin",
    "compiler_alignment_stratified",
    "compiler_alignment_semantic_exhaustive",
    "compiler_alignment_kind_filter",
    "grammar_completion_bounds",
    "grammar_equivalence_cache",
    "grammar_draft_window",
    "compact_active_canvas",
    "mixture_sampling_policy",
    "mixture_exposure_target_profile",
    "mixture_total_decision_budget",
    "mixture_per_root_cap",
    "mixture_per_template_cap",
    "mixture_max_importance_weight",
    "component_plan_loss_weight",
    "component_plan_decode_weight",
    "solver_energy_loss_weight",
    "solver_energy_decode_weight",
    "legal_edit_hazard_loss_weight",
    "legal_edit_hazard_decode_weight",
    "component_edge_loss_weight",
    "component_edge_alignment_loss_weight",
    "component_edge_decode_weight",
    "component_inventory_loss_weight",
    "component_inventory_decode_weight",
    "binder_topology_loss_weight",
    "binder_topology_decode_weight",
    "binder_component_plan_loss_weight",
    "binder_component_plan_decode_weight",
    "binder_arity_loss_weight",
    "binder_arity_decode_weight",
    "slot_component_loss_weight",
    "slot_component_decode_weight",
    "slot_contract_in_context",
    "constraint_graph_mode",
    "symbol_boundary_loss_weight",
    "design_md_dropout",
    "fidelity_loss_weight",
    "semantic_contrast_dir",
    "semantic_contrast_loss_weight",
    "semantic_contrast_margin",
    "semantic_contrast_fraction",
    "symbol_slot_augmentation",
    "mask_pattern",
    "structural_aux_head_profile",
    "compiler_decode_mode",
    "steps",
    "batch_size",
    "lr",
    "train_version",
    "context_backend",
    "sync_checkpoints",
    "local_files_only",
    "output_tokenizer",
)

FINGERPRINT_EXCLUDE_KEYS = frozenset({"steps"})

EXPERIMENT_ONLY_KNOB_CATEGORIES: dict[str, str] = {"train_version": "data"}


def bank_lever_categories() -> dict[str, str]:
    """``lever_catalog()`` categories plus the ExperimentKnobs-only knob keys."""
    from slm_training.levers import lever_catalog

    out = {
        name: str(spec.get("category") or "") for name, spec in lever_catalog().items()
    }
    out.update(EXPERIMENT_ONLY_KNOB_CATEGORIES)
    return out


def short_lever_token(key: str) -> str:
    token = key.replace("_loss_weight", "").replace("_weight", "").replace("_", "-")
    return token[:24]


def thrash_lever_signature(extras: dict[str, Any] | None) -> str:
    """Stable identity for thrash recipes (excludes measurement / private keys)."""
    raw = {
        k: v
        for k, v in (extras or {}).items()
        if not str(k).startswith("_")
        and k
        not in {
            "seed",
            "steps",
            "decode_timeout_seconds",
            "generate_batch_size",
            "latency_probe_records",
            "latency_probe_planned_n",
        }
    }
    # Prefer registered lever subset when present so static/dynamic arms align.
    levers = lever_knobs(raw)
    payload = levers if levers else raw
    return knobs_fingerprint(payload)


def lever_knobs(knobs: dict[str, Any] | None) -> dict[str, Any]:
    """Stable lever subset for confirm retests (excludes seed / measurement)."""
    if not isinstance(knobs, dict):
        return {}
    out: dict[str, Any] = {}
    for key in LEVER_KNOB_KEYS:
        if key in knobs and knobs[key] is not None:
            out[key] = knobs[key]
    return out


def knobs_fingerprint(levers: dict[str, Any]) -> str:
    """Identity hash for champion dedup (excludes steps cycle jitter)."""
    stable = {
        k: v for k, v in (levers or {}).items() if k not in FINGERPRINT_EXCLUDE_KEYS
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def matrix_experiment_knobs(
    matrix: Mapping[str, Any], experiment_id: str
) -> dict[str, Any]:
    by_id = {
        str((row.get("experiment") or {}).get("experiment_id") or ""): dict(
            (row.get("experiment") or {}).get("knobs") or {}
        )
        for row in matrix.get("hypotheses") or []
        if isinstance(row, dict)
    }
    return by_id.get(experiment_id, {})


def matrix_treatment_signature(
    matrix: Mapping[str, Any], candidate_id: str, control_id: str
) -> str | None:
    """Hash only the candidate-vs-control lever delta for timeout retirement."""

    candidate = lever_knobs(matrix_experiment_knobs(matrix, candidate_id))
    control = lever_knobs(matrix_experiment_knobs(matrix, control_id))
    if not candidate:
        return None
    treatment = {
        key: candidate.get(key)
        for key in sorted(set(candidate) | set(control))
        if candidate.get(key) != control.get(key)
    }
    if not treatment:
        return None
    body = json.dumps(treatment, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_experiment_knobs(camp_dir: Path, experiment_id: str) -> dict[str, Any]:
    exp_path = camp_dir / "artifacts" / "experiments" / f"{experiment_id}.json"
    if not exp_path.is_file():
        # Some writers store experiment_id as stem without full path.
        matches = list((camp_dir / "artifacts" / "experiments").glob("*.json"))
        for path in matches:
            data = read_json(path)
            if data.get("experiment_id") == experiment_id:
                knobs = data.get("knobs")
                return knobs if isinstance(knobs, dict) else {}
        return {}
    data = read_json(exp_path)
    knobs = data.get("knobs")
    return knobs if isinstance(knobs, dict) else {}
