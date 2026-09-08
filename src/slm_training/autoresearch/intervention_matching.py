"""Claim-specific paired launch matching; never a promotion permission."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from slm_training.autoresearch.experiment_identity import InterventionContract, digest
from slm_training.levers import CAPACITY_SCALING_LEVERS, require_size_matched_arms

DATA_FIELDS = frozenset({"train_version", "train_dir", "data_manifest_sha", "mixture_weights"})
DURATION_FIELDS = frozenset({"steps", "batch_size", "grad_accum_steps"})
_FIXED = frozenset({"seed", "initialize_from", "tokenizer_layout", "output_tokenizer"})


def assert_intervention_match(
    control: Mapping[str, Any], candidate: Mapping[str, Any], *,
    contract: InterventionContract, resource_totals: tuple[float, float],
    trainable_params: tuple[int, int],
) -> None:
    """Allow only declared changes; match actual planned common resource units.

The data owner still certifies readiness/leakage and the promotion owner still
requires EG_params for growth. Neither permission is issued here.
"""
    digest([dict(control), dict(candidate)])  # Reject nonfinite observations early.
    varied = set(contract.varied_fields)
    actual = {key for key in control.keys() | candidate.keys()
              if control.get(key) != candidate.get(key)}
    if not actual or actual != varied:
        raise ValueError(f"declared/actual intervention mismatch: {sorted(actual ^ varied)}")
    fixed = _FIXED - ({"seed"} if contract.kind == "randomness" else set())
    if varied & fixed:
        raise ValueError("paired arms changed checkpoint, seed or tokenizer contract")
    if varied & DATA_FIELDS and contract.kind != "data":
        raise ValueError("undeclared data intervention")
    if varied & DURATION_FIELDS and contract.kind != "duration":
        raise ValueError("undeclared duration intervention")
    if contract.kind == "data" and not varied <= DATA_FIELDS:
        raise ValueError("data intervention changes non-data fields")
    if contract.kind == "duration" and not varied <= DURATION_FIELDS:
        raise ValueError("duration intervention changes non-duration fields")
    if contract.kind != "capacity":
        require_size_matched_arms(control, candidate, context="declared_intervention")
    elif not varied <= CAPACITY_SCALING_LEVERS.keys():
        raise ValueError("capacity intervention changes non-capacity fields")
    if contract.resource_basis == "updates" and tuple(resource_totals) != (control.get("steps"), candidate.get("steps")):
        raise ValueError("declared update totals disagree with resolved recipe")
    _match_resources(resource_totals, trainable_params, contract)


def _match_resources(totals, params, contract: InterventionContract) -> None:
    if len(totals) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                               for v in totals):
        raise ValueError("resource totals must be two finite positive planned quantities")
    if totals[0] != totals[1]:
        raise ValueError(f"unmatched declared {contract.resource_basis} resource totals")
    if len(params) != 2 or any(type(v) is not int or v <= 0 for v in params):
        raise ValueError("measured trainable parameter counts required")
    if params[0] != params[1] and contract.kind != "capacity":
        raise ValueError("unmatched parameter capacity")

