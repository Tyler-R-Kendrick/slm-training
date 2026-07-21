"""SLM-261: typed loss ledger for TwoTower training-loss decomposition.

The ledger is built from the flat scalar fields already emitted by
:meth:`TwoTowerModel.training_loss`.  It is intentionally independent of the
model implementation so experiment harnesses can reconstruct it from saved
metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "LOSS_LEDGER_SCHEMA_VERSION",
    "LossTermV1",
    "LossLedgerV1",
]

LOSS_LEDGER_SCHEMA_VERSION = "LossLedgerV1"

# Names used in `last_training_metrics`.  Each term is expected to provide a raw
# loss value and a coefficient; the ledger multiplies them to obtain the
# contribution to the total objective.  Weights may be zero (inactive terms).
_KNOWN_TERMS: tuple[tuple[str, str, str], ...] = (
    ("principal_mask_ce", "primary_final_reconstruction_loss", "primary_final_reconstruction_loss_weight"),
    ("recursive_depth_supervision", "recursive_depth_supervision_unweighted_loss", "recursive_depth_supervision_loss_weight"),
    ("diffusion_length", "diffusion_length_loss", "diffusion_length_loss_weight"),
    ("fidelity", "fidelity_loss", "fidelity_loss_weight"),
    ("symbol_boundary", "symbol_boundary_loss", "symbol_boundary_loss_weight"),
    ("ltr", "ltr_loss", "ltr_loss_weight"),
    ("compiler_alignment", "compiler_alignment_loss", "compiler_alignment_loss_weight"),
    ("component_inventory", "component_inventory_loss", "component_inventory_loss_weight"),
    ("component_plan", "component_plan_loss", "component_plan_loss_weight"),
    ("slot_component", "slot_component_loss", "slot_component_loss_weight"),
    ("component_edge", "component_edge_loss", "component_edge_loss_weight"),
    ("binder_arity", "binder_arity_loss", "binder_arity_loss_weight"),
    ("root_reference_arity", "root_reference_arity_loss", "root_reference_arity_loss_weight"),
    ("root_reference_identity", "root_reference_identity_loss", "root_reference_identity_loss_weight"),
    ("component_edge_alignment", "component_edge_alignment_loss", "component_edge_alignment_loss_weight"),
    ("binder_component_plan", "binder_component_plan_loss", "binder_component_plan_loss_weight"),
    ("binder_topology", "binder_topology_loss", "binder_topology_loss_weight"),
    ("fastpath_aux", "fastpath_aux_loss", "fastpath_aux_weight"),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LossTermV1:
    """One named loss term with its raw value, coefficient, and contribution."""

    name: str
    raw: float
    weight: float
    contribution: float
    normalization_denominator: float | None = None
    gradient_norm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw": self.raw,
            "weight": self.weight,
            "contribution": self.contribution,
            "normalization_denominator": self.normalization_denominator,
            "gradient_norm": self.gradient_norm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LossTermV1":
        return cls(
            name=str(data["name"]),
            raw=_safe_float(data.get("raw")),
            weight=_safe_float(data.get("weight")),
            contribution=_safe_float(data.get("contribution")),
            normalization_denominator=_safe_float(data.get("normalization_denominator"))
            if data.get("normalization_denominator") is not None
            else None,
            gradient_norm=_safe_float(data.get("gradient_norm"))
            if data.get("gradient_norm") is not None
            else None,
        )


@dataclass(frozen=True)
class LossLedgerV1:
    """Reconciled decomposition of a single training step's objective.

    The ledger is constructed from ``last_training_metrics`` and therefore
    reflects exactly the same numbers the training loop already logs.  Its
    ``__post_init__`` fails closed when the reconstructed total does not match
    the reported total within the configured tolerance.
    """

    schema_version: str
    vocab_size: int
    full_vocab_uniform_floor: float
    candidate_set_size_mean: float | None
    candidate_uniform_floor: float | None
    active_example_count: int
    active_token_count: int
    normalization_denominator: int
    terms: tuple[LossTermV1, ...]
    total_auxiliary_contribution: float
    total_reconstructed_from_components: float
    reported_total: float
    absolute_reconciliation_error: float
    trainable_parameter_count: int | None
    total_gradient_norm: float | None
    per_component_gradient_norm: dict[str, float] | None

    def __post_init__(self) -> None:
        if self.schema_version != LOSS_LEDGER_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={self.schema_version!r} does not match "
                f"{LOSS_LEDGER_SCHEMA_VERSION!r}"
            )
        aux_sum = sum(t.contribution for t in self.terms if t.name != "principal_mask_ce")
        if not math.isclose(aux_sum, self.total_auxiliary_contribution, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"auxiliary contribution sum ({aux_sum}) != "
                f"total_auxiliary_contribution ({self.total_auxiliary_contribution})"
            )
        principal = next((t.contribution for t in self.terms if t.name == "principal_mask_ce"), 0.0)
        reconstructed = principal + aux_sum
        if not math.isclose(reconstructed, self.total_reconstructed_from_components, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"reconstructed total ({reconstructed}) != "
                f"total_reconstructed_from_components ({self.total_reconstructed_from_components})"
            )
        if self.absolute_reconciliation_error < 0.0:
            raise ValueError("absolute_reconciliation_error must be non-negative")

    def term(self, name: str) -> LossTermV1 | None:
        for t in self.terms:
            if t.name == name:
                return t
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "vocab_size": self.vocab_size,
            "full_vocab_uniform_floor": self.full_vocab_uniform_floor,
            "candidate_set_size_mean": self.candidate_set_size_mean,
            "candidate_uniform_floor": self.candidate_uniform_floor,
            "active_example_count": self.active_example_count,
            "active_token_count": self.active_token_count,
            "normalization_denominator": self.normalization_denominator,
            "terms": [t.to_dict() for t in self.terms],
            "total_auxiliary_contribution": self.total_auxiliary_contribution,
            "total_reconstructed_from_components": self.total_reconstructed_from_components,
            "reported_total": self.reported_total,
            "absolute_reconciliation_error": self.absolute_reconciliation_error,
            "trainable_parameter_count": self.trainable_parameter_count,
            "total_gradient_norm": self.total_gradient_norm,
            "per_component_gradient_norm": dict(self.per_component_gradient_norm or {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LossLedgerV1":
        return cls(
            schema_version=str(data.get("schema_version", LOSS_LEDGER_SCHEMA_VERSION)),
            vocab_size=int(data["vocab_size"]),
            full_vocab_uniform_floor=_safe_float(data.get("full_vocab_uniform_floor")),
            candidate_set_size_mean=_safe_float(data.get("candidate_set_size_mean"))
            if data.get("candidate_set_size_mean") is not None
            else None,
            candidate_uniform_floor=_safe_float(data.get("candidate_uniform_floor"))
            if data.get("candidate_uniform_floor") is not None
            else None,
            active_example_count=int(data.get("active_example_count", 0)),
            active_token_count=int(data.get("active_token_count", 0)),
            normalization_denominator=int(data.get("normalization_denominator", 0)),
            terms=tuple(LossTermV1.from_dict(t) for t in data.get("terms", [])),
            total_auxiliary_contribution=_safe_float(data.get("total_auxiliary_contribution")),
            total_reconstructed_from_components=_safe_float(data.get("total_reconstructed_from_components")),
            reported_total=_safe_float(data.get("reported_total")),
            absolute_reconciliation_error=_safe_float(data.get("absolute_reconciliation_error")),
            trainable_parameter_count=int(data["trainable_parameter_count"])
            if data.get("trainable_parameter_count") is not None
            else None,
            total_gradient_norm=_safe_float(data.get("total_gradient_norm"))
            if data.get("total_gradient_norm") is not None
            else None,
            per_component_gradient_norm=dict(data.get("per_component_gradient_norm") or {})
            or None,
        )

    @classmethod
    def from_metrics(
        cls,
        metrics: dict[str, Any],
        *,
        vocab_size: int,
        active_example_count: int,
        active_token_count: int,
        candidate_set_size_mean: float | None = None,
        trainable_parameter_count: int | None = None,
        total_gradient_norm: float | None = None,
        per_component_gradient_norm: dict[str, float] | None = None,
        reconciliation_tolerance: float = 1e-5,
    ) -> "LossLedgerV1":
        """Build a ledger from a ``last_training_metrics`` dict.

        Inactive terms (raw == 0 and weight == 0) are still included so the
        ledger shape is stable across arms.
        """
        terms: list[LossTermV1] = []
        aux_sum = 0.0
        for term_name, raw_key, weight_key in _KNOWN_TERMS:
            raw = _safe_float(metrics.get(raw_key))
            weight = _safe_float(metrics.get(weight_key))
            # Principal term is always weight 1.0 unless explicitly overridden.
            if term_name == "principal_mask_ce" and weight == 0.0:
                weight = 1.0
            contribution = raw * weight
            if term_name != "principal_mask_ce":
                aux_sum += contribution
            terms.append(
                LossTermV1(
                    name=term_name,
                    raw=raw,
                    weight=weight,
                    contribution=contribution,
                )
            )

        principal_contribution = next(
            t.contribution for t in terms if t.name == "principal_mask_ce"
        )

        # Detached auxiliary losses are accumulated in the plugin and added by
        # the training loop, so they must be counted as a separate weighted term.
        detached = _safe_float(metrics.get("detached_auxiliary_loss"))
        detached_term = LossTermV1(
            name="detached_auxiliary",
            raw=detached,
            weight=1.0,
            contribution=detached,
        )
        terms.append(detached_term)
        aux_sum += detached

        total_reconstructed = principal_contribution + aux_sum
        reported = _safe_float(metrics.get("reported_total_loss")) + detached

        abs_error = abs(total_reconstructed - reported)
        if not math.isclose(total_reconstructed, reported, rel_tol=reconciliation_tolerance, abs_tol=1e-6):
            raise ValueError(
                f"LossLedgerV1 reconciliation failed: reconstructed={total_reconstructed} "
                f"reported={reported} abs_error={abs_error}"
            )

        full_vocab_floor = math.log(max(vocab_size, 1))
        candidate_floor = (
            math.log(max(candidate_set_size_mean, 1.0))
            if candidate_set_size_mean is not None
            else None
        )

        return cls(
            schema_version=LOSS_LEDGER_SCHEMA_VERSION,
            vocab_size=vocab_size,
            full_vocab_uniform_floor=round(full_vocab_floor, 6),
            candidate_set_size_mean=candidate_set_size_mean,
            candidate_uniform_floor=round(candidate_floor, 6) if candidate_floor is not None else None,
            active_example_count=active_example_count,
            active_token_count=active_token_count,
            normalization_denominator=max(active_token_count, 1),
            terms=tuple(terms),
            total_auxiliary_contribution=round(aux_sum, 8),
            total_reconstructed_from_components=round(total_reconstructed, 8),
            reported_total=round(reported, 8),
            absolute_reconciliation_error=round(abs_error, 10),
            trainable_parameter_count=trainable_parameter_count,
            total_gradient_norm=total_gradient_norm,
            per_component_gradient_norm=per_component_gradient_norm,
        )
