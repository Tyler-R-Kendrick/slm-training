"""Exact objective-geometry profiler in the TwoTower adapter subspace (LDI2-02 / SLM-125).

Diagnostic only. Profiles the geometry of the protected objective quantities
(local objective loss, good/bad legal-probability mass, mean good-vs-bad margin) in a
small, explicit low-rank adapter parameter subspace on a **frozen** parent, to test
whether the E284 no-safe-direction result was intrinsic to the evidence/objectives or a
consequence of full-parameter cost and geometry.

Honesty boundary (E285/E286 lessons):

* No training, no checkpoint, no quality claim — a solver result is diagnostic evidence,
  not authorization to run a training campaign.
* One cumulative monotonic :class:`DiagnosticBudget` (hard 3-minute cap) spans support,
  forward, gradient, and solve; on expiry the run emits a stopped record with
  ``result=None`` and never a partial artifact.
* Gradients are taken **only** over the adapter tensors; the parent stays frozen
  (``parent.grad is None``). Full-parameter profiling is refused upstream by
  :func:`tier2_subspace_gradients`.
* The removed E286 batched-VJP path is **not** reintroduced: each protected quantity is
  differentiated with a plain reverse pass over a shared forward graph.

Every logits/objective computation reuses the tested legal-token math in
``local_train`` via a thin, transient ``DecisionStateV2`` -> ``DecisionEventV1`` shim; the
shim is a computational vehicle only and is never persisted.
"""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch

from slm_training.harnesses.preference.decision_diagnostics import (
    DiagnosticBudget,
    not_authorized_report,
    run_bounded_stages,
    write_diagnostic_report,
)
from slm_training.harnesses.preference.decision_events_v2 import (
    DecisionStateV2,
    ObjectiveView,
    objective_view_signature,
    objective_view_support,
)
from slm_training.harnesses.preference.local_decisions import DecisionEventV1
from slm_training.harnesses.preference.local_train import (
    _event_logits_many,
    _fresh_adamw_direction,
    _gradient_alignment,
    _guard_objective_tensors,
    _minimum_norm_gradient,
    _project_conflicting_gradients,
    _scale_gradient,
)
from slm_training.lineage.records import content_sha

__all__ = [
    "PROTECTED_OBJECTIVES",
    "AuthorizationDecision",
    "SubspaceGeometryError",
    "authorize_adapter_geometry",
    "profile_adapter_subspace_geometry",
    "profile_corpus_cell",
    "write_geometry_report",
]

# The protected, minimization-oriented quantities `_guard_objective_tensors` returns in
# grammar-legal probability space. `loss` follows the configured objective; the other
# three are objective-independent legal-space geometry.
PROTECTED_OBJECTIVES: tuple[str, ...] = (
    "loss",
    "good_probability_mass",
    "bad_probability_mass",
    "mean_margin",
)

# `_guard_objective_tensors` already returns every quantity oriented so that *descending*
# improves it (good mass is negated, margin is negated). Recorded for the report so the
# sign convention is explicit and testable.
OBJECTIVE_SENSE: dict[str, str] = {
    "loss": "minimize",
    "good_probability_mass": "minimize(-good_mass) == maximize good mass",
    "bad_probability_mass": "minimize bad mass",
    "mean_margin": "minimize(-margin) == maximize good-vs-bad margin",
}


class SubspaceGeometryError(RuntimeError):
    """Raised when a profiling request is structurally invalid (not on expiry)."""


def _state_view_event(
    state: DecisionStateV2,
    view: ObjectiveView,
    *,
    evidence_confidence: float = 1.0,
) -> DecisionEventV1:
    """Transient ``DecisionEventV1`` view of a V2 (state, objective) pair.

    Reuses the tested legal-token objective math without duplicating it. The synthetic
    identity fields are diagnostic-only and this event is never written to disk; only the
    real ``state_id`` / objective-view signature reach the persisted report.
    """
    if state.architecture != "twotower":
        raise SubspaceGeometryError(
            f"adapter-subspace geometry profiles twotower states; got {state.architecture!r}"
        )
    return DecisionEventV1(
        event_id=f"ldi2-02:{state.state_id}:{view.materializer_id}",
        group_id=state.group_id,
        context_text=state.context_text,
        canvas_ids=state.canvas_ids or (),
        position=state.decision_position,
        good_token_ids=view.good_action_ids,
        bad_token_ids=view.bad_action_ids,
        legal_token_ids=state.legal_action_ids,
        evidence_kind="counterfactual",
        evidence_confidence=float(evidence_confidence),
        decision_kind=state.decision_kind,
        split=state.split,
        policy_checkpoint_sha=state.policy_checkpoint_sha,
        tokenizer_sha=state.tokenizer_sha,
        decode_config_hash=state.decode_config_hash,
        seed=0,
        trajectory_id=f"ldi2-02-profile:{state.group_id}",
    )


def _trainable_pairs(
    corpus: Sequence[tuple[DecisionStateV2, ObjectiveView]],
) -> tuple[list[tuple[DecisionStateV2, ObjectiveView, DecisionEventV1]], list[dict[str, Any]]]:
    """Keep only trainable views with a non-empty good AND bad partition.

    A view with an empty good or bad set cannot form a preference objective; it is
    excluded with a recorded reason rather than silently dropped (an inactive objective is
    explicit, per the diagnostic contract).
    """
    kept: list[tuple[DecisionStateV2, ObjectiveView, DecisionEventV1]] = []
    excluded: list[dict[str, Any]] = []
    for state, view in corpus:
        if not view.trainable:
            excluded.append(
                {"state_id": state.state_id, "reason": "non_trainable_view"}
            )
            continue
        if not view.good_action_ids or not view.bad_action_ids:
            excluded.append(
                {"state_id": state.state_id, "reason": "empty_good_or_bad_partition"}
            )
            continue
        kept.append((state, view, _state_view_event(state, view)))
    return kept, excluded


def _objective_gradients(
    model: Any,
    events: Sequence[DecisionEventV1],
    adapter_params: Sequence[torch.nn.Parameter],
    *,
    objective: str,
    epsilon: float,
    tau: float,
    telemetry: dict[str, int],
) -> dict[str, list[torch.Tensor | None]]:
    """Reverse-pass gradient of each protected quantity w.r.t. the adapter tensors.

    All four quantities share one forward graph (``retain_graph=True``); ``allow_unused``
    keeps params an objective does not touch as ``None`` so the solvers see the true active
    subset. No batched VJP (E286).
    """
    logits_rows = _event_logits_many(model, list(events))
    telemetry["forward_passes"] += 1
    sums: dict[str, torch.Tensor] = {}
    for logits, event in zip(logits_rows, events, strict=True):
        tensors = _guard_objective_tensors(
            logits,
            event,
            objective=objective,
            probability_space="legal_tokens",
            epsilon=epsilon,
            tau=tau,
        )
        for name in PROTECTED_OBJECTIVES:
            sums[name] = tensors[name] if name not in sums else sums[name] + tensors[name]
    count = len(events)
    gradients: dict[str, list[torch.Tensor | None]] = {}
    for index, name in enumerate(PROTECTED_OBJECTIVES):
        mean_quantity = sums[name] / count
        grad = torch.autograd.grad(
            mean_quantity,
            adapter_params,
            retain_graph=index < len(PROTECTED_OBJECTIVES) - 1,
            allow_unused=True,
        )
        telemetry["backward_passes"] += 1
        gradients[name] = [None if g is None else g.detach() for g in grad]
    return gradients


def _is_active(gradient: Sequence[torch.Tensor | None]) -> bool:
    return any(g is not None and bool(torch.any(g != 0)) for g in gradient)


def _flatten(gradient: Sequence[torch.Tensor | None]) -> list[torch.Tensor]:
    return [g for g in gradient if g is not None]


def _solve(
    objective_gradients: dict[str, list[torch.Tensor | None]],
    adapter_params: Sequence[torch.nn.Parameter],
) -> dict[str, Any]:
    """Run the declared solvers/transforms over the active objective gradients.

    Reports without assuming success: weighted-mean control, PCGrad, MGDA/min-norm (with
    the common-descent certificate and mixing weights), and the first-step SGD/AdamW
    transforms aligned against the min-norm direction.
    """
    active = {name: grad for name, grad in objective_gradients.items() if _is_active(grad)}
    inactive = sorted(set(objective_gradients) - set(active))
    if len(active) < 2:
        return {
            "active_objectives": sorted(active),
            "inactive_objectives": inactive,
            "status": "insufficient_active_objectives",
            "reason": "fewer than two active objectives; multi-objective solvers skipped",
        }
    ordered = sorted(active)
    grad_list = [active[name] for name in ordered]

    # Weighted-mean (uniform) control: mean over active objectives per parameter.
    weighted_mean: list[torch.Tensor | None] = []
    for column in zip(*grad_list, strict=True):
        present = [g for g in column if g is not None]
        weighted_mean.append(None if not present else torch.stack(present).mean(dim=0))

    pcgrad_direction, pcgrad_report = _project_conflicting_gradients(grad_list)
    mgda_direction, mgda_report = _minimum_norm_gradient(grad_list)

    sgd_first_step = _scale_gradient(mgda_direction, "unit_norm")
    adamw_first_step = _fresh_adamw_direction(mgda_direction, list(adapter_params))
    adam_first_step = _fresh_adamw_direction(
        mgda_direction, list(adapter_params), weight_decay=0.0
    )

    # Common-descent alignment of each transformed step against the raw objectives.
    def _alignment_against_objectives(direction: Sequence[torch.Tensor | None]) -> dict[str, Any]:
        per_objective = {
            name: _gradient_alignment(direction, active[name]) for name in ordered
        }
        min_dot = min(entry["dot"] for entry in per_objective.values())
        return {
            "per_objective": per_objective,
            "min_dot": min_dot,
            "descends_all": min_dot > 0.0,
        }

    return {
        "active_objectives": ordered,
        "inactive_objectives": inactive,
        "status": "solved",
        "objective_pair_alignment": {
            f"{left}|{right}": _gradient_alignment(active[left], active[right])
            for i, left in enumerate(ordered)
            for right in ordered[i + 1 :]
        },
        "weighted_mean": _alignment_against_objectives(weighted_mean),
        "pcgrad": {"report": pcgrad_report, **_alignment_against_objectives(pcgrad_direction)},
        "mgda": {
            "report": mgda_report,
            "common_descent": bool(mgda_report.get("common_descent", False)),
            "weights": mgda_report.get("weights"),
            **_alignment_against_objectives(mgda_direction),
        },
        "sgd_first_step": _alignment_against_objectives(sgd_first_step),
        "adamw_first_step": _alignment_against_objectives(adamw_first_step),
        "adam_first_step": _alignment_against_objectives(adam_first_step),
    }


_STRATA: tuple[str, ...] = ("decision_kind", "abstract_state_role", "objective_signature")


def _stratum_key(state: DecisionStateV2, view: ObjectiveView, stratum: str) -> str:
    if stratum == "decision_kind":
        return state.decision_kind
    if stratum == "abstract_state_role":
        return state.abstract_state_role
    if stratum == "objective_signature":
        return objective_view_signature(view)
    raise SubspaceGeometryError(f"unknown stratum {stratum!r}")


def profile_corpus_cell(
    model: Any,
    corpus: Sequence[tuple[DecisionStateV2, ObjectiveView]],
    adapter_params: Sequence[torch.nn.Parameter],
    *,
    objective: str = "ftpo_set",
    epsilon: float = 2.0,
    tau: float = 1.0,
    min_train_support: int = 1,
    scalings: Sequence[str] = ("raw", "unit_norm"),
) -> dict[str, Any]:
    """Profile one already-attached adapter cell over the corpus and its strata.

    Reports exact objective-signature support **before** any gradient is computed, then
    the pooled and per-stratum geometry under raw and unit-normalized gradient variants.
    Asserts the frozen parent receives no gradient.
    """
    kept, excluded = _trainable_pairs(corpus)
    support = objective_view_support(
        [(state, view) for state, view, _ in kept], min_train_support=min_train_support
    )
    telemetry = {"forward_passes": 0, "backward_passes": 0}
    cell: dict[str, Any] = {
        "objective": objective,
        "sense": OBJECTIVE_SENSE,
        "support": support,
        "excluded_views": excluded,
        "profiled_states": len(kept),
        "adapter_parameter_dimensions": int(
            sum(int(p.numel()) for p in adapter_params)
        ),
    }
    if not kept:
        cell["status"] = "no_trainable_states"
        cell["telemetry"] = telemetry
        return cell

    train_events = [event for state, _, event in kept if state.split == "train"]
    profile_events = train_events or [event for _, _, event in kept]

    raw_gradients = _objective_gradients(
        model,
        profile_events,
        adapter_params,
        objective=objective,
        epsilon=epsilon,
        tau=tau,
        telemetry=telemetry,
    )
    # Parent must never receive a gradient (frozen-subspace invariant).
    _assert_parent_grad_free(model)

    variants: dict[str, Any] = {}
    for scaling in scalings:
        scaled = {
            name: _scale_gradient(grad, scaling) for name, grad in raw_gradients.items()
        }
        variants[scaling] = _solve(scaled, adapter_params)
    cell["pooled"] = {"gradient_variants": variants}

    strata_report: dict[str, Any] = {}
    for stratum in _STRATA:
        buckets: dict[str, list[DecisionEventV1]] = {}
        for state, view, event in kept:
            buckets.setdefault(_stratum_key(state, view, stratum), []).append(event)
        stratum_cells: dict[str, Any] = {}
        for key, events in sorted(buckets.items()):
            grads = _objective_gradients(
                model,
                events,
                adapter_params,
                objective=objective,
                epsilon=epsilon,
                tau=tau,
                telemetry=telemetry,
            )
            stratum_cells[key] = {
                "states": len(events),
                "unit_norm": _solve(
                    {n: _scale_gradient(g, "unit_norm") for n, g in grads.items()},
                    adapter_params,
                ),
            }
        strata_report[stratum] = stratum_cells
    cell["strata"] = strata_report
    cell["status"] = "profiled"
    cell["telemetry"] = telemetry
    return cell


def _assert_parent_grad_free(model: Any) -> None:
    for name, parameter in model.named_parameters():
        if "lora_" in name.lower():
            continue
        if parameter.grad is not None:
            raise SubspaceGeometryError(
                f"frozen parent parameter {name!r} received a gradient; "
                "adapter-subspace invariant violated"
            )


def profile_adapter_subspace_geometry(
    model_factory: Callable[[], Any],
    corpus: Sequence[tuple[DecisionStateV2, ObjectiveView]],
    spec_factory: Callable[[Any, dict[str, Any]], Any],
    matrix: Sequence[dict[str, Any]],
    *,
    objective: str = "ftpo_set",
    epsilon: float = 2.0,
    tau: float = 1.0,
    min_train_support: int = 1,
    budget: DiagnosticBudget | None = None,
) -> dict[str, Any]:
    """Profile the objective geometry across a rank/target-module adapter matrix.

    ``matrix`` is a sequence of cell descriptors (e.g. ``{"rank": 4, "target_modules": (...)}``);
    ``spec_factory(model, cell)`` builds the :class:`TwoTowerAdapterSpec` for a fresh model.
    Each cell is one bounded stage under a single cumulative deadline, so an expiry mid-matrix
    yields a stopped record with ``result=None`` (no partial artifact, per E285/E286).
    """
    if not matrix:
        raise SubspaceGeometryError("adapter geometry matrix must not be empty")
    if not corpus:
        return not_authorized_report(
            "empty corpus; nothing to profile", budget=budget
        )

    def _make_stage(cell: dict[str, Any]) -> tuple[str, Callable[[], Any]]:
        label = _cell_label(cell)

        def _run() -> dict[str, Any]:
            model = model_factory()
            spec = spec_factory(model, cell)
            model.attach_adapter(spec)
            adapter_params = list(model.adapter_parameters())
            if not adapter_params:
                raise SubspaceGeometryError(f"cell {label!r} resolved no adapter tensors")
            tracing = not tracemalloc.is_tracing()
            if tracing:
                tracemalloc.start()
            try:
                result = profile_corpus_cell(
                    model,
                    corpus,
                    adapter_params,
                    objective=objective,
                    epsilon=epsilon,
                    tau=tau,
                    min_train_support=min_train_support,
                )
                _, peak = tracemalloc.get_traced_memory()
            finally:
                if tracing:
                    tracemalloc.stop()
                del model
                gc.collect()
            result["cell"] = dict(cell)
            result["adapter_identity"] = spec.to_dict() if hasattr(spec, "to_dict") else str(spec)
            result["peak_memory_bytes"] = int(peak)
            return result

        return label, _run

    stages = [_make_stage(cell) for cell in matrix]
    report = run_bounded_stages(stages, budget=budget)
    report["kind"] = "adapter_subspace_geometry"
    report["objective"] = objective
    report["matrix_cells"] = [_cell_label(cell) for cell in matrix]
    report["corpus_states"] = len(corpus)
    decision, reason = authorize_adapter_geometry(report)
    report["authorization"] = {"decision": decision, "reason": reason}
    if report.get("result") is not None:
        report["result_content_sha"] = content_sha(report["result"])
    return report


AuthorizationDecision = Literal[
    "authorized", "repair_evidence", "no_safe_direction", "expired"
]


def authorize_adapter_geometry(
    report: Mapping[str, Any],
) -> tuple[AuthorizationDecision, str]:
    """Map a completed adapter-subspace geometry report to a training authorization.

    Fail-closed: only an explicit bounded common-descent certificate authorizes a
    training arm. Support gaps are routed to ``repair_evidence`` (the corpus is
    incomplete), not to a training decision. This keeps the diagnostic honest:
    a solver result is evidence, not authorization.
    """
    status = report.get("status")
    if status == "expired":
        return "expired", "diagnostic expired before all cells completed"
    if status == "not_authorized":
        return "no_safe_direction", str(
            report.get("reason", "diagnostic refused the request")
        )
    if status != "completed":
        return (
            "no_safe_direction",
            f"unexpected diagnostic status {status!r}; failing closed",
        )

    result = report.get("result")
    if not isinstance(result, Mapping) or not result:
        return "no_safe_direction", "completed diagnostic carried no result to authorize"

    def _rank_sort(label_cell: tuple[str, Any]) -> int:
        label = label_cell[0]
        prefix = label.split(":", 1)[0]
        try:
            return int(prefix.replace("rank", ""))
        except ValueError:
            return 999999

    for label, cell in sorted(result.items(), key=_rank_sort):
        if not isinstance(cell, Mapping):
            return "no_safe_direction", f"cell {label!r} is not a mapping; invalid report"
        if cell.get("status") != "profiled":
            return "repair_evidence", f"cell {label!r} was not profiled"
        support = cell.get("support") or {}
        coverage = support.get("held_out_coverage") or {}
        if not coverage.get("passed", True):
            uncovered = coverage.get("uncovered", [])
            return (
                "repair_evidence",
                f"cell {label!r} has {len(uncovered)} uncovered held-out objective signature(s)",
            )

    for label, cell in sorted(result.items(), key=_rank_sort):
        variants = (cell.get("pooled") or {}).get("gradient_variants", {})
        unit = variants.get("unit_norm", {})
        if unit.get("status") != "solved":
            continue
        mgda = unit.get("mgda", {})
        if bool(mgda.get("common_descent")) and bool(mgda.get("descends_all")):
            return (
                "authorized",
                f"bounded common descent certified in cell {label!r}; "
                "adapter training arm may be scheduled",
            )

    return (
        "no_safe_direction",
        "no rank/module cell certified a common descent direction on the protected objectives",
    )


def _cell_label(cell: dict[str, Any]) -> str:
    rank = cell.get("rank", "?")
    modules = cell.get("target_modules")
    module_tag = "all" if modules is None else "+".join(sorted(modules))
    return f"rank{rank}:{module_tag}"


def write_geometry_report(path: Any, report: dict[str, Any]) -> None:
    """Persist a geometry report atomically (delegates to the diagnostics writer)."""
    write_diagnostic_report(path, report)
