#!/usr/bin/env python3
"""Run SLM-293's bounded local frozen-base factor-head saturation control.

This is deliberately a capacity control, not a ship evaluation.  It trains the
already-live auxiliary heads on one declared train record, freezes every base
generator parameter, and records the exact head metrics.  Decoder causal-use
claims remain fail-closed until a bounded decode produces the corresponding
``DecodeStats`` eligible-position counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.harnesses.model_build.data import ExampleRecord
from slm_training.models.semantic_connector import SemanticConnector
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs/design/iter-slm293-property-role-saturation-20260725.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/iter-slm293-property-role-saturation-20260725.md"
SEED = 293
STEPS = 24
FACTOR_METRICS = {
    "inventory": "component_inventory_topk_recall",
    "topology": "component_edge_alignment_accuracy",
    "cardinality": "binder_arity_accuracy",
    "property_ownership": "slot_component_accuracy",
    "binder": "binder_component_plan_accuracy",
    "reference_edge": "root_reference_arity_accuracy",
}
DECODE_COUNTERS = {
    "inventory": "component_inventory_choice_changes",
    "topology": "component_edge_choice_changes",
    "cardinality": "binder_arity_choice_changes",
    "property_ownership": "slot_component_choice_changes",
    "binder": "binder_component_plan_choice_changes",
    "reference_edge": "root_reference_arity_choice_changes",
}


def _factor_controls(scores: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    """Evaluate all factor-value controls at an existing eligible head choice.

    This is deliberately separate from decoder counters: it establishes that the
    learned factor values, gold replacement, permutation, and zeroing are all
    executable without changing model parameters.  The decoder counters record
    the later, stronger full-generation causal-use control.
    """
    if (
        scores.ndim != 2
        or targets.shape != (scores.size(0),)
        or not bool(((targets >= 0) & (targets < scores.size(1))).all())
    ):
        raise ValueError("factor control requires aligned eligible score rows + targets")
    gold = torch.full_like(scores, -20.0)
    gold[torch.arange(scores.size(0), device=scores.device), targets] = 20.0
    controls = {
        "learned": scores,
        "gold_substituted": gold,
        "shuffled": scores.roll(1, dims=0),
        "zeroed": torch.zeros_like(scores),
    }
    baseline = controls["zeroed"].argmax(dim=1)
    return {
        name: {
            "strict_factor_accuracy": float(
                values.argmax(dim=1).eq(targets).float().mean().item()
            ),
            "eligible_choices": int(scores.size(0)),
            "choice_changes_vs_zero": int(
                values.argmax(dim=1).ne(baseline).sum().item()
            ),
        }
        for name, values in controls.items()
    }


def _records() -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id="slm293_card_title",
            prompt="card with title",
            openui='root = Card([title])\ntitle = TextContent(":slot_0")',
            placeholders=[":slot_0"],
            split="train",
            source="slm293_local_capacity_control",
        ),
        ExampleRecord(
            id="slm293_stack_submit",
            prompt="stack with title and submit",
            openui=(
                'root = Stack([title, submit])\n'
                'title = TextContent(":slot_0")\n'
                'submit = Button(":slot_1")'
            ),
            placeholders=[":slot_0", ":slot_1"],
            split="train",
            source="slm293_local_capacity_control",
        ),
    ]


def _model(records: list[ExampleRecord]) -> TwoTowerModel:
    return TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            seed=SEED,
            d_model=16,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=16,
            max_target_len=64,
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            semantic_connector="low_rank",
            component_inventory_loss_weight=1.0,
            component_plan_loss_weight=1.0,
            slot_component_loss_weight=1.0,
            component_edge_loss_weight=1.0,
            component_edge_alignment_loss_weight=1.0,
            binder_component_plan_loss_weight=1.0,
            binder_topology_loss_weight=1.0,
            binder_arity_loss_weight=1.0,
            root_reference_arity_loss_weight=1.0,
        ),
        device="cpu",
    )


def _frozen_parameter_digest(model: TwoTowerModel) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.endswith("_head.weight") or name.endswith("_head.bias"):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _head_factor_controls(
    model: TwoTowerModel, records: list[ExampleRecord]
) -> dict[str, Any]:
    """Read the existing heads at one parser-derived eligible factor choice."""
    ctx, ctx_pad = model._encode_context([record.prompt for record in records])
    pooled = model._pool_context(ctx, ctx_pad)
    tokenizer = model.tokenizer
    components = model._component_inventory_token_ids()
    component_index = {token_id: index for index, token_id in enumerate(components)}
    binders = model._binder_component_token_ids()
    binder_index = {token_id: index for index, token_id in enumerate(binders)}
    card = tokenizer.token_to_id["Card"]
    stack = tokenizer.token_to_id["Stack"]
    text = tokenizer.token_to_id["TextContent"]
    button = tokenizer.token_to_id["Button"]
    root = binder_index[tokenizer.bind_id(0)]
    title = binder_index[tokenizer.bind_id(1)]
    assert model.component_inventory_head is not None
    assert model.component_edge_head is not None
    assert model.binder_arity_head is not None
    assert model.slot_component_head is not None
    assert model.binder_component_plan_head is not None
    assert model.root_reference_arity_head is not None
    inventory = model.component_inventory_head(pooled)[:, [card, stack]]
    edge_logits = model.component_edge_head(pooled).view(
        len(records), len(components), len(components)
    )
    edge = torch.stack(
        [
            edge_logits[0, component_index[card], [component_index[text], component_index[button]]],
            edge_logits[1, component_index[stack], [component_index[text], component_index[button]]],
        ]
    )
    arity = model.binder_arity_head(pooled).view(
        len(records), len(binders), len(binders) + 1
    )[:, root, [1, 2]]
    slot = model._slot_component_logits(
        [records[0].placeholders[0], records[1].placeholders[1]],
        ctx,
        ctx_pad,
        torch.tensor([0, 1], device=ctx.device),
    )[:, [component_index[text], component_index[button]]]
    binder_logits = model.binder_component_plan_head(pooled).view(
        len(records), len(binders), len(components)
    )
    binder = torch.stack(
        [
            binder_logits[0, title, [component_index[text], component_index[button]]],
            binder_logits[1, binder_index[tokenizer.bind_id(2)], [component_index[text], component_index[button]]],
        ]
    )
    reference = model.root_reference_arity_head(pooled)[:, [1, 2]]
    first_second = torch.tensor([0, 1], device=ctx.device)
    return {
        "inventory": _factor_controls(inventory, first_second),
        "topology": _factor_controls(edge, first_second),
        "cardinality": _factor_controls(arity, first_second),
        "property_ownership": _factor_controls(slot, first_second),
        "binder": _factor_controls(binder, first_second),
        "reference_edge": _factor_controls(reference, first_second),
    }


def _semantic_connector_audit(model: TwoTowerModel) -> dict[str, Any]:
    """Fail closed when a configured connector has no production decode owner."""
    installed = [
        name
        for name, module in model.named_modules()
        if isinstance(module, SemanticConnector)
    ]
    if installed:
        raise AssertionError("SLM-293 expects the legacy connector to remain unowned")
    return {
        "connector_config": str(model.config.semantic_connector),
        "connector_modules_installed": installed,
        "eligible_decode_positions": 0,
        "choice_changes": 0,
        "structural_meaning_v2_delta": 0.0,
        "disposition": "non_causal_not_on_decoder_path",
        "reason": (
            "SemanticConnector is a standalone fixture module; TwoTower owns no "
            "instance, so learned, gold, shuffled, and zeroed connector outputs "
            "cannot reach an eligible decode choice."
        ),
    }


def run(*, steps: int = STEPS) -> dict[str, Any]:
    if not 1 <= steps <= 24:
        raise ValueError("steps must be between 1 and 24 for the bounded control")
    torch.manual_seed(SEED)
    records = _records()
    model = _model(records)
    frozen = []
    trainable = []
    for name, parameter in model.named_parameters():
        is_head = name.endswith("_head.weight") or name.endswith("_head.bias")
        parameter.requires_grad_(is_head)
        (trainable if is_head else frozen).append(name)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    frozen_before = _frozen_parameter_digest(model)
    metric_peaks = {metric: 0.0 for metric in FACTOR_METRICS.values()}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(records)
        detached = model.take_detached_auxiliary_loss()
        (loss + (detached if detached is not None else 0.0)).backward()
        optimizer.step()
        for metric in metric_peaks:
            metric_peaks[metric] = max(
                metric_peaks[metric], float(model.last_training_metrics.get(metric, 0.0))
            )
    frozen_after = _frozen_parameter_digest(model)
    if frozen_after != frozen_before:
        raise AssertionError("frozen base generator changed during saturation control")
    controls = _head_factor_controls(model, records)
    connector_audit = _semantic_connector_audit(model)
    factors = {
        factor: {
            "head_accuracy": metric_peaks[metric],
            "head_metric": metric,
            "decoder_counter": DECODE_COUNTERS[factor],
            "head_factor_controls": controls[factor],
            "connector_interventions": {
                control: "not_applicable_no_decoder_owner"
                for control in ("learned", "gold_substituted", "shuffled", "zeroed")
            },
            "connector_eligible_decode_positions": connector_audit[
                "eligible_decode_positions"
            ],
            "connector_choice_changes": connector_audit["choice_changes"],
            "connector_disposition": connector_audit["disposition"],
            "twotower_decoder_interventions": "pending_bounded_decode",
            "strict_semantic_metrics": "not_run",
            "disposition": (
                "capacity_saturated_decoder_control_pending"
                if metric_peaks[metric] >= 0.90
                else "head_not_saturated"
            ),
        }
        for factor, metric in FACTOR_METRICS.items()
    }
    return {
        "schema": "SLM293PropertyRoleSaturationV1",
        "run_id": "slm293-property-role-saturation-r1",
        "status": "capacity_control_complete_connector_dispositioned",
        "claim_class": "local_head_capacity_only",
        "recipe": {
            "device": "cpu",
            "seed": SEED,
            "steps": steps,
            "peak_head_accuracy": metric_peaks,
            "records": len(records),
            "split": "train",
            "base_generator_trainable_parameters": 0,
            "base_generator_digest_before": frozen_before,
            "base_generator_digest_after": frozen_after,
            "auxiliary_head_parameters_trainable": len(trainable),
            "no_checkpoint_written": True,
            "no_remote_or_hf_run": True,
        },
        "factors": factors,
        "semantic_connector_audit": connector_audit,
        "limitations": [
            "Two-record capacity saturation is not a held-out or ship result.",
            "No bounded full decode completed in this capacity control, so no decoder counter, meaning-v2, or strict-program metric is claimed.",
            "The legacy semantic_connector remains fixture-only and is not a decoder intervention path.",
        ],
        "version_stamp": build_version_stamp(
            "model.twotower", "harness.model_build.eval"
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {factor} | {data['head_accuracy']:.2f} | {data['decoder_counter']} | "
        f"{data['disposition']} | {data['connector_disposition']} |"
        for factor, data in report["factors"].items()
    )
    return f"""# SLM-293 — local property-role saturation control

Status: capacity control complete; legacy connector dispositioned non-causal.

This bounded CPU run froze the base generator and trained only existing auxiliary
heads on {report['recipe']['records']} declared train records for {report['recipe']['steps']} steps. It is
not a held-out evaluation, a checkpoint, or a ship claim.

| Factor | Head accuracy | Required live decode counter | Head disposition | Connector disposition |
| --- | ---: | --- | --- | --- |
{rows}

The paired control executes learned, gold-substituted, shuffled across examples,
and zeroed values at parser-derived eligible factor choices. The next held-out
decode must collect all four controls' decoder eligible-position argmax-change
counters, then report meaningful-v2 and strict-program outcomes. Until then
every factor remains non-promotable.

The explicitly configured legacy `semantic_connector=low_rank` installs no
`SemanticConnector` module in TwoTower. Its eligible decode positions and
choice changes are therefore exactly zero, and its structural meaning-v2 delta
is zero: **non-causal, not on the decoder path**. This is a connector
disposition, not a claim that the separate TwoTower auxiliary heads are ready
to promote.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = run(steps=args.steps)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report))


if __name__ == "__main__":
    main()
