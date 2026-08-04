"""Causal throughput targets derived from declared work and hardware inputs."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable


def parse_parameter_count(value: Any) -> int | None:
    """Parse dashboard/model-card counts such as ``≈135M``."""

    match = re.search(r"([0-9][0-9,.]*)\s*([KMB])?", str(value or ""), re.I)
    if match is None:
        return None
    amount = float(match.group(1).replace(",", ""))
    scale = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(
        (match.group(2) or "").upper(), 1
    )
    result = int(amount * scale)
    return result if result > 0 else None


def _assumption_values(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    return {str(row["key"]): float(row["value"]) for row in rows}


def _profile(
    *,
    name: str,
    streams: int,
    parameters: int,
    output_tokens: int,
    forward_range: tuple[int, int],
    oracle_steps: int,
    oracle_operations_per_step: int,
    assumptions: dict[str, float],
) -> dict[str, Any]:
    compute_rate = (
        assumptions["peak_compute_gops"] * 1e9 * assumptions["compute_utilization"]
    )
    bandwidth_rate = (
        assumptions["memory_bandwidth_gbs"] * 1e9 * assumptions["bandwidth_utilization"]
    )
    serial_seconds = (
        oracle_steps
        * (
            assumptions["dispatch_step_us"]
            + oracle_operations_per_step * assumptions["oracle_operation_us"]
        )
        / 1e6
    )

    def estimate(forwards: int) -> dict[str, float]:
        operations = parameters * assumptions["ops_per_parameter_forward"] * forwards
        bytes_moved = parameters * assumptions["bytes_per_parameter_forward"] * forwards
        compute_seconds = operations / compute_rate
        memory_seconds = bytes_moved / bandwidth_rate
        per_stream_seconds = max(compute_seconds, memory_seconds) + serial_seconds
        tokens_per_second = output_tokens / per_stream_seconds
        return {
            "forwards": forwards,
            "operations": operations,
            "bytes": bytes_moved,
            "compute_seconds": compute_seconds,
            "memory_seconds": memory_seconds,
            "serial_seconds": serial_seconds,
            "decode_seconds": per_stream_seconds,
            "tokens_per_second": tokens_per_second * streams,
            "bottleneck": (
                "memory" if memory_seconds >= compute_seconds else "compute"
            ),
        }

    best = estimate(forward_range[0])
    floor = estimate(forward_range[1])
    return {
        "id": name,
        "label": "Single stream" if streams == 1 else "Aggregate",
        "streams": streams,
        "floor_tps": floor["tokens_per_second"],
        "ceiling_tps": best["tokens_per_second"],
        "floor_decode_seconds": floor["decode_seconds"],
        "ceiling_decode_seconds": best["decode_seconds"],
        "floor_bottleneck": floor["bottleneck"],
        "floor_case": floor,
        "ceiling_case": best,
    }


def build_performance_target(
    *,
    checkpoint_references: Iterable[dict[str, Any]],
    assumptions: list[dict[str, Any]],
    output_tokens: int,
    structural_turn_range: tuple[int, int],
    grammar_inputs: dict[str, int],
) -> dict[str, Any]:
    """Build a target envelope without reading observed throughput."""

    references = list(checkpoint_references)
    selected: dict[str, Any] | None = None
    parameters: int | None = None
    for reference in references:
        candidate = parse_parameter_count(reference.get("parameters"))
        if candidate is not None:
            selected = reference
            parameters = candidate
            if reference.get("role") == "Latest checkpoint":
                break
    if parameters is None:
        return {
            "status": "unavailable",
            "reason": "No checkpoint/model parameter count is declared.",
            "profiles": [],
            "levers": [],
            "mismatches": [],
        }

    values = _assumption_values(assumptions)
    minimum_forwards = max(1, int(structural_turn_range[0]))
    maximum_forwards = max(minimum_forwards, int(structural_turn_range[1]))
    oracle_steps = max(1, int(output_tokens))
    symbol_mask_words = max(
        1,
        math.ceil(
            (
                int(grammar_inputs.get("components", 0))
                + int(grammar_inputs.get("structural_tokens", 0))
            )
            / 64
        ),
    )
    oracle_operations_per_step = (
        1
        + symbol_mask_words
        + max(1, int(grammar_inputs.get("maximum_alternatives", 1)))
    )
    grammar_inputs = {
        **grammar_inputs,
        "symbol_mask_words": symbol_mask_words,
        "oracle_operations_per_step": oracle_operations_per_step,
    }
    aggregate_streams = max(1, int(values["independent_streams"]))
    profiles = [
        _profile(
            name="single_stream",
            streams=1,
            parameters=parameters,
            output_tokens=output_tokens,
            forward_range=(minimum_forwards, maximum_forwards),
            oracle_steps=oracle_steps,
            oracle_operations_per_step=oracle_operations_per_step,
            assumptions=values,
        ),
        _profile(
            name="aggregate",
            streams=aggregate_streams,
            parameters=parameters,
            output_tokens=output_tokens,
            forward_range=(minimum_forwards, maximum_forwards),
            oracle_steps=oracle_steps,
            oracle_operations_per_step=oracle_operations_per_step,
            assumptions=values,
        ),
    ]
    single = profiles[0]
    base_floor = float(single["floor_tps"])

    def lever(
        key: str,
        direction: str,
        requirement: str,
        effect: str,
        source: str,
    ) -> dict[str, Any]:
        return {
            "lever": key,
            "current": values.get(
                key, parameters if key == "model_parameters" else None
            ),
            "direction": direction,
            "requirement": requirement,
            "effect": effect,
            "source": source,
        }

    levers = [
        lever(
            "model_parameters",
            "lower",
            "Promote the smallest sufficient model; growth must satisfy EG_params.",
            "Model compute and weight traffic scale linearly with P.",
            f"{selected.get('role', 'checkpoint')} {selected.get('run_id', '')}".strip(),
        ),
        lever(
            "structural_forwards",
            "lower",
            f"Move {maximum_forwards} toward the certified {minimum_forwards}-forward bound.",
            "I1/I2 singleton bypass and I4 batching remove complete forwards.",
            "grammar topology + ModelBuildConfig",
        ),
        lever(
            "bytes_per_parameter_forward",
            "lower",
            "Quantize/cache weights without weakening exact legality.",
            "Raises the memory-bound ceiling.",
            "declared target profile",
        ),
        lever(
            "memory_bandwidth_gbs",
            "higher",
            "Use a declared host/device with higher sustainable bandwidth.",
            "Raises the memory-bound ceiling.",
            "declared target profile",
        ),
        lever(
            "dispatch_step_us",
            "lower",
            "Fuse dispatch without weakening request-local legality.",
            f"Applies across {oracle_steps} output-token legality steps.",
            "declared target profile",
        ),
        lever(
            "oracle_operation_us",
            "lower",
            (
                f"Reduce {oracle_operations_per_step} modeled operations/step via "
                "bitsets, checked lookup, and semantic-state reuse."
            ),
            "Improves only the request-local serial term.",
            "DSL alternatives + component/structural-symbol mask width",
        ),
        lever(
            "independent_streams",
            "higher for aggregate only",
            "Batch independent requests while preserving single-stream latency reporting.",
            "Changes aggregate TPS, never the single-stream target.",
            "declared target profile",
        ),
    ]
    mismatches = [
        {
            "code_or_input": "Old dashboard P",
            "current": "grammar-description memory estimate",
            "required": f"whole-model checkpoint parameters ({parameters:,})",
            "impact": "Old TPS floor divided by hundreds of parameters instead of model-scale P.",
        },
        {
            "code_or_input": "Target authority",
            "current": "browser-side formula",
            "required": "canonical backend payload",
            "impact": "Formula, assumptions, and evidence joins are now testable together.",
        },
        {
            "code_or_input": "Concurrency",
            "current": "single and aggregate mixed",
            "required": "separate profiles",
            "impact": "A core-count multiplier can no longer masquerade as latency TPS.",
        },
        {
            "code_or_input": "Observed metrics",
            "current": "comparison only",
            "required": "never an input to the target",
            "impact": "The target remains normative; actuals diagnose implementation gaps.",
        },
    ]
    return {
        "status": "estimated",
        "schema": "dsl_performance_target/v1",
        "parameter_count": parameters,
        "parameter_source": {
            "role": selected.get("role"),
            "run_id": selected.get("run_id"),
            "declared": selected.get("parameters"),
        },
        "output_tokens": output_tokens,
        "forward_range": [minimum_forwards, maximum_forwards],
        "grammar_inputs": grammar_inputs,
        "profiles": profiles,
        "levers": levers,
        "mismatches": mismatches,
        "formula": (
            "TPS = streams × output_tokens / "
            "(max(P·ops·forwards/(ηc·Cpeak), "
            "P·bytes·forwards/(ηb·Bpeak)) + "
            "oracle_steps·(dispatch + oracle_ops_per_step·oracle_op_cost))"
        ),
        "basis": (
            "Normative planning estimate from declared model geometry, grammar-derived "
            "forward/oracle work, and a declared hardware profile. Observed runs are "
            "excluded and compared separately."
        ),
        "single_stream_floor_tps": base_floor,
        "finite": math.isfinite(base_floor) and base_floor > 0,
    }


__all__ = ["build_performance_target", "parse_parameter_count"]
