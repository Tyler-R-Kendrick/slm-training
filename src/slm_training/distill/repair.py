"""Failure-cone repair examples from traces (P2/P3 E61 substrate)."""

from __future__ import annotations

from typing import Any

from slm_training.dsl.schema import ExampleRecord


def _canvas_to_unknown(canvas: list[int], unknown: list[int] | None) -> set[int]:
    return set(unknown or [])


def extract_failure_cone(
    trace: dict[str, Any],
    *,
    window: int = 2,
) -> list[dict[str, Any]]:
    """Pair failing intermediate canvases with later verified corrections.

    Heuristic cone: positions remasked with ``grammar_stream`` or that remain
    unknown after a step, expanded by ``window`` neighbors.
    """
    steps = list(trace.get("steps") or [])
    final = (trace.get("final") or {}).get("canvas")
    labels = dict(trace.get("labels") or {})
    if not labels.get("accepted") or final is None:
        return []

    examples: list[dict[str, Any]] = []
    for step in steps:
        remasks = step.get("remasks") or []
        cone: set[int] = set()
        for remask in remasks:
            reason = str(remask.get("reason") or "")
            if reason.startswith("grammar") or reason.startswith("policy"):
                cone.update(int(p) for p in (remask.get("positions") or []))
        unknown = step.get("unknown_positions") or []
        cone.update(int(p) for p in unknown)
        if not cone:
            continue
        expanded: set[int] = set()
        for pos in cone:
            for d in range(-window, window + 1):
                expanded.add(max(0, pos + d))
        canvas = step.get("canvas")
        if not canvas:
            continue
        examples.append(
            {
                "step": step.get("step"),
                "canvas": list(canvas),
                "target_canvas": list(final),
                "cone_positions": sorted(expanded),
                "prompt": (trace.get("meta") or {}).get("prompt"),
                "record_id": (trace.get("meta") or {}).get("record_id"),
                "trace_id": trace.get("trace_id"),
            }
        )
    return examples


def repair_records_from_traces(
    traces: list[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
) -> list[ExampleRecord]:
    """Materialize repair ExampleRecords (final accepted text as target)."""
    _ = tokenizer
    out: list[ExampleRecord] = []
    for idx, trace in enumerate(traces):
        if not (trace.get("labels") or {}).get("accepted"):
            continue
        text = ((trace.get("final") or {}).get("text") or "").strip()
        prompt = str((trace.get("meta") or {}).get("prompt") or "")
        if not text or not prompt:
            continue
        cones = extract_failure_cone(trace)
        if not cones:
            continue
        out.append(
            ExampleRecord(
                id=f"repair_{trace.get('trace_id') or idx}",
                prompt=prompt,
                openui=text,
                split="train",
                source="self_distilled_repair",
                meta={
                    "source_family": "self_distilled_repair",
                    "failure_cone": True,
                    "n_cone_steps": len(cones),
                    "trace_id": trace.get("trace_id"),
                    "policy_checkpoint_sha": (trace.get("meta") or {}).get(
                        "policy_checkpoint_sha"
                    ),
                },
            )
        )
    return out
