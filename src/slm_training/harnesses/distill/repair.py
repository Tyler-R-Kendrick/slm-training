"""Failure-cone repair examples from traces (P2/P3 E61 substrate)."""

from __future__ import annotations

from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.distill.semantic_repair import ConflictSlice


def _canvas_to_unknown(canvas: list[int], unknown: list[int] | None) -> set[int]:
    return set(unknown or [])


def extract_failure_cone(
    trace: dict[str, Any],
    *,
    window: int = 2,
) -> list[dict[str, Any]]:
    """Pair failing canvases with later verified, localized corrections.

    Verifier-emitted ``ConflictSlice`` evidence is primary. Legacy remask and
    unknown-position neighborhoods remain visible as ``HEURISTIC`` diagnostics
    but cannot authorize primary repair training.
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
        protected: set[int] = set()
        completeness = "HEURISTIC"
        provenance = "trace.remask_heuristic"
        for remask in remasks:
            reason = str(remask.get("reason") or "")
            raw_slice = remask.get("conflict_slice")
            if isinstance(raw_slice, dict):
                slice_ = ConflictSlice.from_dict(raw_slice)
                if slice_.can_authorize_repair():
                    cone.update(
                        int(p)
                        for p in (
                            *slice_.failing_node_ids,
                            *slice_.dependency_frontier,
                        )
                        if isinstance(p, int) or str(p).isdigit()
                    )
                    protected.update(
                        int(p)
                        for p in slice_.protected_node_ids
                        if isinstance(p, int) or str(p).isdigit()
                    )
                    completeness = slice_.completeness_class
                    provenance = slice_.source_provenance
                    continue
            if reason.startswith("grammar") or reason.startswith("policy"):
                cone.update(int(p) for p in (remask.get("positions") or []))
        unknown = step.get("unknown_positions") or []
        cone.update(int(p) for p in unknown)
        if not cone:
            continue
        expanded: set[int] = set(cone)
        if completeness == "HEURISTIC":
            for pos in cone:
                for d in range(-window, window + 1):
                    expanded.add(max(0, pos + d))
        expanded.difference_update(protected)
        canvas = step.get("canvas")
        if not canvas:
            continue
        examples.append(
            {
                "step": step.get("step"),
                "canvas": list(canvas),
                "target_canvas": list(final),
                "cone_positions": sorted(expanded),
                "protected_positions": sorted(protected),
                "completeness_class": completeness,
                "authorized": completeness in {"EXACT", "SOUND_OVERAPPROX"},
                "source_provenance": provenance,
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
    require_authorized: bool = True,
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
        if require_authorized:
            cones = [cone for cone in cones if cone["authorized"]]
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
                    "failure_cones": cones,
                    "trace_id": trace.get("trace_id"),
                    "policy_checkpoint_sha": (trace.get("meta") or {}).get(
                        "policy_checkpoint_sha"
                    ),
                },
            )
        )
    return out
