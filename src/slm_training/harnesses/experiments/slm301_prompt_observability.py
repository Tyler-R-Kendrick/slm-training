"""SLM-301 (LAR1-04): prompt-observability arms for v2 semantic-meaning coverage.

The v2 contract (``evals.meaningful_program._prompt_contract``) can only score
placeholder/component coverage that is *observable*: a visible prompt inventory
section, a ``GenerationRequest.slot_contract``, or prompt component prose. The
corpus reality (0/16 test_seeds, 0/35 SLM-294 rico rows with an inventory
section) means placeholder coverage is UNKNOWN unless the request leaks the
gold-derived slot contract. This harness classifies that observability gap and
builds the matched arm pair that closes it with
``models.template_fill.ensure_prompt_inventory`` — exactly what production
applies under ``honest_slot_contract``.

All classification is model-free and pure; decode happens in the CLI.
"""

from __future__ import annotations

from typing import Any

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import (
    PromptContractV2,
    _prompt_contract,
)
from slm_training.models.template_fill import ensure_prompt_inventory, inventory_from_prompt

__all__ = [
    "OBSERVABLE_PROMPT",
    "OBSERVABLE_REQUEST_ONLY",
    "UNKNOWN",
    "EXPERIMENT_ID",
    "MIN_USEFUL_PAIRED_DELTA",
    "coverage_class",
    "build_matched_requests",
    "observability_row",
    "observability_report",
    "paired_outcome_table",
    "paired_delta",
]

EXPERIMENT_ID = "slm301-prompt-observability"

#: Placeholder coverage is observable from the prompt itself (visible
#: inventory section); the request adds nothing.
OBSERVABLE_PROMPT = "OBSERVABLE_PROMPT"
#: Coverage exists only via the request slot contract — the gap the
#: +inventory arm closes for prompt-only consumers.
OBSERVABLE_REQUEST_ONLY = "OBSERVABLE_REQUEST_ONLY"
#: No component fact from prompt or request; coverage cannot be scored.
UNKNOWN = "UNKNOWN"

#: Predeclared minimum useful paired delta (v2-strict, arm B − arm A) on
#: OBSERVABLE_REQUEST_ONLY / UNKNOWN rows.
MIN_USEFUL_PAIRED_DELTA = 0.10


def _has_coverage(contract: PromptContractV2) -> bool:
    # The observability gap this experiment measures is *placeholder* coverage:
    # the corpus reality is 0 visible inventory sections, so placeholder
    # coverage is UNKNOWN pre-inventory everywhere (component coverage from
    # prose is partial and reported separately in each row).
    return contract.placeholder_coverage_known


def coverage_class(
    record: ExampleRecord, request: GenerationRequest | None = None
) -> str:
    """Classify where v2 contract coverage comes from for one record.

    Compares the record-only contract against the request-augmented contract:
    - ``OBSERVABLE_PROMPT``: coverage already exists without the request.
    - ``OBSERVABLE_REQUEST_ONLY``: coverage appears only when the request's
      slot contract is visible.
    - ``UNKNOWN``: no coverage fact even with the request.
    """
    base = _prompt_contract(record, None)
    if _has_coverage(base):
        return OBSERVABLE_PROMPT
    full = _prompt_contract(record, request)
    if _has_coverage(full):
        return OBSERVABLE_REQUEST_ONLY
    return UNKNOWN


def build_matched_requests(
    record: ExampleRecord,
) -> tuple[GenerationRequest, GenerationRequest]:
    """Return (arm A as-is, arm B +inventory) requests for one record.

    Arm B is identical to arm A except its prompt gains the visible
    ``Placeholders: ...`` inventory suffix from ``ensure_prompt_inventory``
    over A's slot contract (the production honest-slot path). Prompts differ
    only by that suffix; an already-inventoried prompt is never rewritten.
    """
    req_a = GenerationRequest.from_record(record)
    data = req_a.to_dict()
    data["prompt"] = ensure_prompt_inventory(req_a.prompt, list(req_a.slot_contract))
    req_b = GenerationRequest.from_dict(data)
    return req_a, req_b


def observability_row(record: ExampleRecord, *, decode: str = "not_run") -> dict[str, Any]:
    """One deterministic per-row observability classification (model-free)."""
    req_a, req_b = build_matched_requests(record)
    cls = coverage_class(record, req_a)
    base = _prompt_contract(record, None)
    return {
        "id": record.id,
        "coverage_class": cls,
        "prompt_has_inventory": bool(
            inventory_from_prompt(record.prompt, heuristic=False)
        ),
        "component_coverage_from_prose": base.component_coverage_known,
        "n_slot_contract": len(req_a.slot_contract),
        "inventory_arm_changes_prompt": req_b.prompt != req_a.prompt,
        "decode": decode,
    }


def observability_report(
    records: list[ExampleRecord], *, decode: str = "not_run"
) -> dict[str, Any]:
    """Deterministic phase-0 report: per-row classes + class histogram."""
    rows = [observability_row(record, decode=decode) for record in records]
    histogram: dict[str, int] = {}
    for row in rows:
        histogram[row["coverage_class"]] = histogram.get(row["coverage_class"], 0) + 1
    return {
        "n": len(rows),
        "class_histogram": dict(sorted(histogram.items())),
        "rows": rows,
    }


def paired_outcome_table(
    pairs: list[tuple[str, bool | None, bool | None]],
) -> dict[str, int]:
    """Count A-vs-B v2-strict outcome combos keyed by record id.

    ``pairs`` is ``(record_id, arm_a_pass, arm_b_pass)`` with ``None`` for an
    unmeasured arm (decode_timeout / not_run) — unpaired, never a pass or fail.
    Keys: ``both_pass``, ``a_pass_b_fail``, ``a_fail_b_pass``, ``both_fail``,
    ``unpaired``.
    """
    counts = {
        "both_pass": 0,
        "a_pass_b_fail": 0,
        "a_fail_b_pass": 0,
        "both_fail": 0,
        "unpaired": 0,
    }
    for _record_id, a, b in pairs:
        if a is None or b is None:
            counts["unpaired"] += 1
        elif a and b:
            counts["both_pass"] += 1
        elif a and not b:
            counts["a_pass_b_fail"] += 1
        elif not a and b:
            counts["a_fail_b_pass"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def paired_delta(pairs: list[tuple[str, bool | None, bool | None]]) -> dict[str, Any]:
    """Mean paired (B − A) v2-strict delta over measured rows.

    Unmeasured rows are excluded from the denominator (never counted as a
    regression or a win). ``delta`` is ``None`` when nothing was measured.
    """
    measured = [(a, b) for _rid, a, b in pairs if a is not None and b is not None]
    n = len(measured)
    delta = (
        sum(int(b) - int(a) for a, b in measured) / n if n else None
    )
    return {
        "n_measured": n,
        "n_unpaired": len(pairs) - n,
        "delta": delta,
        "min_useful_delta": MIN_USEFUL_PAIRED_DELTA,
        "meets_min_useful_delta": (
            delta is not None and delta >= MIN_USEFUL_PAIRED_DELTA
        ),
    }
