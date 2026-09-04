"""Decode timings carried forward from the predecessor cycle.

One responsibility: reading a previous cycle's decode p95 and compiler-timeout
signal out of its artefacts, and naming the cause when one is found.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autotrain_budget import nearest_rank_p95
from scripts.autotrain_io import (
    read_json,
)
from slm_training.autoresearch.thrash_regime import (
    is_compiler_ms_timeout_signal,
)


def predecessor_decode_p95_seconds(
    root: Path | None, predecessor_campaign_id: str | None = None
) -> dict[str, Any]:
    """Measured per-record decode cost of the predecessor screening cycle.

    Reads the predecessor's ``runs/*/eval_smoke.json`` scoreboards and returns
    the pooled per-record p95 in seconds (``details[].latency_ms`` includes
    timed-out records because ``eval_runner`` measures the true wall around
    ``generate``), falling back to the scoreboard headline
    ``latency_ms_p95_including_incomplete`` / ``latency_ms_p95`` /
    ``compiler_ms_mean``. Also reports the decoded-record count, the timeout
    count, the resulting incomplete rate and the per-record timeout that was
    applied (``evaluation_policy.decode_timeout_seconds`` -> matrix knobs ->
    ``thrash_timing.json``). ``source`` is ``measured_p95`` when a cost was
    read, else ``policy_default``. Never raises; without a predecessor the
    caller falls back to the policy floor.
    """

    out: dict[str, Any] = {
        "p95_seconds": None,
        "source": "policy_default",
        "field": None,
        "campaign_id": None,
        "scoreboards": 0,
        "decoded_records": 0,
        "timeout_records": 0,
        "incomplete_rate": None,
        "timeout_seconds": None,
    }
    if root is None or not root.is_dir():
        return out
    camp_dir: Path | None = None
    if predecessor_campaign_id:
        camp_dir = root / str(predecessor_campaign_id)
    else:
        newest_mtime = -1.0
        try:
            boards = list(root.glob("*/runs/*/eval_smoke.json"))
        except OSError:
            boards = []
        for path in boards:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                camp_dir = path.parents[2]
    if camp_dir is None or not camp_dir.is_dir():
        return out
    out["campaign_id"] = camp_dir.name
    latencies_ms: list[float] = []
    headline: dict[str, list[float]] = {
        "latency_ms_p95_including_incomplete": [],
        "latency_ms_p95": [],
        "compiler_ms_mean": [],
    }
    decoded = 0
    timeouts = 0
    timeout_used: float | None = None
    boards_n = 0
    for path in sorted(camp_dir.glob("runs/*/eval_smoke.json")):
        payload = read_json(path)
        if not payload:
            continue
        boards_n += 1
        details = [d for d in (payload.get("details") or []) if isinstance(d, dict)]
        latencies_ms.extend(
            float(d["latency_ms"])
            for d in details
            if isinstance(d.get("latency_ms"), (int, float))
        )
        n_docs = len(details)
        if n_docs <= 0:
            try:
                n_docs = int(payload.get("completed_latency_n") or 0) + int(
                    payload.get("incomplete_latency_n") or 0
                )
            except (TypeError, ValueError):
                n_docs = 0
        if n_docs <= 0:
            try:
                n_docs = int(payload.get("document_n") or 0)
            except (TypeError, ValueError):
                n_docs = 0
        decoded += max(0, n_docs)
        raw_timeouts = payload.get("decode_timeout_document_count")
        if raw_timeouts is None:
            raw_timeouts = payload.get("decode_timeout_count")
        try:
            timeouts += max(0, int(raw_timeouts or 0))
        except (TypeError, ValueError):
            pass
        metrics = (
            payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        )
        for key, bucket in headline.items():
            value = payload.get(key)
            if value is None:
                value = metrics.get(key)
            if isinstance(value, (int, float)) and float(value) > 0:
                bucket.append(float(value))
        eval_policy = payload.get("evaluation_policy")
        if isinstance(eval_policy, dict):
            applied = eval_policy.get("decode_timeout_seconds")
            if isinstance(applied, (int, float)) and float(applied) > 0:
                timeout_used = max(float(applied), timeout_used or 0.0)
    if timeout_used is None:
        matrix = read_json(camp_dir / "matrix-proposal.json")
        for hyp in matrix.get("hypotheses") or []:
            exp = hyp.get("experiment") if isinstance(hyp, dict) else None
            knobs = exp.get("knobs") if isinstance(exp, dict) else None
            applied = (
                knobs.get("decode_timeout_seconds") if isinstance(knobs, dict) else None
            )
            if isinstance(applied, (int, float)) and float(applied) > 0:
                timeout_used = max(float(applied), timeout_used or 0.0)
    if timeout_used is None:
        timing = read_json(camp_dir / "thrash_timing.json")
        fit = (
            timing.get("decode_fit")
            if isinstance(timing.get("decode_fit"), dict)
            else {}
        )
        applied = fit.get("fitted_decode_timeout_seconds")
        if isinstance(applied, (int, float)) and float(applied) > 0:
            timeout_used = float(applied)
    p95_ms: float | None = None
    field: str | None = None
    if latencies_ms:
        p95_ms = nearest_rank_p95(latencies_ms)
        field = "details.latency_ms"
    else:
        for key, bucket in headline.items():
            if bucket:
                p95_ms = max(bucket)
                field = key
                break
    if p95_ms is not None and p95_ms > 0:
        out["p95_seconds"] = float(p95_ms) / 1000.0
        out["source"] = "measured_p95"
        out["field"] = field
    out["scoreboards"] = int(boards_n)
    out["decoded_records"] = int(decoded)
    out["timeout_records"] = int(timeouts)
    out["incomplete_rate"] = (
        float(min(1.0, timeouts / decoded)) if decoded > 0 else None
    )
    out["timeout_seconds"] = timeout_used
    return out


def predecessor_timeout_cause(
    root: Path | None,
    predecessor_campaign_id: str | None,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> str:
    """``budget_timeout`` / ``slow_decode_timeout`` / ``none`` for the predecessor."""

    from slm_training.autoresearch.thrash_regime import classify_timeout_cause

    ev = (
        dict(evidence)
        if evidence is not None
        else predecessor_decode_p95_seconds(root, predecessor_campaign_id)
    )
    return classify_timeout_cause(
        p95_seconds=ev.get("p95_seconds"),
        timeout_seconds=ev.get("timeout_seconds"),
        timeout_count=ev.get("timeout_records"),
    )


def predecessor_compiler_ms_timeout(
    root: Path, predecessor_campaign_id: str | None
) -> bool:
    """True when the predecessor cycle timed out under compiler_ms dominance."""

    if not predecessor_campaign_id:
        return False
    camp_dir = root / predecessor_campaign_id
    reasons: list[str] = []
    delivery_path = camp_dir / "sdlc_delivery.json"
    if delivery_path.is_file():
        delivery = read_json(delivery_path)
        reasons.extend(str(r) for r in delivery.get("reasons") or [])
        if delivery.get("measurement_complete") is False:
            incomplete = True
        else:
            incomplete = any(
                str(r).startswith("measurement_incomplete:") for r in reasons
            )
    else:
        incomplete = False
    # Per-doc timeout detail from eval smoke scoreboards when present.
    detail = ""
    timeout_n = 0
    for path in camp_dir.glob("runs/*/eval_smoke.json"):
        try:
            payload = read_json(path)
        except Exception:  # noqa: BLE001 — best-effort signal only
            continue
        try:
            timeout_n = max(
                timeout_n,
                int(payload.get("decode_timeout_count") or 0),
                int(payload.get("decode_timeout_document_count") or 0),
            )
        except (TypeError, ValueError):
            pass
        for doc in payload.get("details") or []:
            if not isinstance(doc, dict):
                continue
            d = str(doc.get("decode_outcome_detail") or "")
            if "compiler_ms" in d:
                detail = d
                break
        if detail:
            break
    if not incomplete and timeout_n <= 0 and not detail:
        return False
    signal = is_compiler_ms_timeout_signal(
        reasons=reasons,
        decode_outcome_detail=detail or None,
        incomplete=incomplete or timeout_n > 0,
        decode_timeout_count=timeout_n,
    )
    if not signal:
        return False
    # Budget feedback loop: a timeout whose measured per-record p95 exceeds
    # the timeout the recipe could afford is a budget fact, not a decode-cost
    # residual. It is recalibrated by _fit_screening_decode_timeout_seconds
    # (n_probe / timeout from p95), never routed into DECODE_RESIDUAL_SLUGS.
    from slm_training.autoresearch.thrash_regime import TIMEOUT_CAUSE_BUDGET

    evidence = predecessor_decode_p95_seconds(root, predecessor_campaign_id)
    cause = predecessor_timeout_cause(root, predecessor_campaign_id, evidence=evidence)
    if cause == TIMEOUT_CAUSE_BUDGET:
        print(
            "THRASH_TIMEOUT_CAUSE budget_timeout "
            f"pred={predecessor_campaign_id} "
            f"p95_s={evidence.get('p95_seconds')} "
            f"timeout_s={evidence.get('timeout_seconds')} "
            "route=recalibrate_budget_not_residual",
            flush=True,
        )
        return False
    return True
