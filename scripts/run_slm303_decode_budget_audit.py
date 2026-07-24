#!/usr/bin/env python3
"""SLM-303 (LAR1-05): decode-budget audit — census, budget sweep, disposition.

Question: which historical zero / interference-flagged scoreboards are model
behavior, and which are runtime/harness budget artifacts (decode timeouts,
fallbacks)? Answered three ways:

1. ``census`` — enumerate every local ``outputs/runs/*/checkpoints/*.pt`` plus
   every MODEL_CARD roster row; verify committed SHA fragments
   (prefix…suffix) against actual sha256 of local bytes; group every local
   ``eval_*.json`` scoreboard (and committed docs/design iter JSONs for
   historical context) by checkpoint; classify each scoreboard with the
   SLM-303 decode-outcome taxonomy (``unmeasured`` when fallback_count is
   null). Hash-pins the two remediated checkpoints that carry the only
   nonzero-timeout scoreboards.
2. ``sweep-cell`` / ``report`` — PREREGISTERED budget sweep: same checkpoint,
   same 3 smoke records (test_seeds smoke ids — the original suite directory
   is empty, so the suite is rebuilt from test_seeds; disclosed), same seed,
   checkpoint-declared decode policy; arms differ ONLY in budget: baseline
   (recorded canvas cap 256/128, inferred 10s decode timeout from the recorded
   3×3333.79ms chunk latency) vs 10x (100s, 2560/1280). Each
   (checkpoint, arm, record) cell runs in its own hard-killed worker process
   so every command stays under the run cap; a killed cell is ``decode_timeout``
   evidence, never a fabricated result.
3. ``report`` — disposition: append-only historical annotations + a small
   powered-rerun recommendation, written to
   ``docs/design/iter-slm303-decode-budget-audit-20260724.{json,md}``.

Example:
  python -m scripts.run_slm303_decode_budget_audit census
  python -m scripts.run_slm303_decode_budget_audit sweep-cell \
      --checkpoint ltr2 --arm baseline --record-id smoke_hero_01
  python -m scripts.run_slm303_decode_budget_audit report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build.decode_outcome import (
    DECODE_OUTCOMES,
    DecodeOutcomeRecordV1,
    classify_decode_outcome,
    outcome_counts,
)
from slm_training.versioning import build_version_stamp

COMPONENT = "harness.experiments.slm303_decode_budget_audit"
EVAL_COMPONENT = "harness.model_build.eval"

RUNS_ROOT = Path("outputs/runs")
MODEL_CARD = Path("docs/MODEL_CARD.md")
DESIGN_DIR = Path("docs/design")
TEST_SEEDS = Path("src/slm_training/resources/test_seeds.jsonl")

AUDIT_DIR = RUNS_ROOT / "slm303_decode_budget_audit"
CENSUS_JSON = AUDIT_DIR / "census.json"
CELLS_DIR = AUDIT_DIR / "cells"

DEFAULT_JSON_OUT = DESIGN_DIR / "iter-slm303-decode-budget-audit-20260724.json"
DEFAULT_MD_OUT = DESIGN_DIR / "iter-slm303-decode-budget-audit-20260724.md"

# The only two checkpoints with ACTUAL nonzero-timeout scoreboards. Canvas
# caps and timeout evidence come from their recorded eval_smoke.json.
SWEEP_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "ltr2": {
        "path": "outputs/runs/iter-published-remediated-64step-ltr2-20260715/checkpoints/best_weighted_nll.pt",
        "scoreboard": "outputs/runs/iter-published-remediated-64step-ltr2-constrained-fullsmoke-20260715/eval_smoke.json",
        "recorded_canvas_cap": 256,
        "recorded_decode_timeout_count": 3,
    },
    "lexer_ltr2": {
        "path": "outputs/runs/iter-published-remediated-64step-lexer-ltr2-20260715/checkpoints/best_weighted_nll.pt",
        "scoreboard": "outputs/runs/iter-published-remediated-64step-lexer-ltr2-constrained-fullsmoke-20260715/eval_smoke.json",
        "recorded_canvas_cap": 128,
        "recorded_decode_timeout_count": 3,
    },
}

# The recorded timeout scoreboards show latency_ms = 3333.79 on all 3 smoke
# records: one chunk of 3 killed at ~10001ms → decode_timeout_seconds = 10.
BASELINE_TIMEOUT_S = 10.0
BASELINE_TIMEOUT_EVIDENCE = (
    "inferred from recorded eval_smoke.json: all 3 smoke rows share "
    "latency_ms=3333.79 (one chunk of 3 killed at ~10001ms → 10s timeout); "
    "the original CLI flag was not persisted (config=null), disclosed as inferred"
)
SWEEP_BUDGET_MULTIPLIER = 10

SMOKE_RECORD_IDS = ("smoke_hero_01", "smoke_button_01", "smoke_callout_01")

_SHA_FRAGMENT_RE = re.compile(r"\b([0-9a-f]{6,})…([0-9a-f]{4,})\b")
_SHA_FULL_RE = re.compile(r"\b[0-9a-f]{64}\b")
_CHECKPOINT_PATH_RE = re.compile(r"outputs/runs/\S*?checkpoints/[\w.\-]+\.pt")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_matches_committed(committed: str | None, actual: str) -> bool | None:
    """Verify a committed SHA reference against actual bytes.

    ``committed`` may be a full 64-hex digest or a ``prefix…suffix`` fragment
    (as used in MODEL_CARD roster rows). Returns None when no committed
    reference exists (unverifiable, never a pass).
    """
    if not committed:
        return None
    committed = committed.strip().strip("`")
    if _SHA_FULL_RE.fullmatch(committed):
        return committed == actual
    match = _SHA_FRAGMENT_RE.search(committed)
    if not match:
        return None
    prefix, suffix = match.groups()
    return actual.startswith(prefix) and actual.endswith(suffix)


def classify_scoreboard(scoreboard: dict[str, Any]) -> dict[str, Any]:
    """Classify one scoreboard dict with the decode-outcome taxonomy.

    Scoreboard-level view: ``unmeasured`` when fallback telemetry is absent
    (fallback_count null), ``runtime_timeout_interference`` when decode
    timeouts fired, ``fallback_interference`` when fallbacks fired, else
    ``model_behavior`` — a clean zero is a model verdict, not a budget
    artifact. Per-record rows are classified when details[] exists.
    """
    fallback_count = scoreboard.get("fallback_count")
    timeout_count = scoreboard.get("decode_timeout_count")
    n = scoreboard.get("n")
    row_outcomes: list[dict[str, Any]] = []
    for row in scoreboard.get("details") or []:
        if not isinstance(row, dict) or "id" not in row:
            continue
        row_outcomes.append(
            {
                "id": row.get("id"),
                "outcome": classify_decode_outcome(
                    parse_ok=row.get("parse_ok"),
                    error=row.get("error"),
                    fallback_counters=int(row.get("fallback_count") or 0),
                    timed_out=False,  # historical rows carry no timeout fact
                    abstained=not str(row.get("prediction") or "").strip(),
                    harness_exception=False,
                ),
                "note": (
                    "historical row: no per-record timeout/fallback fields; "
                    "timeout attribution comes from the suite-level count"
                ),
            }
        )
    if fallback_count is None:
        classification = "unmeasured"
    elif timeout_count:
        classification = "runtime_timeout_interference"
    elif fallback_count:
        classification = "fallback_interference"
    else:
        classification = "model_behavior"
    return {
        "classification": classification,
        "n": n,
        "decode_timeout_count": timeout_count,
        "fallback_count": fallback_count,
        "decode_canvas_cap": scoreboard.get("decode_canvas_cap"),
        "parse_rate": scoreboard.get("parse_rate"),
        "meaningful_program_rate": scoreboard.get("meaningful_program_rate"),
        "row_outcomes": row_outcomes,
    }


def build_sweep_arms(recorded_canvas_cap: int) -> dict[str, dict[str, Any]]:
    """Preregistered arms; they differ ONLY in budget fields."""
    return {
        "baseline": {
            "timeout_s": BASELINE_TIMEOUT_S,
            "canvas_cap": recorded_canvas_cap,
            "timeout_evidence": BASELINE_TIMEOUT_EVIDENCE,
        },
        "budget10x": {
            "timeout_s": BASELINE_TIMEOUT_S * SWEEP_BUDGET_MULTIPLIER,
            "canvas_cap": recorded_canvas_cap * SWEEP_BUDGET_MULTIPLIER,
            "timeout_evidence": "10x the preregistered baseline budget",
        },
    }


def pair_cells(
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair baseline vs budget10x cells per (checkpoint, record); assert the
    arms differ only in budget fields (paired-design check)."""
    pairs: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        by_key.setdefault((cell["checkpoint"], cell["record_id"]), {})[
            cell["arm"]
        ] = cell
    for (checkpoint, record_id), arms in sorted(by_key.items()):
        base = arms.get("baseline")
        wide = arms.get("budget10x")
        if base is None or wide is None:
            pairs.append(
                {
                    "checkpoint": checkpoint,
                    "record_id": record_id,
                    "status": "unpaired",
                    "arms_present": sorted(arms),
                }
            )
            continue
        base_budget = base["budget"]
        wide_budget = wide["budget"]
        non_budget_mismatch = [
            field
            for field in ("checkpoint_sha256", "seed", "record_sha256")
            if base.get(field) != wide.get(field)
        ]
        pairs.append(
            {
                "checkpoint": checkpoint,
                "record_id": record_id,
                "status": "paired" if not non_budget_mismatch else "pair_violation",
                "pair_violation_fields": non_budget_mismatch,
                "budget_baseline": base_budget,
                "budget_10x": wide_budget,
                "outcome_baseline": base["outcome"],
                "outcome_10x": wide["outcome"],
                "cell_status_baseline": base.get("status"),
                "cell_status_10x": wide.get("status"),
                "parse_ok_baseline": base.get("parse_ok"),
                "parse_ok_10x": wide.get("parse_ok"),
                "elapsed_ms_baseline": base.get("elapsed_ms"),
                "elapsed_ms_10x": wide.get("elapsed_ms"),
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------


def _roster_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        for path in _CHECKPOINT_PATH_RE.findall(line):
            fragment = _SHA_FRAGMENT_RE.search(line)
            full = _SHA_FULL_RE.search(line)
            committed = None
            if full:
                committed = full.group(0)
            elif fragment:
                committed = f"{fragment.group(1)}…{fragment.group(2)}"
            rows.append({"path": path, "committed_sha": committed})
    return rows


def _iter_scoreboards() -> list[tuple[Path, dict[str, Any]]]:
    """Local eval_*.json scoreboards (path, payload)."""
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(RUNS_ROOT.glob("*/eval*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - unreadable artifact is skipped, noted
            continue
        if isinstance(payload, dict) and (
            "decode_timeout_count" in payload or "fallback_count" in payload
        ):
            out.append((path, payload))
    return out


def _design_scoreboards() -> list[tuple[Path, dict[str, Any]]]:
    """Committed docs/design iter JSONs carrying scoreboard-shaped dicts."""
    out: list[tuple[Path, dict[str, Any]]] = []

    def _walk(node: Any, path: Path) -> None:
        if isinstance(node, dict):
            if "decode_timeout_count" in node and "fallback_count" in node:
                out.append((path, node))
                return
            for value in node.values():
                _walk(value, path)
        elif isinstance(node, list):
            for value in node:
                _walk(value, path)

    for path in sorted(DESIGN_DIR.glob("iter*.json")):
        try:
            _walk(json.loads(path.read_text(encoding="utf-8")), path)
        except Exception:  # noqa: BLE001
            continue
    return out


def run_census(*, out_path: Path = CENSUS_JSON) -> dict[str, Any]:
    roster = _roster_rows(MODEL_CARD.read_text(encoding="utf-8"))
    roster_by_path = {row["path"]: row for row in roster}

    local_checkpoints = sorted(RUNS_ROOT.glob("*/checkpoints/*.pt"))
    checkpoints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in local_checkpoints:
        rel = str(path)
        seen.add(rel)
        actual = sha256_file(path)
        committed = (roster_by_path.get(rel) or {}).get("committed_sha")
        checkpoints.append(
            {
                "path": rel,
                "exists_locally": True,
                "in_roster": rel in roster_by_path,
                "committed_sha": committed,
                "actual_sha256": actual,
                "committed_sha_match": sha_matches_committed(committed, actual),
            }
        )
    for rel, row in sorted(roster_by_path.items()):
        if rel not in seen:
            checkpoints.append(
                {
                    "path": rel,
                    "exists_locally": False,
                    "in_roster": True,
                    "committed_sha": row["committed_sha"],
                    "actual_sha256": None,
                    "committed_sha_match": None,
                }
            )

    scoreboards: list[dict[str, Any]] = []
    for path, payload in _iter_scoreboards():
        entry = {
            "scoreboard_path": str(path),
            "source": "local_outputs",
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "suite": payload.get("suite"),
            **classify_scoreboard(payload),
        }
        scoreboards.append(entry)
    for path, node in _design_scoreboards():
        scoreboards.append(
            {
                "scoreboard_path": str(path),
                "source": "docs_design_historical",
                "checkpoint": node.get("checkpoint"),
                "checkpoint_sha256": node.get("checkpoint_sha256"),
                "suite": node.get("suite"),
                **classify_scoreboard(node),
            }
        )

    hash_pins: dict[str, Any] = {}
    for key, spec in SWEEP_CHECKPOINTS.items():
        path = Path(spec["path"])
        pin: dict[str, Any] = {"path": spec["path"], "exists_locally": path.exists()}
        if path.exists():
            actual = sha256_file(path)
            pin["actual_sha256"] = actual
            recorded = None
            sb_path = Path(spec["scoreboard"])
            if sb_path.exists():
                recorded = json.loads(sb_path.read_text(encoding="utf-8")).get(
                    "checkpoint_sha256"
                )
            pin["scoreboard_recorded_sha256"] = recorded
            pin["matches_scoreboard_record"] = (
                None if recorded is None else actual == recorded
            )
        hash_pins[key] = pin

    census = {
        "schema": "slm303_decode_census/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "taxonomy": list(DECODE_OUTCOMES),
        "checkpoints": checkpoints,
        "scoreboards": scoreboards,
        "hash_pins": hash_pins,
        "summary": {
            "n_local_checkpoints": len(local_checkpoints),
            "n_roster_rows": len(roster),
            "n_committed_sha_verified": sum(
                1 for c in checkpoints if c["committed_sha_match"] is True
            ),
            "n_committed_sha_mismatch": sum(
                1 for c in checkpoints if c["committed_sha_match"] is False
            ),
            "n_unverifiable": sum(
                1 for c in checkpoints if c["committed_sha_match"] is None
            ),
            "scoreboard_classes": dict(
                sorted(
                    {
                        cls: sum(
                            1 for s in scoreboards if s["classification"] == cls
                        )
                        for cls in {
                            s["classification"] for s in scoreboards
                        }
                    }.items()
                )
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")
    return census


# ---------------------------------------------------------------------------
# Sweep cells (hard-killed worker per (checkpoint, arm, record))
# ---------------------------------------------------------------------------


def _load_smoke_records() -> dict[str, Any]:
    from slm_training.dsl.schema import ExampleRecord

    records: dict[str, Any] = {}
    with TEST_SEEDS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["id"] in SMOKE_RECORD_IDS:
                records[row["id"]] = ExampleRecord(
                    id=row["id"],
                    prompt=row["prompt"],
                    openui=row["openui"],
                    placeholders=list(row.get("placeholders", [])),
                    split=row.get("split", "smoke"),
                    source=row.get("source", "smoke"),
                )
    return records


def _cell_worker(
    checkpoint_path: str, arm_budget: dict[str, Any], record_id: str, queue: Any
) -> None:
    """Load the checkpoint fresh, decode one smoke record under the arm budget."""
    try:
        from slm_training.harnesses.model_build.plugin import GenerationRequest
        from slm_training.models.twotower import TwoTowerModel

        record = _load_smoke_records()[record_id]
        model = TwoTowerModel.from_checkpoint(checkpoint_path, device="cpu")
        model.config.grammar_ltr_max_tokens = int(arm_budget["canvas_cap"])
        request = GenerationRequest.from_record(record)
        t0 = time.perf_counter()
        pred = model.generate_batch_requests(
            [request], max_len=int(arm_budget["canvas_cap"])
        )[0]
        queue.put(
            {
                "status": "ok",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "prediction": pred,
            }
        )
    except Exception as exc:  # noqa: BLE001 - decode failure is data
        queue.put({"status": "harness_error", "error": str(exc)[:400]})


def run_sweep_cell(
    *,
    checkpoint_key: str,
    arm: str,
    record_id: str,
    out_dir: Path = CELLS_DIR,
) -> dict[str, Any]:
    import multiprocessing as mp

    spec = SWEEP_CHECKPOINTS[checkpoint_key]
    arms = build_sweep_arms(int(spec["recorded_canvas_cap"]))
    budget = arms[arm]
    checkpoint_path = spec["path"]
    checkpoint_sha = sha256_file(Path(checkpoint_path))
    record = _load_smoke_records()[record_id]
    record_sha = hashlib.sha256(
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    ctx = mp.get_context("fork")
    queue: Any = ctx.Queue()
    proc = ctx.Process(
        target=_cell_worker, args=(checkpoint_path, budget, record_id, queue)
    )
    t0 = time.perf_counter()
    proc.start()
    proc.join(float(budget["timeout_s"]))
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if proc.is_alive():
        proc.terminate()
        proc.join(30)
        outcome = classify_decode_outcome(
            parse_ok=None, timed_out=True, fallback_counters=0
        )
        cell = {
            "status": "decode_timeout",
            "outcome": outcome,
            "stop_reason": "decode_timeout",
            "elapsed_ms": elapsed_ms,
            "prediction": None,
            "parse_ok": None,
        }
    else:
        result = (
            queue.get()
            if not queue.empty()
            else {
                "status": "harness_error",
                "error": f"worker exited without result (code {proc.exitcode})",
            }
        )
        if result["status"] == "ok":
            pred = str(result["prediction"])
            from slm_training.dsl.parser import validate

            try:
                validate(pred)
                parse_ok: bool | None = True
                error = None
            except Exception as exc:  # noqa: BLE001 - parse fact, never raised
                parse_ok = False
                error = str(exc)[:200]
            outcome = classify_decode_outcome(
                parse_ok=parse_ok,
                error=error,
                abstained=not pred.strip(),
                fallback_counters=0,
                timed_out=False,
            )
            cell = {
                "status": "ok",
                "outcome": outcome,
                "stop_reason": "empty_prediction" if not pred.strip() else "completed",
                "elapsed_ms": result["elapsed_ms"],
                "prediction": pred,
                "parse_ok": parse_ok,
                "error": error,
            }
        else:
            error_text = str(result.get("error") or "")
            not_rerunnable = "output contract" in error_text
            cell = {
                "status": "not_rerunnable" if not_rerunnable else "harness_error",
                # No decode attempt happened → no outcome, never a guess.
                "outcome": None if not_rerunnable else classify_decode_outcome(
                    parse_ok=None, harness_exception=True
                ),
                "stop_reason": "not_rerunnable" if not_rerunnable else "harness_error",
                "elapsed_ms": elapsed_ms,
                "prediction": None,
                "parse_ok": None,
                "error": result.get("error"),
            }

    record_out = (
        DecodeOutcomeRecordV1(
            request_id=record_id,
            outcome=cell["outcome"],
            stop_reason=cell["stop_reason"],
            budget={
                "timeout_s": budget["timeout_s"],
                "canvas_cap": budget["canvas_cap"],
                "gen_steps": None,
                "max_attempts": 1,
            },
            elapsed_ms=cell["elapsed_ms"],
            fallback_used=False,
            detail=cell.get("error"),
        )
        if cell["outcome"] is not None
        else None
    )
    budget_dict = {
        "timeout_s": budget["timeout_s"],
        "canvas_cap": budget["canvas_cap"],
        "gen_steps": None,
        "max_attempts": 1,
    }
    cell.update(
        {
            "schema": "slm303_sweep_cell/v1",
            "checkpoint": checkpoint_key,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "arm": arm,
            "budget": budget_dict,
            "record_id": record_id,
            "record_sha256": record_sha,
            "seed": 0,
            "outcome_record": record_out.to_dict() if record_out else None,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{checkpoint_key}__{arm}__{record_id}.json"
    path.write_text(json.dumps(cell, indent=2) + "\n", encoding="utf-8")
    return cell


# ---------------------------------------------------------------------------
# Report (census + sweep cells → disposition + durable docs)
# ---------------------------------------------------------------------------


def _predeclaration() -> dict[str, Any]:
    return {
        "experiment_id": "SLM-303",
        "hypothesis": (
            "A material share of historical zero/interference-flagged "
            "scoreboards are runtime/harness budget artifacts (decode timeout "
            "or fallback), not model behavior; widening the decode budget 10x "
            "flips timeout-classified rows to model outcomes."
        ),
        "falsifier": (
            "10x budget does not change any timeout-classified record's "
            "outcome on the hash-pinned checkpoints, or the census shows the "
            "flagged scoreboards were already clean model behavior."
        ),
        "arms": {
            "baseline": "recorded canvas cap (256/128) + inferred 10s timeout",
            "budget10x": "10x timeout and 10x canvas cap; nothing else varies",
        },
        "budget_evidence": BASELINE_TIMEOUT_EVIDENCE,
        "records": {
            "ids": list(SMOKE_RECORD_IDS),
            "source": str(TEST_SEEDS),
            "disclosure": (
                "the original eval suite directory is empty; the sweep "
                "rebuilds the same 3 smoke records (ids match the recorded "
                "scoreboard details) from test_seeds.jsonl"
            ),
        },
        "seed": 0,
        "fixed": (
            "same checkpoint bytes (hash-pinned), same records, same seed, "
            "checkpoint-declared decode policy; arms differ only in budget"
        ),
        "checkpoints": {
            key: {
                "path": spec["path"],
                "recorded_canvas_cap": spec["recorded_canvas_cap"],
                "recorded_decode_timeout_count": spec["recorded_decode_timeout_count"],
                "selection": "only checkpoints with ACTUAL nonzero-timeout scoreboards",
            }
            for key, spec in SWEEP_CHECKPOINTS.items()
        },
        "excluded": {
            "slm230_bounded_recursive_r4_r2": (
                "zero-timeout scoreboard; budget interference not implicated — not needed"
            ),
            "e173/slm294 and all other local checkpoints": (
                "no committed SHA and no hash-pinned timeout scoreboard → not "
                "hash-verifiable for this audit (slm294 additionally "
                "feasibility-excluded: >280s/16 tokens constrained decode on rico)"
            ),
        },
        "claim_class": "diagnostic",
    }


def _disposition(
    census: dict[str, Any], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    timeout_boards = [
        s
        for s in census["scoreboards"]
        if s["classification"] == "runtime_timeout_interference"
    ]
    fallback_boards = [
        s
        for s in census["scoreboards"]
        if s["classification"] == "fallback_interference"
    ]
    unmeasured_boards = [
        s for s in census["scoreboards"] if s["classification"] == "unmeasured"
    ]
    flips = [
        p
        for p in pairs
        if p.get("status") == "paired"
        and p.get("outcome_baseline") == "runtime_timeout"
        and p.get("outcome_10x") != "runtime_timeout"
    ]
    not_rerunnable = [p for p in pairs if p.get("outcome_baseline") is None]
    annotations = []
    for board in timeout_boards:
        annotations.append(
            {
                "scoreboard": board["scoreboard_path"],
                "annotation": (
                    f"runtime/harness artifact: {board['decode_timeout_count']}"
                    f"/{board['n']} decode timeouts at canvas cap "
                    f"{board['decode_canvas_cap']}; zero parse rate on these "
                    "rows is budget interference, NOT a model verdict"
                ),
            }
        )
    for board in fallback_boards:
        annotations.append(
            {
                "scoreboard": board["scoreboard_path"],
                "annotation": (
                    f"fallback interference: fallback_count={board['fallback_count']}; "
                    "fallback outputs never classify as model success"
                ),
            }
        )
    if not pairs:
        sweep_verdict = "sweep cells not run"
    elif not_rerunnable:
        sweep_verdict = (
            f"not_rerunnable: {len(not_rerunnable)}/{len(pairs)} paired cells "
            "blocked by the output-contract gate (checkpoint v0 vs required "
            "symbol_only/v2, no migration path); the 10x-budget flip question "
            "is UNANSWERED by re-decode — census + taxonomy carry the audit"
        )
    else:
        sweep_verdict = (
            f"{len(flips)} timeout-classified rows flipped outcome under 10x budget"
        )
    recommendation = {
        "powered_rerun": (
            "at most one: retrain the remediated recipe from symbol-only "
            "targets (output contract v2 — the v0 checkpoints are blocked by "
            "require_current_output_contract with no migration path), then a "
            "preregistered full-smoke re-eval at recorded vs 10x decode "
            "budget with per-record decode_outcome fields (now emitted by "
            "eval_runner), n≥16 smoke+fixture records for Wilson resolution"
        ),
        "rationale": (
            "census localizes all timeout interference to the two remediated "
            "checkpoints; no other hash-verifiable checkpoint carries a "
            "timeout-flagged scoreboard, and re-decode of the v0 checkpoints "
            "under current code is impossible"
        ),
    }
    return {
        "annotations": annotations,
        "counts": {
            "runtime_timeout_interference": len(timeout_boards),
            "fallback_interference": len(fallback_boards),
            "unmeasured": len(unmeasured_boards),
        },
        "sweep_verdict": sweep_verdict,
        "recommendation": recommendation,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["census"]["summary"]
    lines = [
        "# SLM-303 (LAR1-05): decode-budget audit — census, 10x sweep, disposition",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- claim_class: `{payload['predeclaration']['claim_class']}` (diagnostic; not ship evidence)",
        f"- taxonomy: `{', '.join(payload['census']['taxonomy'])}`",
        f"- budget evidence: {payload['predeclaration']['budget_evidence']}",
        "",
        "## Census",
        "",
        f"- local checkpoints enumerated: {summary['n_local_checkpoints']} "
        f"(roster rows: {summary['n_roster_rows']})",
        f"- committed SHA verified: {summary['n_committed_sha_verified']}; "
        f"mismatch: {summary['n_committed_sha_mismatch']}; "
        f"unverifiable (no committed SHA): {summary['n_unverifiable']}",
        f"- scoreboard classes: `{json.dumps(summary['scoreboard_classes'])}`",
        "",
        "### Hash pins (sweep checkpoints)",
        "",
        "| key | path | sha256 | matches scoreboard record |",
        "| --- | --- | --- | --- |",
    ]
    for key, pin in payload["census"]["hash_pins"].items():
        lines.append(
            f"| {key} | `{pin['path']}` | `{pin.get('actual_sha256')}` | "
            f"{pin.get('matches_scoreboard_record')} |"
        )
    lines += ["", "## Budget sweep (preregistered, paired)", ""]
    sweep = payload.get("sweep") or {}
    if sweep.get("pairs"):
        lines += [
            "| checkpoint | record | baseline outcome | 10x outcome | baseline ms | 10x ms |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for pair in sweep["pairs"]:
            if pair.get("status") != "paired":
                lines.append(
                    f"| {pair['checkpoint']} | {pair['record_id']} | unpaired | | | |"
                )
                continue
            lines.append(
                f"| {pair['checkpoint']} | {pair['record_id']} | "
                f"{pair['outcome_baseline'] or pair.get('cell_status_baseline')} | "
                f"{pair['outcome_10x'] or pair.get('cell_status_10x')} | "
                f"{pair['elapsed_ms_baseline']} | {pair['elapsed_ms_10x']} |"
            )
        lines += [
            "",
            f"- outcome counts (baseline): `{json.dumps(sweep['outcome_counts']['baseline'])}`",
            f"- outcome counts (10x): `{json.dumps(sweep['outcome_counts']['budget10x'])}`",
        ]
    else:
        lines.append("- sweep cells not present; see census + disposition only")
    lines += ["", "## Disposition (append-only annotations)", ""]
    disp = payload["disposition"]
    for note in disp["annotations"]:
        lines.append(f"- `{note['scoreboard']}`: {note['annotation']}")
    lines += [
        "",
        f"- sweep verdict: {disp['sweep_verdict']}",
        f"- powered-rerun recommendation: {disp['recommendation']['powered_rerun']}",
        f"  - rationale: {disp['recommendation']['rationale']}",
        "",
        "Historical iter docs and MODEL_CARD rows are untouched; these "
        "annotations are additive.",
        "",
    ]
    return "\n".join(lines)


def run_report(
    *,
    json_out: Path = DEFAULT_JSON_OUT,
    md_out: Path = DEFAULT_MD_OUT,
) -> dict[str, Any]:
    census = run_census()
    cells: list[dict[str, Any]] = []
    if CELLS_DIR.exists():
        for path in sorted(CELLS_DIR.glob("*.json")):
            cells.append(json.loads(path.read_text(encoding="utf-8")))
    pairs = pair_cells(cells)
    outcomes_by_arm = {
        arm: outcome_counts(
            [c["outcome"] for c in cells if c["arm"] == arm and c["outcome"]]
        )
        for arm in ("baseline", "budget10x")
    }
    payload: dict[str, Any] = {
        "schema": "slm303_decode_budget_audit/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predeclaration": _predeclaration(),
        "census": census,
        "sweep": {
            "n_cells": len(cells),
            "pairs": pairs,
            "outcome_counts": outcomes_by_arm,
            "cells": cells,
        },
        "disposition": _disposition(census, pairs),
    }
    payload["version_stamp"] = build_version_stamp(COMPONENT, EVAL_COMPONENT)
    json_out.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    md_out.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("census", help="checkpoint/scoreboard census + hash pins")
    cell = sub.add_parser("sweep-cell", help="run one (checkpoint, arm, record) cell")
    cell.add_argument("--checkpoint", required=True, choices=sorted(SWEEP_CHECKPOINTS))
    cell.add_argument("--arm", required=True, choices=("baseline", "budget10x"))
    cell.add_argument("--record-id", required=True, choices=list(SMOKE_RECORD_IDS))
    rep = sub.add_parser("report", help="merge census + cells into durable docs")
    rep.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    rep.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    if args.mode == "census":
        census = run_census()
        print(json.dumps(census["summary"], indent=2))
        print(f"wrote {CENSUS_JSON}")
        return 0
    if args.mode == "sweep-cell":
        result = run_sweep_cell(
            checkpoint_key=args.checkpoint,
            arm=args.arm,
            record_id=args.record_id,
        )
        print(
            f"{args.checkpoint}/{args.arm}/{args.record_id}: "
            f"{result['outcome']} ({result['status']}, {result['elapsed_ms']}ms)"
        )
        return 0
    payload = run_report(json_out=args.json_out, md_out=args.md_out)
    print(f"sweep verdict: {payload['disposition']['sweep_verdict']}")
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
