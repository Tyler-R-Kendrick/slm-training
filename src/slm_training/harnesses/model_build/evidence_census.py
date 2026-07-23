"""Deterministic, no-write census of committed ship-gate scoreboards."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from slm_training.harness_core.record_schema import normalize_experiment_record
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_MIN_SUITE_N,
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)
from slm_training.versioning import component_version

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUITES = frozenset(DEFAULT_SHIP_GATES)
_ADJUDICATIONS = frozenset(
    {"supported_negative", "inconclusive_until_powered", "invalid/confounded"}
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=_REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scoreboard_suites(value: Any) -> dict[str, dict[str, Any]] | None:
    if not isinstance(value, dict) or not value:
        return None
    if not set(value) <= _SUITES:
        return None
    if not all(isinstance(row, dict) for row in value.values()):
        return None
    return value


def extract_scoreboards(
    payload: Any, *, stem: str = "record"
) -> list[tuple[str, dict[str, dict[str, Any]], dict]]:
    """Select explicit boards, then use the canonical historical normalizer."""
    if not isinstance(payload, dict):
        return []
    found: list[tuple[str, dict[str, dict[str, Any]], dict]] = []
    suites = _scoreboard_suites(payload.get("suites"))
    if suites is not None:
        found.append(("/suites", suites, payload))
    results = payload.get("results")
    if isinstance(results, list):
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                continue
            suites = _scoreboard_suites(row.get("suites"))
            if suites is not None:
                found.append((f"/results/{index}/suites", suites, row))
    if found:
        return found
    record, _reason = normalize_experiment_record(payload, stem=stem)
    if record is None:
        return []
    suites = _scoreboard_suites(record["suites"])
    if suites is None:
        return []
    return [
        (
            str(record.get("source_pointer") or "/normalized"),
            suites,
            record["board_context"],
        )
    ]


def _original_verdict(context: dict[str, Any]) -> Any:
    for key in ("verdict", "decision", "status", "pass", "passed"):
        if key in context:
            return context[key]
    gates = context.get("gates")
    if isinstance(gates, dict) and "pass" in gates:
        return gates["pass"]
    return None


def _claim_direction(verdict: Any) -> str | None:
    if isinstance(verdict, bool):
        return "positive" if verdict else "negative"
    if not isinstance(verdict, str):
        return None
    normalized = verdict.lower().replace("-", "_").replace(" ", "_")
    if any(token in normalized for token in ("positive", "pass", "adopt", "promote")):
        return "positive"
    if any(token in normalized for token in ("negative", "fail", "reject", "not_ship")):
        return "negative"
    return None


def _exact_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    rate_evidence = value.get("rate_evidence")
    if not isinstance(rate_evidence, dict):
        return None
    preferred = (
        "meaningful_program_rate",
        "meaningful_program_v1_rate",
        "parse_rate",
    )
    candidates = [rate_evidence.get(name) for name in preferred]
    candidates.extend(rate_evidence.values())
    for evidence in candidates:
        if not isinstance(evidence, dict):
            continue
        numerator = evidence.get("numerator")
        denominator = evidence.get("denominator")
        interval = evidence.get("interval")
        if (
            not isinstance(numerator, int)
            or isinstance(numerator, bool)
            or not isinstance(denominator, int)
            or isinstance(denominator, bool)
            or denominator <= 0
            or not 0 <= numerator <= denominator
            or not isinstance(interval, dict)
        ):
            continue
        low, high = interval.get("low"), interval.get("high")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return float(low), float(high)
    return None


def _primary_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    direct = _exact_interval(value)
    if direct is not None:
        return direct
    suites = value.get("suites")
    if isinstance(suites, dict):
        for suite_name in DEFAULT_SHIP_GATES:
            interval = _exact_interval(suites.get(suite_name))
            if interval is not None:
                return interval
    return None


def _comparison_overlap(context: dict[str, Any]) -> bool | None:
    for left_name, right_name in (
        ("control", "candidate"),
        ("baseline", "candidate"),
        ("left", "right"),
    ):
        left = _primary_interval(context.get(left_name))
        right = _primary_interval(context.get(right_name))
        if left is not None and right is not None:
            return max(left[0], right[0]) <= min(left[1], right[1])
    return None


def _adjudication(gates: dict[str, Any]) -> str | None:
    if gates["pass"]:
        return None
    if gates["measurement_integrity_failures"] or gates["runtime_failures"]:
        return "invalid/confounded"
    if gates["evidence_volume_failures"]:
        return "inconclusive_until_powered"
    if gates["quality_threshold_failures"]:
        return "supported_negative"
    return "invalid/confounded"


def _reason_codes(gates: dict[str, Any]) -> list[str]:
    return [
        name.removesuffix("_failures")
        for name in (
            "evidence_volume_failures",
            "measurement_integrity_failures",
            "quality_threshold_failures",
            "runtime_failures",
        )
        if gates[name]
    ]


def verify_adjudication_chain(rows: list[dict[str, Any]]) -> None:
    """Reject reordered or mutated persisted adjudication events."""
    previous_event_sha256 = ""
    for row in rows:
        if row.get("previous_event_sha256") != previous_event_sha256:
            raise ValueError("adjudication previous-event hash mismatch")
        candidate = dict(row)
        event_id = candidate.pop("event_id", None)
        if event_id != _canonical_sha(candidate):
            raise ValueError("adjudication event hash mismatch")
        previous_event_sha256 = str(event_id)


def append_adjudications(
    candidates: list[dict[str, Any]],
    prior: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Preserve a verified ledger prefix and append new/superseding events."""
    ledger = [dict(row) for row in prior]
    verify_adjudication_chain(ledger)
    prior_identities = {
        (
            row["source"]["path"],
            row["source"]["file_sha256"],
            row["source"]["json_pointer"],
            row["source"]["scoreboard_sha256"],
            row["gate_replay_sha256"],
        )
        for row in ledger
    }
    superseded = {
        row["supersedes_event_id"]
        for row in ledger
        if row.get("supersedes_event_id")
    }
    def slot(row: dict[str, Any]) -> tuple[str, str]:
        pointer = row["source"]["json_pointer"]
        if pointer == "":
            pointer = "/suites"
        elif re.fullmatch(r"/results/\d+", pointer):
            pointer += "/suites"
        return row["source"]["path"], pointer

    active_by_slot = {
        slot(row): row["event_id"]
        for row in ledger
        if row["event_id"] not in superseded
    }
    previous = ledger[-1]["event_id"] if ledger else ""
    for candidate in candidates:
        identity = (
            candidate["source"]["path"],
            candidate["source"]["file_sha256"],
            candidate["source"]["json_pointer"],
            candidate["source"]["scoreboard_sha256"],
            candidate["gate_replay_sha256"],
        )
        if identity in prior_identities:
            continue
        row = dict(candidate)
        ledger_slot = slot(row)
        row["supersedes_event_id"] = active_by_slot.get(ledger_slot)
        row["previous_event_sha256"] = previous
        row["event_id"] = _canonical_sha(row)
        ledger.append(row)
        previous = row["event_id"]
        active_by_slot[ledger_slot] = row["event_id"]
    return ledger


def build_census(
    revision: str = "HEAD",
    *,
    prior_adjudications: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Replay committed scoreboards at ``revision`` without mutating sources."""
    commit = _git("rev-parse", f"{revision}^{{commit}}")
    committed_at = _git("show", "-s", "--format=%cI", commit)
    implementation_commit = _git("rev-parse", "HEAD")
    implementation_at = _git("show", "-s", "--format=%cI", implementation_commit)
    implementation_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    paths = [
        path
        for path in _git("ls-tree", "-r", "--name-only", commit, "docs/design").splitlines()
        if path.endswith(".json")
    ]
    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    artifacts_with_scoreboards = 0
    claimed_verdicts = 0

    for path in sorted(paths):
        raw = subprocess.check_output(
            ["git", "show", f"{commit}:{path}"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        file_sha = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            exclusions["invalid_json"] += 1
            continue
        if isinstance(payload, dict) and payload.get("schema") == (
            "ship_gate_evidence_census/v1"
        ):
            exclusions["self_generated_census"] += 1
            continue
        scoreboards = extract_scoreboards(payload, stem=Path(path).stem)
        if not scoreboards:
            exclusions["unsupported_or_no_scoreboard"] += 1
            continue
        artifacts_with_scoreboards += 1
        for pointer, suites, context in scoreboards:
            gates = evaluate_ship_gates(suites)
            verdict = _original_verdict(context)
            direction = _claim_direction(verdict)
            overlap = _comparison_overlap(context) if direction else None
            if direction is not None:
                claimed_verdicts += 1
            adjudicated = _adjudication(gates)
            if adjudicated is not None and adjudicated not in _ADJUDICATIONS:
                raise AssertionError(adjudicated)
            scoreboard_sha = _canonical_sha(suites)
            source = {
                "commit": commit,
                "path": path,
                "file_sha256": file_sha,
                "json_pointer": pointer,
                "scoreboard_sha256": scoreboard_sha,
            }
            event_without_id = {
                "source": source,
                "original_verdict": verdict,
                "adjudicated_verdict": adjudicated,
                "reason_codes": _reason_codes(gates),
                "gate_replay_sha256": _canonical_sha(gates),
                "recorded_at": committed_at,
                "actor": "repository_census",
                "claim_direction": direction,
                "paired_interval_overlap": overlap,
            }
            rows.append(
                {
                    **event_without_id,
                    "suite_count": len(suites),
                    "suite_rows_below_min_n": sum(
                        int(
                            isinstance(metrics.get("n"), (int, float))
                            and not isinstance(metrics.get("n"), bool)
                            and float(metrics["n"]) < DEFAULT_MIN_SUITE_N
                        )
                        for metrics in suites.values()
                    ),
                    "gate_pass": gates["pass"],
                    "failure_counts": {
                        name: len(gates[name])
                        for name in (
                            "evidence_volume_failures",
                            "measurement_integrity_failures",
                            "quality_threshold_failures",
                            "runtime_failures",
                        )
                    },
                }
            )

    rows.sort(key=lambda row: (row["source"]["path"], row["source"]["json_pointer"]))
    adjudications = append_adjudications(rows, prior_adjudications)
    adjudication_counts = Counter(
        row["adjudicated_verdict"]
        for row in rows
        if row["adjudicated_verdict"] is not None
    )
    suite_rows = sum(row["suite_count"] for row in rows)
    below_min_n = sum(row["suite_rows_below_min_n"] for row in rows)
    interval_assessable = sum(
        row["claim_direction"] is not None
        and row["paired_interval_overlap"] is not None
        for row in rows
    )
    interval_overlaps = sum(
        row["claim_direction"] is not None and row["paired_interval_overlap"] is True
        for row in rows
    )
    return {
        "schema": "ship_gate_evidence_census/v1",
        "adjudication_schema": "gate_census_adjudications/v1",
        "source_revision": commit,
        "policy": {
            "default_min_suite_n": DEFAULT_MIN_SUITE_N,
            "suite_names": sorted(_SUITES),
            "gate_component_version": component_version("gates.ship"),
        },
        "selection_contract": {
            "paths": "committed docs/design/*.json",
            "accepted_json_pointers": [
                "/suites",
                "/results/<index>/suites",
                "canonical historical normalizer source_pointer",
            ],
            "requires_numeric_n_for_every_suite": False,
            "mutates_source_artifacts": False,
        },
        "summary": {
            "json_artifacts_scanned": len(paths),
            "artifacts_with_scoreboards": artifacts_with_scoreboards,
            "scoreboards_replayed": len(rows),
            "suite_rows_replayed": suite_rows,
            "suite_rows_below_min_n_before_quality": below_min_n,
            "scoreboards_all_present_rows_below_min_n": sum(
                row["suite_rows_below_min_n"] == row["suite_count"] for row in rows
            ),
            "adjudication_counts": dict(sorted(adjudication_counts.items())),
            "claimed_verdicts_found": claimed_verdicts,
            "claimed_verdicts_interval_assessable": interval_assessable,
            "claimed_positives_or_negatives_with_overlapping_intervals": interval_overlaps,
            "interval_overlap_note": (
                "Overlap is computed only for declared control/candidate-style pairs "
                "whose selected rate evidence carries exact binomial counts and "
                "interval bounds. It is descriptive, never a significance test."
            ),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "adjudications": adjudications,
        "version_stamp": {
            "stamp_schema": "version_stamp/v1",
            "code_commit": implementation_commit,
            "code_dirty": implementation_dirty,
            "components": {
                "gates.ship": component_version("gates.ship"),
                "harness.gate_census": component_version("harness.gate_census"),
            },
            "stamped_at": implementation_at,
        },
    }


def render_markdown(census: dict[str, Any]) -> str:
    summary = census["summary"]
    adjudications = summary["adjudication_counts"]
    lines: Iterable[str] = (
        "# SLM-286 ship-gate evidence census",
        "",
        f"- Source revision: `{census['source_revision']}`",
        f"- Census implementation revision: `{census['version_stamp']['code_commit']}`",
        f"- Committed JSON artifacts scanned: `{summary['json_artifacts_scanned']}`",
        f"- Scoreboards replayed: `{summary['scoreboards_replayed']}`",
        f"- Suite rows replayed: `{summary['suite_rows_replayed']}`",
        (
            "- Suite rows below `DEFAULT_MIN_SUITE_N` before reading quality: "
            f"`{summary['suite_rows_below_min_n_before_quality']}`"
        ),
        (
            "- Scoreboards with every present suite below the minimum: "
            f"`{summary['scoreboards_all_present_rows_below_min_n']}`"
        ),
        f"- `supported_negative`: `{adjudications.get('supported_negative', 0)}`",
        (
            "- `inconclusive_until_powered`: "
            f"`{adjudications.get('inconclusive_until_powered', 0)}`"
        ),
        f"- `invalid/confounded`: `{adjudications.get('invalid/confounded', 0)}`",
        (
            "- Claimed positives/negatives with assessable paired intervals: "
            f"`{summary['claimed_verdicts_interval_assessable']}`"
        ),
        (
            "- Claimed positives/negatives with overlapping intervals: "
            f"`{summary['claimed_positives_or_negatives_with_overlapping_intervals']}`"
        ),
        "",
        "## Recipe and decision",
        "",
        "- Device/backend: CPU-only Git object replay; no model or checkpoint.",
        "- Matrix set: committed `docs/design/*.json` explicit boards plus the canonical historical normalizer.",
        "- Honesty mode: current ship policy replay; evidence reachability before quality.",
        "- Result: historical negative labels are not currently supportable as model-quality negatives.",
        "- Ship/default/promotion: none.",
        "",
        "## Provenance and limitations",
        "",
        "- Every adjudication row binds the source commit, file SHA-256, JSON pointer, canonical scoreboard SHA-256, and gate-replay SHA-256.",
        "- Source artifacts are immutable; verified prior ledger bytes remain a prefix and corrections append superseding events.",
        "- Unsupported artifacts are counted as exclusions; canonical nested and legacy single-suite records are normalized once.",
        f"- {summary['interval_overlap_note']}",
        "- OpenWiki source instructions were updated; local regeneration was not claimed because the non-interactive provider token was unavailable.",
        "",
        "Interval overlap is descriptive only and is never treated as a significance test.",
        "Historical source artifacts were not modified; adjudications are append-only rows in the JSON census.",
        "",
        "## Statistical lineage",
        "",
        "- Wilson (1927), DOI: https://doi.org/10.1080/01621459.1927.10502953",
        "- Hoenig and Heisey (2001), DOI: https://doi.org/10.1198/000313001300339897",
        "",
    )
    return "\n".join(lines)
