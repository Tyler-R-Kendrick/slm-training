"""SLM-288 / LAR0-04: meaningful-program metric validation harness.

Subcommands:

* ``histograms`` — replay the typed v1 reason-code report (and v2 codes) over
  the committed frontier replay bundle; emit per-suite histograms.
* ``freeze-packet`` — build the frozen, hash-pinned, blinded annotation
  packet + rubric from retained generation envelopes. Declares any shortfall
  versus the ~100-output target honestly instead of padding.
* ``export-blind`` — write the rater-facing blinded JSONL (no provenance).
* ``analyze`` — import ≥2 independent rater label files; compute Cohen's
  kappa, adjudication rate, metric precision/recall overall and by reason
  code, the predeclared recall-floor sensitivity sweep {0.3, 0.5, 0.7}, and
  the metric disposition.

All steps are deterministic and no-write outside their explicit outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.judge_independence import cohen_kappa
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.harnesses.model_build.eval_runner import (
    MEANINGFUL_V1_REASON_SCHEMA,
    meaningful_program_v1_report,
)

REPLAY_BUNDLE = Path("src/slm_training/resources/evals/meaningful_v2_frontier_replay.json")
PACKET_PATH = Path("src/slm_training/resources/evals/slm288_annotation_packet_v1.json")
RUBRIC_PATH = Path("src/slm_training/resources/evals/slm288_meaningful_rubric_v1.md")

PACKET_SCHEMA = "slm288_annotation_packet/v1"
LABEL_SCHEMA = "slm288_meaningful_label/v1"
ANALYSIS_SCHEMA = "slm288_meaningful_validation/v1"
DECLARED_PACKET_TARGET = 100
PREDECLARED_RECALL_FLOORS = (0.3, 0.5, 0.7)

RUBRIC = """# SLM-288 meaningful-program rubric (v1)

You are labeling UI-layout programs (OpenUI Lang) for **semantic
meaningfulness**: does this output constitute a genuine attempt at the
requested layout, rather than a trivial, empty, or off-task shell?

Label each item exactly one of:

- `meaningful` — a real attempt: parses, has populated containers with
  content components, binds the requested data placeholders, and covers the
  component types the prompt asks for.
- `not_meaningful` — trivial or off-task: unparseable, empty
  containers (e.g. `Stack([])` / `Card([])`), no content components, no
  placeholders, or clearly missing the requested component inventory.
- `unknown` — you cannot determine meaningfulness from the prompt and
  output shown (e.g. output appears truncated). Unknown is never counted as
  a failure or a success.

Also record any applicable reason codes:
`parse_failed`, `free_form_output_string`, `empty_root_stack`, `empty_card`,
`empty_children`, `no_content_components`, `no_placeholders`,
`low_component_recall`, `component_recall_unobservable`.

You are blinded: model/checkpoint identity is intentionally withheld. Judge
only the prompt and output shown. Do not look up repository artifacts.
"""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_bundle(path: Path = REPLAY_BUNDLE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_records(bundle: dict[str, Any]) -> dict[str, ExampleRecord]:
    records: dict[str, ExampleRecord] = {}
    for key, raw in (bundle.get("records") or {}).items():
        record = ExampleRecord(
            id=str(raw["id"]),
            prompt=str(raw["prompt"]),
            openui=str(raw["openui"]),
            placeholders=raw.get("placeholders"),
            split=raw.get("split"),
            source=raw.get("source"),
            meta=raw.get("meta"),
        )
        records[str(record.id)] = record
        records[key] = record
    return records


def _iter_rows(bundle: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for eval_set in bundle["sets"]:
        for detail in eval_set["details"]:
            yield eval_set, detail


def score_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Score every retained row with the typed v1 report and v2 codes."""
    records = _bundle_records(bundle)
    rows: list[dict[str, Any]] = []
    for eval_set, detail in _iter_rows(bundle):
        pred = detail.get("prediction") or ""
        record = records.get(str(detail.get("id", "")))
        report = meaningful_program_v1_report(pred, gold=record)
        row: dict[str, Any] = {
            "set_label": eval_set["label"],
            "suite": eval_set["suite"],
            "id": detail.get("id"),
            "prediction": pred,
            "prediction_sha256": detail.get("prediction_sha256"),
            "v1_report": report,
        }
        if record is not None:
            v2 = binding_aware_meaningful_v2(pred, record=record)
            row["v2_verdict"] = v2.verdict
            row["v2_reason_codes"] = list(v2.reason_codes)
        rows.append(row)
    return rows


def build_histograms(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-suite and overall v1 reason-code histograms (fail vs unknown)."""
    def _bucket(selected: list[dict[str, Any]]) -> dict[str, Any]:
        fail_codes: Counter[str] = Counter()
        unknown_codes: Counter[str] = Counter()
        verdicts = Counter()
        for row in selected:
            report = row["v1_report"]
            verdicts[str(report["verdict"]).lower()] += 1
            for check in report["checks"]:
                if check["status"] == "fail":
                    fail_codes[check["code"]] += 1
                elif check["status"] == "unknown":
                    unknown_codes[check["code"]] += 1
        return {
            "n": len(selected),
            "verdicts": dict(sorted(verdicts.items())),
            "fail_reason_histogram": dict(sorted(fail_codes.items())),
            "unknown_reason_histogram": dict(sorted(unknown_codes.items())),
        }

    suites = sorted({row["suite"] for row in rows})
    return {
        "schema": "slm288_reason_histograms/v1",
        "report_schema": MEANINGFUL_V1_REASON_SCHEMA,
        "overall": _bucket(rows),
        "per_suite": {suite: _bucket([r for r in rows if r["suite"] == suite]) for suite in suites},
        "per_set": {
            label: _bucket([r for r in rows if r["set_label"] == label])
            for label in sorted({row["set_label"] for row in rows})
        },
    }


def freeze_packet(
    rows: list[dict[str, Any]],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Frozen, hash-pinned, blinded annotation packet (deterministic)."""
    records = _bundle_records(bundle)
    items: list[dict[str, Any]] = []
    for row in rows:
        record = records.get(str(row["id"]))
        if record is None:
            # Prompt-unavailable rows cannot be labeled honestly.
            continue
        items.append(
            {
                "packet_id": f"pkt_{len(items) + 1:03d}",
                "suite": row["suite"],
                "prompt": record.prompt,
                "prediction": row.get("prediction"),
                "prediction_sha256": row.get("prediction_sha256"),
                "blinded": True,
                "provenance": {
                    "set_label": row["set_label"],
                    "detail_id": row["id"],
                    "replay_bundle": str(REPLAY_BUNDLE),
                    "v1_report": row["v1_report"],
                    "v2_reason_codes": row.get("v2_reason_codes"),
                },
            }
        )
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    rubric_sha = _sha256_text(RUBRIC)
    strata = Counter(
        ("pass" if row["v1_report"]["verdict"] else "fail") for row in rows
    )
    packet = {
        "schema": PACKET_SCHEMA,
        "declared_target_n": DECLARED_PACKET_TARGET,
        "n": len(items),
        "shortfall": (
            f"retained corpus holds {len(items)} prompt-available rows; the "
            f"~{DECLARED_PACKET_TARGET}-output target is unmet, so agreement "
            "statistics on this packet are diagnostic/wiring evidence, not a "
            "powered human-validation result"
            if len(items) < DECLARED_PACKET_TARGET
            else None
        ),
        "strata": dict(sorted(strata.items())),
        "rubric_sha256": rubric_sha,
        "packet_sha256": _sha256_text(canonical),
        "items": items,
    }
    return packet


def export_blind(packet: dict[str, Any]) -> str:
    """Rater-facing JSONL: no checkpoint/set provenance, no verdicts."""
    lines = []
    for item in packet["items"]:
        lines.append(
            json.dumps(
                {
                    "packet_id": item["packet_id"],
                    "suite": item["suite"],
                    "prompt": item["prompt"],
                    "prediction": item["prediction"],
                },
                sort_keys=True,
            )
        )
    return "\n".join(lines) + "\n"


def load_labels(paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """packet_id -> rater_id -> label row."""
    labels: dict[str, dict[str, dict[str, Any]]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema", LABEL_SCHEMA) != LABEL_SCHEMA:
                raise ValueError(f"bad label schema in {path}: {row.get('schema')!r}")
            if row.get("label") not in {"meaningful", "not_meaningful", "unknown"}:
                raise ValueError(f"bad label value in {path}: {row.get('label')!r}")
            labels.setdefault(str(row["packet_id"]), {})[str(row["rater_id"])] = row
    return labels


def _binary(label: str) -> bool | None:
    if label == "meaningful":
        return True
    if label == "not_meaningful":
        return False
    return None  # unknown: excluded from binary agreement, never a failure


def analyze_labels(
    packet: dict[str, Any],
    labels: dict[str, dict[str, dict[str, Any]]],
    *,
    rater_ids: tuple[str, ...] = ("rater_a", "rater_b"),
) -> dict[str, Any]:
    items = sorted(packet["items"], key=lambda item: item["packet_id"])
    kappa_left: list[str] = []
    kappa_right: list[str] = []
    metric_pairs: list[tuple[bool, bool]] = []  # (metric verdict, human label)
    per_reason: dict[str, list[tuple[bool, bool]]] = {}
    adjudicated = 0
    complete = 0
    unknown_labels = 0
    agreed: list[tuple[dict[str, Any], bool]] = []
    for item in items:
        packet_id = item["packet_id"]
        raters = labels.get(packet_id, {})
        try:
            left = raters[rater_ids[0]]
            right = raters[rater_ids[1]]
        except KeyError:
            continue
        complete += 1
        l_bin = _binary(left["label"])
        r_bin = _binary(right["label"])
        if l_bin is None or r_bin is None:
            unknown_labels += 1
            continue
        kappa_left.append(left["label"])
        kappa_right.append(right["label"])
        if l_bin != r_bin:
            adjudicated += 1
            continue  # needs adjudication; excluded from metric PR until then
        agreed.append((item, l_bin))

    def _pr(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
        tp = sum(m and h for m, h in pairs)
        fp = sum(m and not h for m, h in pairs)
        fn = sum(not m and h for m, h in pairs)
        tn = sum(not m and not h for m, h in pairs)
        return {
            "n": len(pairs),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }

    for item, human in agreed:
        report = item["provenance"]["v1_report"]
        verdict = bool(report["verdict"])
        metric_pairs.append((verdict, human))
        fail_codes = [c["code"] for c in report["checks"] if c["status"] == "fail"]
        for code in fail_codes or ["<metric_pass>"]:
            per_reason.setdefault(code, []).append((verdict, human))

    # Predeclared sensitivity sweep over recall floors (no tuning on labels).
    bundle_records = _bundle_records(_load_bundle())
    sweep: dict[str, Any] = {}
    for floor in PREDECLARED_RECALL_FLOORS:
        floor_pairs: list[tuple[bool, bool]] = []
        for item, human in agreed:
            record = bundle_records.get(item["provenance"]["detail_id"])
            report = meaningful_program_v1_report(
                item["prediction"], gold=record, min_component_recall=floor
            )
            floor_pairs.append((bool(report["verdict"]), human))
        sweep[str(floor)] = _pr(floor_pairs)

    kappa = cohen_kappa(kappa_left, kappa_right)
    return {
        "schema": ANALYSIS_SCHEMA,
        "packet_sha256": packet["packet_sha256"],
        "rater_ids": list(rater_ids),
        "coverage": {
            "packet_n": packet["n"],
            "fully_labeled": complete,
            "unknown_labels_excluded": unknown_labels,
            "needs_adjudication": adjudicated,
            "adjudication_rate": adjudicated / complete if complete else None,
        },
        "human_agreement": {
            "cohen_kappa": kappa,
            "n_pairs": len(kappa_left),
            "threshold": 0.6,
            "meets_threshold": (kappa is not None and kappa >= 0.6),
        },
        "metric_vs_labels": _pr(metric_pairs),
        "metric_pr_by_reason": {
            code: _pr(pairs) for code, pairs in sorted(per_reason.items())
        },
        "recall_floor_sweep": {
            "predeclared_floors": list(PREDECLARED_RECALL_FLOORS),
            "results": sweep,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("histograms")
    freeze = sub.add_parser("freeze-packet")
    freeze.add_argument("--packet-out", type=Path, default=PACKET_PATH)
    freeze.add_argument("--rubric-out", type=Path, default=RUBRIC_PATH)
    blind = sub.add_parser("export-blind")
    blind.add_argument("--packet", type=Path, default=PACKET_PATH)
    blind.add_argument("--out", type=Path, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--packet", type=Path, default=PACKET_PATH)
    analyze.add_argument("--labels", type=Path, action="append", required=True)
    analyze.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    bundle = _load_bundle()
    rows = score_bundle(bundle)

    if args.command == "histograms":
        print(json.dumps(build_histograms(rows), indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-packet":
        packet = freeze_packet(rows, bundle)
        args.packet_out.write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.rubric_out.write_text(RUBRIC, encoding="utf-8")
        print(
            json.dumps(
                {
                    "n": packet["n"],
                    "packet_sha256": packet["packet_sha256"],
                    "rubric_sha256": packet["rubric_sha256"],
                    "shortfall": packet["shortfall"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "export-blind":
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        args.out.write_text(export_blind(packet), encoding="utf-8")
        print(f"wrote {args.out} ({packet['n']} blinded items)")
        return 0
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    labels = load_labels(args.labels)
    analysis = analyze_labels(packet, labels)
    rendered = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
