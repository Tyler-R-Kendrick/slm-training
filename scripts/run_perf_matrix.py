#!/usr/bin/env python3
"""Run inference-speed experiment matrix (docs/design/perf-experiment-matrix.md).

P/Q-series rows are decode-only overlays on an existing checkpoint (default: the
committed playground demo). Each row records latency + phase breakdown and
checks quality guardrails against P0 (parse rate / placeholder fidelity).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import load_jsonl
from slm_training.models.decode_stats import DecodeStats, aggregate_stats
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.models.twotower import TwoTowerModel


@dataclass(frozen=True)
class PerfExperiment:
    eid: str
    run_id: str
    description: str
    # Decode overlays
    grammar_incremental_state: bool = True
    grammar_verify_chosen_only: bool = False
    grammar_skip_exact_stream_probe: bool = True
    grammar_copy_probes: bool = True
    grammar_early_exit_pick: bool = True
    grammar_multitoken_accept: bool = False
    grammar_multitoken_max: int = 8
    grammar_canvas_lookahead: int = 0
    grammar_ltr_primary: bool = True
    grammar_ltr_repair: bool = False
    grammar_finalize_validate: bool = False
    use_dynamic_quant: bool = False
    use_compile: bool = False
    generate_max_attempts: int = 1
    grammar_finalize_on_last_attempt_only: bool = False
    parallel_unmask: str = "adaptive"
    # When True, disable P1 incremental state (legacy O(T^2) grammar path).
    legacy_grammar_state: bool = False
    # Disable Q1/Q2 for ablation baselines.
    disable_copy_probes: bool = False
    disable_early_exit: bool = False


def experiments() -> list[PerfExperiment]:
    """P0–P8 + Q1/Q2/Q9 + R1/R2/R4/R5/R9 + playground rows."""
    return [
        PerfExperiment(
            "P0",
            "perf_p0_baseline",
            "Baseline LTR primary (legacy grammar, no P2–Q2)",
            grammar_incremental_state=False,
            legacy_grammar_state=True,
            grammar_skip_exact_stream_probe=False,
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "P1",
            "perf_p1_incremental_grammar",
            "Per-row persistent DFA + decoded-prefix text cache",
            grammar_incremental_state=True,
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "P2",
            "perf_p2_verify_chosen",
            "Verify-chosen-only stream probing + skip exact DFA probes",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_skip_exact_stream_probe=True,
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "P3",
            "perf_p3_multitoken",
            "Multi-token acceptance per denoiser forward",
            grammar_incremental_state=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "P4",
            "perf_p4_lookahead32",
            "Prefix+K=32 mask lookahead canvas truncation",
            grammar_incremental_state=True,
            grammar_canvas_lookahead=32,
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "P5",
            "perf_p5_quant",
            "Dynamic int8 Linear quantization (CPU)",
            grammar_incremental_state=True,
            use_dynamic_quant=True,
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "P6",
            "perf_p6_maskgit",
            "MaskGIT-primary with adaptive parallel unmask",
            grammar_incremental_state=True,
            grammar_ltr_primary=False,
            parallel_unmask="adaptive",
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "P7",
            "perf_p7_playground_budget",
            "Playground-like repair+finalize with attempts=1, finalize-last-only",
            grammar_incremental_state=True,
            grammar_ltr_repair=True,
            grammar_finalize_validate=True,
            generate_max_attempts=1,
            grammar_finalize_on_last_attempt_only=True,
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "P8",
            "perf_p8_combo",
            "P1+P2+P3+lookahead32 (pre-Q recipe)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            grammar_canvas_lookahead=32,
            disable_copy_probes=True,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "Q1",
            "perf_q1_copy_probes",
            "Copy-based O(chunk) DFA admit probes + admit memo (on P1)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=False,
            disable_early_exit=True,
        ),
        PerfExperiment(
            "Q2",
            "perf_q2_early_exit",
            "Whitespace fast-admit + early-exit pick (on P1)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=False,
            disable_copy_probes=True,
        ),
        PerfExperiment(
            "Q9",
            "perf_q9_combo",
            "Shippable recipe: P8 + Q1 + Q2",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            grammar_canvas_lookahead=32,
        ),
        PerfExperiment(
            "R1",
            "perf_r1_skip_exact_admit",
            "Skip dfa_admits when tid already in exact DFA allowed set (on Q9)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            grammar_canvas_lookahead=32,
        ),
        PerfExperiment(
            "R2",
            "perf_r2_synced_fastpath",
            "Skip redundant set_prefix when engine already at prefix (on Q9)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            grammar_canvas_lookahead=32,
        ),
        PerfExperiment(
            "R4",
            "perf_r4_repair_p3p4",
            "Repair path uses multitoken+lookahead (PG + R4)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            grammar_canvas_lookahead=32,
            grammar_ltr_repair=True,
            grammar_finalize_validate=True,
            generate_max_attempts=1,
            grammar_finalize_on_last_attempt_only=True,
        ),
        PerfExperiment(
            "R5",
            "perf_r5_attempt_budget",
            "Wire generate_max_attempts; skip redundant BOS ensure (PG)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            grammar_canvas_lookahead=32,
            grammar_ltr_repair=True,
            grammar_finalize_validate=True,
            generate_max_attempts=1,
            grammar_finalize_on_last_attempt_only=True,
        ),
        PerfExperiment(
            "R9",
            "perf_r9_combo",
            "Shippable recipe: Q9 + R1/R2/R4/R5 (decode + repair)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            grammar_canvas_lookahead=32,
        ),
        PerfExperiment(
            "PG",
            "perf_pg_playground",
            "Playground path with R9 levers (repair+finalize)",
            grammar_incremental_state=True,
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_multitoken_max=8,
            grammar_canvas_lookahead=32,
            grammar_ltr_repair=True,
            grammar_finalize_validate=True,
            generate_max_attempts=1,
            grammar_finalize_on_last_attempt_only=True,
        ),
    ]


def _apply(model: TwoTowerModel, exp: PerfExperiment) -> None:
    cfg = model.config
    cfg.grammar_constrained = True
    cfg.grammar_ltr_primary = bool(exp.grammar_ltr_primary)
    cfg.grammar_ltr_repair = bool(exp.grammar_ltr_repair)
    cfg.grammar_finalize_validate = bool(exp.grammar_finalize_validate)
    cfg.grammar_incremental_state = (
        False if exp.legacy_grammar_state else bool(exp.grammar_incremental_state)
    )
    cfg.grammar_verify_chosen_only = bool(exp.grammar_verify_chosen_only)
    cfg.grammar_skip_exact_stream_probe = bool(exp.grammar_skip_exact_stream_probe)
    cfg.grammar_copy_probes = (
        False if exp.disable_copy_probes else bool(exp.grammar_copy_probes)
    )
    cfg.grammar_early_exit_pick = (
        False if exp.disable_early_exit else bool(exp.grammar_early_exit_pick)
    )
    cfg.grammar_multitoken_accept = bool(exp.grammar_multitoken_accept)
    cfg.grammar_multitoken_max = int(exp.grammar_multitoken_max)
    cfg.grammar_canvas_lookahead = int(exp.grammar_canvas_lookahead)
    cfg.generate_max_attempts = int(exp.generate_max_attempts)
    cfg.grammar_finalize_on_last_attempt_only = bool(
        exp.grammar_finalize_on_last_attempt_only
    )
    cfg.parallel_unmask = str(exp.parallel_unmask)
    if exp.use_dynamic_quant:
        model.apply_dynamic_quant()
    if exp.use_compile:
        from slm_training.runtime.accel import maybe_compile

        model.denoiser = maybe_compile(model.denoiser, enabled=True, mode="default")


def _load_prompts(test_dir: Path, suite: str, limit: int) -> list[tuple[str, str | None]]:
    """Return (prompt, gold_openui) pairs from a test suite jsonl."""
    path = test_dir / suite / "records.jsonl"
    if not path.is_file():
        for alt in (
            test_dir / "records.jsonl",
            Path("fixtures/test_seeds.jsonl"),
            Path("fixtures/train_seeds.jsonl"),
        ):
            if alt.is_file():
                path = alt
                break
    records = load_jsonl(path)[:limit]
    out: list[tuple[str, str | None]] = []
    for r in records:
        out.append((r.prompt, getattr(r, "openui", None)))
    if not out:
        out = [
            ("A hero card with title and subtitle", None),
            ("Login form with email and password", None),
            ("Settings page with a toggle list", None),
        ]
    return out


def _bridge_available() -> bool:
    try:
        from slm_training.dsl import lang_core

        return bool(lang_core.bridge_available())
    except Exception:  # noqa: BLE001
        return False


def _quality_pipeline_ok() -> bool:
    """True when validate() accepts a known-good OpenUI snippet (bridge/Lark healthy)."""
    try:
        from slm_training.dsl.lang_core import validate

        validate('root = Card(":t.x")\n')
        return True
    except Exception:  # noqa: BLE001
        return False


def _quality(pred: str, gold: str | None) -> dict[str, float]:
    """Prefer meaningful-program check (same as eval_runner); track raw syntax too."""
    from slm_training.dsl.placeholders import extract_placeholders
    from slm_training.harnesses.model_build.eval_runner import _is_meaningful_program

    gold_rec = None
    if gold:
        try:
            from slm_training.dsl.schema import ExampleRecord

            gold_rec = ExampleRecord(
                id="perf",
                prompt="p",
                openui=gold,
                placeholders=list(extract_placeholders(gold)),
            )
        except Exception:  # noqa: BLE001
            gold_rec = None
    ok, _err, serialized = _is_meaningful_program(pred, gold=gold_rec)
    scored = serialized or pred
    raw_ok = 0.0
    try:
        from slm_training.dsl.lang_core import validate

        validate(pred)
        raw_ok = 1.0
    except Exception:  # noqa: BLE001
        raw_ok = 0.0
    fidelity = 1.0
    if gold:
        pred_set = set(extract_placeholders(scored))
        gold_set = set(extract_placeholders(gold))
        fidelity = (len(pred_set & gold_set) / len(gold_set)) if gold_set else 1.0
    return {
        "parse_ok": 1.0 if ok else 0.0,
        "raw_syntax_ok": raw_ok,
        "placeholder_fidelity": fidelity,
    }


def run_one(
    exp: PerfExperiment,
    *,
    checkpoint: Path,
    prompts: list[tuple[str, str | None]],
    device: str,
    warmup: int,
    out_dir: Path,
) -> dict[str, Any]:
    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    model.eval()
    _apply(model, exp)

    for i in range(max(0, warmup)):
        model.generate(prompts[i % len(prompts)][0])

    rows: list[DecodeStats] = []
    parse_sum = 0.0
    raw_sum = 0.0
    fid_sum = 0.0
    texts: list[str] = []
    t0 = time.perf_counter()
    for prompt, gold in prompts:
        text, stats = model.generate_with_stats(prompt)
        rows.append(stats)
        texts.append(text)
        q = _quality(text, gold)
        parse_sum += q["parse_ok"]
        raw_sum += q.get("raw_syntax_ok", q["parse_ok"])
        fid_sum += q["placeholder_fidelity"]
    wall = time.perf_counter() - t0
    n = max(1, len(prompts))
    summary = aggregate_stats(rows)
    tokens = sum(int(r.tokens_emitted) for r in rows)
    result = {
        "id": exp.eid,
        "run_id": exp.run_id,
        "description": exp.description,
        "n": n,
        "wall_sec": round(wall, 4),
        "latency_ms_mean": round((wall / n) * 1000.0, 2),
        "latency_ms_p50": summary.get("total_ms_p50"),
        "latency_ms_p95": summary.get("total_ms_p95"),
        "tokens_emitted": tokens,
        "tokens_per_sec": round(tokens / wall, 3) if wall > 0 else None,
        "parse_rate": round(parse_sum / n, 4),
        "raw_syntax_rate": round(raw_sum / n, 4),
        "placeholder_fidelity": round(fid_sum / n, 4),
        "phase_summary": summary,
        "flags": {
            "grammar_incremental_state": bool(model.config.grammar_incremental_state),
            "grammar_verify_chosen_only": bool(model.config.grammar_verify_chosen_only),
            "grammar_copy_probes": bool(getattr(model.config, "grammar_copy_probes", True)),
            "grammar_early_exit_pick": bool(
                getattr(model.config, "grammar_early_exit_pick", True)
            ),
            "grammar_multitoken_accept": bool(model.config.grammar_multitoken_accept),
            "grammar_canvas_lookahead": int(model.config.grammar_canvas_lookahead),
            "grammar_ltr_primary": bool(model.config.grammar_ltr_primary),
            "grammar_ltr_repair": bool(model.config.grammar_ltr_repair),
            "use_dynamic_quant": bool(model.config.use_dynamic_quant),
        },
        "sample_output": texts[0] if texts else "",
    }
    run_dir = out_dir / exp.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "matrix_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _guardrails(baseline: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Quality must not regress vs P0 beyond small tolerance."""
    parse_floor = float(baseline.get("parse_rate") or 0.0) - 0.05
    fid_floor = float(baseline.get("placeholder_fidelity") or 0.0) - 0.05
    parse_ok = float(row.get("parse_rate") or 0.0) >= parse_floor
    fid_ok = float(row.get("placeholder_fidelity") or 0.0) >= fid_floor
    speedup = None
    base_lat = float(baseline.get("latency_ms_mean") or 0.0)
    row_lat = float(row.get("latency_ms_mean") or 0.0)
    if base_lat > 0 and row_lat > 0:
        speedup = round(base_lat / row_lat, 3)
    return {
        "pass": bool(parse_ok and fid_ok),
        "parse_ok": parse_ok,
        "fidelity_ok": fid_ok,
        "speedup_vs_p0": speedup,
        "parse_floor": round(parse_floor, 4),
        "fidelity_floor": round(fid_floor, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PLAYGROUND_DEMO_CHECKPOINT,
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("outputs/test_data/v1"),
    )
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated experiment ids (e.g. P0,P8,Q9).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/perf_matrix"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/perf-matrix-results.json"),
    )
    args = parser.parse_args(argv)

    bridge = _bridge_available()
    pipeline_ok = _quality_pipeline_ok()
    prompts = _load_prompts(args.test_dir, args.suite, args.limit)
    wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    rows_def = experiments()
    if wanted:
        rows_def = [e for e in rows_def if e.eid in wanted]

    # Vacuous if the quality pipeline itself is broken (missing bridge deps, etc.).
    vacuous = (not pipeline_ok) and bridge
    if not pipeline_ok and not bridge:
        # Offline Lark-only hosts: try a Lark validate path once more.
        vacuous = not pipeline_ok

    results: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    for exp in rows_def:
        print(f"==> {exp.eid} {exp.run_id}")
        result = run_one(
            exp,
            checkpoint=args.checkpoint,
            prompts=prompts,
            device=args.device,
            warmup=args.warmup,
            out_dir=args.out_dir,
        )
        if exp.eid == "P0":
            baseline = result
            result["guardrails"] = {
                "pass": not vacuous,
                "speedup_vs_p0": 1.0,
                "note": (
                    "vacuous_gate: quality pipeline broken"
                    if vacuous
                    else "baseline"
                ),
                "vacuous": vacuous,
            }
        elif baseline is not None:
            result["guardrails"] = _guardrails(baseline, result)
            if vacuous:
                result["guardrails"]["vacuous_baseline"] = True
                result["guardrails"]["pass"] = False
        else:
            result["guardrails"] = {"pass": True, "note": "no P0 in this run"}
        results.append(result)
        (args.out_dir / exp.run_id / "matrix_result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    board = {
        "checkpoint": str(args.checkpoint),
        "suite": args.suite,
        "limit": args.limit,
        "device": args.device,
        "bridge_available": bridge,
        "quality_pipeline_ok": pipeline_ok,
        "vacuous_guardrails": vacuous,
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    board_path = args.out_dir / "scoreboard.json"
    board_path.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(board, indent=2))
    print(f"wrote {board_path} and {args.docs_out}")
    if vacuous:
        print(
            "ERROR: vacuous guardrails — quality pipeline cannot validate known-good "
            "OpenUI (install tools/openui_bridge deps or check Lark fallback)."
        )
        return 2
    failed = [
        r["id"]
        for r in results
        if r.get("guardrails") and not r["guardrails"].get("pass", True)
    ]
    if failed:
        print(f"guardrail failures: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
