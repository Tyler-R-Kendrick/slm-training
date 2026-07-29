#!/usr/bin/env python3
"""Run inference-speed experiment matrix (docs/design/perf-experiment-matrix.md).

P/Q-series rows are decode-only overlays on an existing checkpoint (default: the
committed playground demo). Each row records latency + phase breakdown and
checks quality guardrails against P0 (parse rate / placeholder fidelity).
C-series rows compare compiler-drafted decode against the same-run C0 control.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import load_jsonl
from slm_training.models.decode_stats import DecodeStats, aggregate_stats
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.models.twotower import TwoTowerModel
from slm_training.levers import DEFAULT_EVAL_DATA_DIR
from slm_training.versioning import build_version_stamp


def _timed_ms(fn, repeats: int) -> tuple[float, Any]:
    """Warm once, then return median milliseconds and the final value."""
    value = fn()
    samples: list[float] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        value = fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), value


def _completion_direct_bench(repeats: int) -> dict[str, Any]:
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.dsl.pack import (
        _openui_completion_domain,
        _openui_completion_domain_reference,
    )
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
    from slm_training.models.grammar import make_grammar_state

    tokenizer = DSLNativeTokenizer.build()
    prefix = tuple(
        [tokenizer.bos_id]
        + tokenizer.encode("root = Card([b1,", add_special=False)
    )
    packed_request = CompletionDomainRequestV1(
        prefix_ids=prefix,
        tokenizer=tokenizer,
        slot_contract=(":slot_0", ":slot_1"),
        remaining_tokens=32,
        state=make_grammar_state(),
    )
    reference_request = CompletionDomainRequestV1(
        prefix_ids=prefix,
        tokenizer=tokenizer,
        slot_contract=(":slot_0", ":slot_1"),
        remaining_tokens=32,
        state=make_grammar_state(),
    )
    reference_ms, reference = _timed_ms(
        lambda: _openui_completion_domain_reference(reference_request), repeats
    )
    _openui_completion_domain(packed_request)

    def packed_graph_query():
        session = packed_request.state.completion_session
        if session is not None:
            session._results.clear()  # benchmark the graph/DP, not result replay
        return _openui_completion_domain(packed_request)

    packed_ms, packed = _timed_ms(packed_graph_query, repeats)
    speedup = reference_ms / packed_ms if packed_ms else None
    return {
        "n": 1,
        "prefix": "root = Card([b1,",
        "reference_ms": round(reference_ms, 4),
        "packed_ms": round(packed_ms, 4),
        "speedup": round(speedup, 3) if speedup is not None else None,
        "correct": packed == reference and len(packed.candidates) == 12,
        "gate": "warm graph/DP speedup >= 10 and exact 12-path parity",
        "pass": bool(packed == reference and len(packed.candidates) == 12 and speedup >= 10),
    }


def _completion_corpus_bench(repeats: int) -> dict[str, Any]:
    from slm_training.dsl.grammar.fastpath import compiler_draft
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.dsl.pack import (
        _openui_completion_domain,
        _openui_completion_domain_reference,
    )
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
    from slm_training.models.grammar import make_grammar_state

    tokenizer = DSLNativeTokenizer.build()
    texts = (
        "",
        "root",
        "root =",
        "root = Card([",
        "root = Card([b1",
        "root = Card([b1,",
        'root = TextContent(":slot_0")',
    )
    def requests(selected=texts):
        return [
            CompletionDomainRequestV1(
                prefix_ids=tuple(
                    [tokenizer.bos_id] + tokenizer.encode(text, add_special=False)
                ),
                tokenizer=tokenizer,
                slot_contract=(":slot_0", ":slot_1"),
                remaining_tokens=32,
                state=make_grammar_state(),
            )
            for text in selected
        ]

    packed_requests = requests()
    reference_requests = requests()
    reference_ms, reference = _timed_ms(
        lambda: [
            _openui_completion_domain_reference(row) for row in reference_requests
        ],
        repeats,
    )
    packed_ms, packed = _timed_ms(
        lambda: [_openui_completion_domain(row) for row in packed_requests],
        repeats,
    )
    # Separate cold guardrail: fresh row state and no stateless forest cache.
    # Other immutable tokenizer/schema tables stay warm for both arms.
    compiler_draft._official_schema()

    def cold(provider, selected=texts):
        compiler_draft._STATELESS_FOREST_CACHE.clear()
        rows = requests(selected)
        started = time.perf_counter()
        result = [provider(row) for row in rows]
        return (time.perf_counter() - started) * 1000.0, result

    # Discard one cold-row warmup per arm so immutable tokenizer/schema memo
    # population is not charged to whichever implementation happens to run
    # first, then alternate measured order.
    cold_rows = []
    cold_ratios: list[float] = []
    cold_correct = True
    for text in texts[:3]:
        cold(_openui_completion_domain_reference, (text,))
        cold(_openui_completion_domain, (text,))
        cold_reference_samples: list[float] = []
        cold_packed_samples: list[float] = []
        cold_reference = cold_packed = []
        for index in range(max(1, min(int(repeats), 3))):
            order = (
                (
                    ("reference", _openui_completion_domain_reference),
                    ("packed", _openui_completion_domain),
                )
                if index % 2 == 0
                else (
                    ("packed", _openui_completion_domain),
                    ("reference", _openui_completion_domain_reference),
                )
            )
            for name, provider in order:
                elapsed, result = cold(provider, (text,))
                if name == "reference":
                    cold_reference_samples.append(elapsed)
                    cold_reference = result
                else:
                    cold_packed_samples.append(elapsed)
                    cold_packed = result
        reference_cold = statistics.median(cold_reference_samples)
        packed_cold = statistics.median(cold_packed_samples)
        cold_correct = cold_correct and cold_packed == cold_reference
        cold_ratios.append(packed_cold / reference_cold)
        cold_rows.append(
            {
                "prefix": text,
                "reference_ms": round(reference_cold, 4),
                "packed_ms": round(packed_cold, 4),
                "packed_over_reference": round(
                    packed_cold / reference_cold, 3
                ),
            }
        )
    ratio = packed_ms / reference_ms if reference_ms else None
    cold_ratio = max(cold_ratios)
    return {
        "n": len(packed_requests),
        "reference_ms": round(reference_ms, 4),
        "packed_ms": round(packed_ms, 4),
        "packed_over_reference": round(ratio, 3) if ratio is not None else None,
        "cold_simple_prefixes": cold_rows,
        "cold_packed_over_reference_max": round(cold_ratio, 3),
        "correct": packed == reference and cold_correct,
        "gate": "exact parity and cold packed time <= 1.15x reference",
        "pass": bool(
            packed == reference
            and cold_correct
            and cold_ratio <= 1.15
        ),
    }


def _legacy_choice_allowed(state: Any, remaining: int) -> set[int]:
    """Pre-kernel greedy feasibility loop retained only as benchmark control."""
    allowed: set[int] = set()
    for token_id in state._candidate_ids():
        probe = state.clone()
        if not probe.advance_id(token_id):
            continue
        if token_id == state.tokenizer.eos_id:
            completion = 0
        else:
            completion = 1025
            for count in range(1, min(1024, remaining - 1) + 1):
                next_id = probe._completion_id()
                if not probe.advance_id(next_id):
                    break
                if next_id == probe.tokenizer.eos_id:
                    completion = count
                    break
        if completion <= remaining - 1:
            allowed.add(token_id)
    return allowed


def _completion_choice_bench(repeats: int) -> dict[str, Any]:
    from slm_training.models.choice_tokenizer import (
        ChoiceDecodeState,
        ChoiceTokenizer,
    )

    tokenizer = ChoiceTokenizer.build()
    state = ChoiceDecodeState(tokenizer, slot_count=2)
    remaining = 12
    reference_ms, reference = _timed_ms(
        lambda: _legacy_choice_allowed(state, remaining), repeats
    )
    warm_cache_ms, _ = _timed_ms(lambda: state.allowed_ids(remaining), repeats)

    def packed_query():
        tokenizer.allowed_cache.clear()
        return state.allowed_ids(remaining)

    packed_ms, packed = _timed_ms(packed_query, repeats)
    speedup = reference_ms / packed_ms if packed_ms else None
    return {
        "n": 1,
        "remaining_tokens": remaining,
        "reference_ms": round(reference_ms, 4),
        "packed_ms": round(packed_ms, 4),
        "warm_allowed_cache_ms": round(warm_cache_ms, 4),
        "speedup": round(speedup, 3) if speedup is not None else None,
        "correct": packed == reference,
        "gate": "exact parity and speedup >= 3",
        "pass": bool(packed == reference and speedup >= 3),
    }


def _completion_solver_bench(repeats: int) -> dict[str, Any]:
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        build_completion_forest,
    )
    from slm_training.dsl.solver.openui_support import OpenUIForestExpander
    from slm_training.dsl.solver.state import SolverBounds
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tokenizer = DSLNativeTokenizer.build()
    prefix = tuple(
        [tokenizer.bos_id]
        + tokenizer.encode("root = Card([", add_special=False)
    )
    bounds = SolverBounds(
        max_tokens=4000,
        max_nodes=24,
        max_depth=12,
        max_backtracks=200,
        max_verifier_calls=24,
    )

    packed_expander = OpenUIForestExpander(
        tokenizer,
        prefix,
        pack_id="openui",
        constraint_version="bench",
        bounds=bounds,
    )
    packed_root = packed_expander.root_state()

    def packed():
        return [
            packed_expander.successor(
                packed_root, packed_root.holes[0].hole_id, value
            ).status.value
            for value in packed_root.holes[0].values
        ]

    root = OpenUIForestExpander(
        tokenizer,
        prefix,
        pack_id="openui",
        constraint_version="bench",
        bounds=bounds,
    ).root_state()

    def reference():
        rows = []
        for value in root.holes[0].values:
            token_ids = tuple(int(t) for t in value.payload.get("token_ids", ()))
            forest = build_completion_forest(
                tokenizer, list(prefix + token_ids), max_path_tokens=8
            )
            rows.append("continue" if forest.paths else forest.coverage)
        return rows

    reference_ms, reference_rows = _timed_ms(reference, repeats)
    packed_ms, packed_rows = _timed_ms(packed, repeats)
    speedup = reference_ms / packed_ms if packed_ms else None
    return {
        "n": len(reference_rows),
        "reference_ms": round(reference_ms, 4),
        "packed_ms": round(packed_ms, 4),
        "speedup": round(speedup, 3) if speedup is not None else None,
        "reference_nonempty": bool(reference_rows),
        "packed_nonempty": bool(packed_rows),
        "cached_successors": len(packed_expander._successors),
        "gate": "both paths complete and every identical successor expands once",
        "pass": bool(
            reference_rows
            and packed_rows
            and len(packed_expander._successors) == len(packed_root.holes[0].values)
        ),
    }


def _completion_decode_bench(repeats: int, device: str) -> dict[str, Any]:
    from dataclasses import replace

    from slm_training.data.contract import canonicalize_example_template_markers
    from slm_training.dsl.grammar.fastpath import compiler_draft
    from slm_training.dsl.pack import (
        _openui_completion_domain,
        _openui_completion_domain_reference,
        get_pack,
        register_pack,
    )
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.models.decode_stats import collect_decode_stats
    from slm_training.models.twotower import TwoTowerConfig

    record = canonicalize_example_template_markers(
        ExampleRecord(
            id="completion-kernel-bench",
            prompt="card",
            openui='root = Card([title])\ntitle = TextContent(":hero.title")\n',
            placeholders=[":hero.title"],
            split="train",
            source="fixture",
        )
    )
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="lexer",
            compiler_decode_mode="tree",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=32,
            max_target_len=32,
            grammar_ltr_max_tokens=32,
            gen_steps=1,
            seed=0,
        ),
        device=device,
    )
    model.eval()
    ctx, ctx_pad = model._encode_context(["card"])
    pack = get_pack()
    authority = pack.grammar_capability_authority
    assert authority is not None

    def run(provider):
        register_pack(
            replace(
                pack,
                grammar_capability_authority=replace(
                    authority, completion_domain=provider
                ),
            )
        )
        try:
            with collect_decode_stats() as stats:
                ids = model._compiler_ltr_decode_one(
                    ctx, ctx_pad, 16, mode="tree", slot_contract=None
                )
            return tuple(int(v) for v in ids.tolist()), stats
        finally:
            register_pack(pack)

    samples: dict[str, list[tuple[float, tuple[int, ...], DecodeStats]]] = {
        "reference": [],
        "packed": [],
    }
    count = max(1, min(int(repeats), 3))
    for index in range(count):
        order = (
            (
                ("reference", _openui_completion_domain_reference),
                ("packed", _openui_completion_domain),
            )
            if index % 2 == 0
            else (
                ("packed", _openui_completion_domain),
                ("reference", _openui_completion_domain_reference),
            )
        )
        for name, provider in order:
            compiler_draft._STATELESS_FOREST_CACHE.clear()
            started = time.perf_counter()
            ids, stats = run(provider)
            samples[name].append(
                ((time.perf_counter() - started) * 1000.0, ids, stats)
            )
    reference_ms = statistics.median(row[0] for row in samples["reference"])
    packed_ms = statistics.median(row[0] for row in samples["packed"])
    reference_compiler_ms = statistics.median(
        row[2].compiler_ms for row in samples["reference"]
    )
    packed_compiler_ms = statistics.median(
        row[2].compiler_ms for row in samples["packed"]
    )
    reference_ids = samples["reference"][-1][1]
    packed_ids = samples["packed"][-1][1]
    packed_stats = samples["packed"][-1][2]
    compiler_speedup = (
        reference_compiler_ms / packed_compiler_ms
        if packed_compiler_ms
        else None
    )
    return {
        "n": 1,
        "device": device,
        "reference_wall_ms": round(reference_ms, 4),
        "packed_wall_ms": round(packed_ms, 4),
        "reference_compiler_ms": round(reference_compiler_ms, 4),
        "packed_compiler_ms": round(packed_compiler_ms, 4),
        "compiler_speedup": (
            round(compiler_speedup, 3) if compiler_speedup is not None else None
        ),
        "correct": reference_ids == packed_ids,
        "packed_forwards": packed_stats.forwards_count,
        "packed_completion_session_starts": packed_stats.completion_session_starts,
        "packed_completion_full_prefix_lex_bytes": (
            packed_stats.completion_full_prefix_lex_bytes
        ),
        "packed_completion_candidate_engine_allocations": (
            packed_stats.completion_candidate_engine_allocations
        ),
        "gate": "identical output and compiler_ms speedup >= 5",
        "pass": bool(reference_ids == packed_ids and compiler_speedup >= 5),
    }


_COMPLETION_WORKLOADS = {
    "direct": _completion_direct_bench,
    "corpus": _completion_corpus_bench,
    "choice": _completion_choice_bench,
    "solver": _completion_solver_bench,
}


def _run_completion_kernel_mode(args: argparse.Namespace) -> int:
    names = [
        name.strip().lower()
        for name in args.completion_kernel_workload.split(",")
        if name.strip()
    ]
    unknown = set(names) - (set(_COMPLETION_WORKLOADS) | {"decode"})
    if unknown:
        raise ValueError(f"unknown completion-kernel workloads: {sorted(unknown)}")
    results = {}
    for name in names:
        print(f"==> completion-kernel {name}")
        result = (
            _completion_decode_bench(args.completion_repeats, args.device)
            if name == "decode"
            else _COMPLETION_WORKLOADS[name](args.completion_repeats)
        )
        result["repeats"] = args.completion_repeats
        result["measured_at"] = dt.datetime.now(dt.UTC).isoformat()
        result["argv"] = list(sys.argv[1:])
        result["host"] = {
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
        result["version_stamp"] = build_version_stamp(
            "matrix.perf",
            "harness.model_build.eval",
            "model.twotower",
            "dsl.operators.registry",
        )
        results[name] = result
    prior: dict[str, Any] = {}
    if args.docs_out.is_file():
        try:
            loaded = json.loads(args.docs_out.read_text(encoding="utf-8"))
            if loaded.get("schema_version") == "completion-kernel-perf/v1":
                prior = loaded
        except (OSError, ValueError):
            pass
    workloads = {**prior.get("workloads", {}), **results}
    board = {
        **prior,
        "schema_version": "completion-kernel-perf/v1",
        "recipe": {
            "device": args.device,
            "execution": "independently capped workload shards",
            "honesty_mode": "fixture_perf_not_ship",
            "hard_cap_minutes": 3,
        },
        "selected_pass": all(row["pass"] for row in results.values()),
        "overall_pass": all(row["pass"] for row in workloads.values()),
        "workloads": workloads,
        "version_stamp": build_version_stamp(
            "matrix.perf",
            "harness.model_build.eval",
            "model.twotower",
            "dsl.operators.registry",
        ),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(board, indent=2))
    return 0 if board["overall_pass"] else 1


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
    compiler_decode_mode: str = "off"
    compiler_prefill_max_states: int = 0
    compiler_prefill_token_budget: int = 0
    grammar_equivalence_cache: bool = False
    grammar_active_symbol_bitsets: bool = False
    grammar_completion_bounds: bool = False
    compact_active_canvas: bool = True
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
        PerfExperiment(
            "C0",
            "perf_c0_r9_control",
            "Current R9 full-vocabulary control",
            grammar_verify_chosen_only=True,
            grammar_multitoken_accept=True,
            grammar_canvas_lookahead=32,
        ),
        PerfExperiment(
            "C1",
            "perf_c1_forced",
            "Compiler maximal forced spans; full projection at branches",
            compiler_decode_mode="forced",
        ),
        PerfExperiment(
            "C2",
            "perf_c2_restricted",
            "Restricted semantic action and symbol projection",
            compiler_decode_mode="restricted",
        ),
        PerfExperiment(
            "C3",
            "perf_c3_tree",
            "Packed completion-trie verification",
            compiler_decode_mode="tree",
        ),
        PerfExperiment(
            "C4",
            "perf_c4_hierarchy",
            "Full compiler hierarchy with prefix-seeded V7 fallback",
            compiler_decode_mode="tree",
            grammar_ltr_repair=True,
            generate_max_attempts=1,
        ),
        PerfExperiment(
            "C5",
            "perf_c5_terminal_equivalence",
            "Cached terminal-equivalence token classes",
            grammar_equivalence_cache=True,
        ),
        PerfExperiment(
            "C6",
            "perf_c6_active_symbol_bitsets",
            "Static grammar classes intersected with request-active symbols",
            grammar_equivalence_cache=True,
            grammar_active_symbol_bitsets=True,
        ),
        PerfExperiment(
            "C7",
            "perf_c7_completion_compact",
            "Conservative completion bounds with compact active canvases",
            grammar_completion_bounds=True,
            compact_active_canvas=True,
            grammar_ltr_primary=False,
        ),
        PerfExperiment(
            "C8",
            "perf_c8_constraint_system",
            "Combined grammar cache, active symbols, bounds, and compact canvas",
            grammar_equivalence_cache=True,
            grammar_active_symbol_bitsets=True,
            grammar_completion_bounds=True,
            compact_active_canvas=True,
            grammar_ltr_primary=False,
        ),
        PerfExperiment(
            "C9",
            "perf_c9_serial_compiler_prefill",
            "Compiler tree with one ambiguous state per neural prefill",
            compiler_decode_mode="tree",
            compiler_prefill_max_states=1,
        ),
        PerfExperiment(
            "C10",
            "perf_c10_auto_compiler_prefill",
            "Compiler tree with bounded device-aware speculative prefills",
            compiler_decode_mode="tree",
            compiler_prefill_max_states=0,
            compiler_prefill_token_budget=0,
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
    cfg.compiler_decode_mode = str(exp.compiler_decode_mode)
    cfg.compiler_prefill_max_states = int(exp.compiler_prefill_max_states)
    cfg.compiler_prefill_token_budget = int(exp.compiler_prefill_token_budget)
    cfg.grammar_equivalence_cache = bool(exp.grammar_equivalence_cache)
    cfg.grammar_active_symbol_bitsets = bool(exp.grammar_active_symbol_bitsets)
    cfg.grammar_completion_bounds = bool(exp.grammar_completion_bounds)
    cfg.compact_active_canvas = bool(exp.compact_active_canvas)
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
            Path("src/slm_training/resources/test_seeds.jsonl"),
            Path("src/slm_training/resources/train_seeds.jsonl"),
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
    max_len: int | None = None,
) -> dict[str, Any]:
    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    model.eval()
    _apply(model, exp)

    for i in range(max(0, warmup)):
        model.generate(prompts[i % len(prompts)][0], max_len=max_len)

    rows: list[DecodeStats] = []
    parse_sum = 0.0
    raw_sum = 0.0
    fid_sum = 0.0
    texts: list[str] = []
    t0 = time.perf_counter()
    for prompt, gold in prompts:
        text, stats = model.generate_with_stats(prompt, max_len=max_len)
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
        "max_len": max_len,
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
            "compiler_decode_mode": str(model.config.compiler_decode_mode),
            "compiler_prefill_max_states": int(
                model.config.compiler_prefill_max_states
            ),
            "compiler_prefill_token_budget": int(
                model.config.compiler_prefill_token_budget
            ),
        },
        "sample_output": texts[0] if texts else "",
    }
    run_dir = out_dir / exp.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "matrix_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    try:
        from slm_training.autoresearch.run_insights import load_run_insights

        load_run_insights(run_dir, run_id=exp.run_id)
    except Exception as exc:  # noqa: BLE001 - analysis must never fail the matrix
        warnings.warn(f"run insight analysis failed: {exc}", stacklevel=2)
    return result


def _guardrails(baseline: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Quality must not regress vs P0 beyond small tolerance."""
    baseline_parse = float(baseline.get("parse_rate") or 0.0)
    baseline_fid = float(baseline.get("placeholder_fidelity") or 0.0)
    if baseline_parse <= 0.0 or baseline_fid <= 0.0:
        return {
            "pass": False,
            "parse_ok": False,
            "fidelity_ok": False,
            "speedup_vs_p0": None,
            "parse_floor": None,
            "fidelity_floor": None,
            "note": "invalid P0 quality anchor; candidate is not promotable",
        }
    parse_floor = baseline_parse - 0.05
    fid_floor = baseline_fid - 0.05
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
        default=DEFAULT_EVAL_DATA_DIR,
    )
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=None)
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
    parser.add_argument(
        "--completion-kernel-workload",
        default="",
        help=(
            "Run the packed completion-kernel benchmark mode for a comma-separated "
            "subset of direct,corpus,choice,solver,decode."
        ),
    )
    parser.add_argument(
        "--completion-repeats",
        type=int,
        default=3,
        help="Measured repetitions per completion-kernel workload after one warmup.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print selected experiment definitions without running them.",
    )
    args = parser.parse_args(argv)
    if args.completion_kernel_workload:
        if args.docs_out == Path("docs/design/perf-matrix-results.json"):
            args.docs_out = Path(
                "docs/design/completion-kernel-perf-results.json"
            )
        return _run_completion_kernel_mode(args)
    wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    rows_def = experiments()
    if wanted:
        rows_def = [e for e in rows_def if e.eid in wanted]
    if args.list:
        print(
            json.dumps(
                [
                    {"id": row.eid, "run_id": row.run_id, "description": row.description}
                    for row in rows_def
                ],
                indent=2,
            )
        )
        return 0

    bridge = _bridge_available()
    pipeline_ok = _quality_pipeline_ok()
    prompts = _load_prompts(args.test_dir, args.suite, args.limit)

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
            max_len=args.max_len,
        )
        if exp.eid in {"P0", "C0"}:
            baseline = result
            quality_anchor = (
                float(result.get("parse_rate") or 0.0) > 0.0
                and float(result.get("placeholder_fidelity") or 0.0) > 0.0
            )
            result["guardrails"] = {
                "pass": bool(not vacuous and quality_anchor),
                "speedup_vs_p0": 1.0,
                "note": (
                    "vacuous_gate: quality pipeline broken"
                    if vacuous
                    else (
                        "baseline"
                        if quality_anchor
                        else "invalid zero-quality baseline; not promotable"
                    )
                ),
                "vacuous": vacuous,
            }
        elif baseline is not None:
            result["guardrails"] = _guardrails(baseline, result)
            if vacuous:
                result["guardrails"]["vacuous_baseline"] = True
                result["guardrails"]["pass"] = False
        else:
            # A speed row without the same-run P0 control is not quality-gated.
            # Mark it incomplete instead of silently making an unanchored row
            # look promotable in a focused run.
            result["guardrails"] = {
                "pass": False,
                "note": "missing P0 control; rerun with --only P0,<candidate>",
            }
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
    by_id = {row["id"]: row for row in results}
    c0 = by_id.get("C0")
    c4 = by_id.get("C4")
    if c0 and c4:
        base_phase = c0.get("phase_summary") or {}
        candidate_phase = c4.get("phase_summary") or {}
        lower_forwards = float(candidate_phase.get("forwards_count_mean") or 0.0) < float(
            base_phase.get("forwards_count_mean") or 0.0
        )
        lower_p50 = float(c4.get("latency_ms_p50") or 0.0) < float(
            c0.get("latency_ms_p50") or 0.0
        )
        gate_pass = bool((c4.get("guardrails") or {}).get("pass"))
        board["compiler_default_recommendation"] = {
            "mode": "tree" if gate_pass and lower_forwards and lower_p50 else "off",
            "quality_guardrails_pass": gate_pass,
            "fewer_forwards": lower_forwards,
            "lower_p50": lower_p50,
            "note": (
                "promote only after a non-vacuous quality baseline"
                if not gate_pass
                else "same-run C0/C4 decision"
            ),
        }
    from slm_training.evals.agentv import publish_agentv_evaluation

    board["agentv"] = publish_agentv_evaluation(
        args.out_dir,
        name="compiler-decode-c-series",
        claim="compiler_decode_perf_and_quality_guardrails",
        cases=[
            {
                "id": row["id"],
                "criteria": (
                    "Preserve parse and placeholder fidelity within five absolute "
                    "points of the same-run control before speed claims."
                ),
                "pass": bool((row.get("guardrails") or {}).get("pass")),
                "failures": []
                if (row.get("guardrails") or {}).get("pass")
                else [str((row.get("guardrails") or {}).get("note") or "guardrail_failed")],
                "result": row,
                "metadata": {"suite": args.suite, "device": args.device},
            }
            for row in results
            if row["id"].startswith("C")
        ],
    )
    board["version_stamp"] = build_version_stamp(
        "matrix.perf",
        "harness.model_build.eval",
        "model.twotower",
    )
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
            "OpenUI (install src/apps/openui_bridge deps or check Lark fallback)."
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
