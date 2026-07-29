#!/usr/bin/env python3
"""Run inference-speed experiment matrix (docs/design/perf-experiment-matrix.md).

P/Q-series rows are decode-only overlays on an existing checkpoint (default: the
committed playground demo). Each row records latency + phase breakdown and
checks quality guardrails against P0 (parse rate / placeholder fidelity).
C-series rows compare compiler-drafted decode against the same-run C0 control.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import enum
import hashlib
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
from slm_training.levers import DEFAULT_EVAL_DATA_DIR, MAX_RUN_MINUTES
from slm_training.versioning import build_version_stamp

_COMPLETION_REQUIRED_WORKLOADS = ("direct", "corpus", "choice", "solver", "decode")
_COMPLETION_STAMP_COMPONENTS = (
    "matrix.perf",
    "harness.model_build.eval",
    "harness.solver_bench",
    "model.twotower",
    "dsl.operators.registry",
    "config.levers",
)
_COMPLETION_MIN_SAMPLE_NS = 10_000_000
_COMPLETION_MAX_RELATIVE_MAD = 0.15


def _canonical_json_value(value: Any) -> Any:
    """Return a deterministic JSON value for parity digests."""
    if dataclasses.is_dataclass(value):
        return _canonical_json_value(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return _canonical_json_value(value.value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        rows = [_canonical_json_value(item) for item in value]
        return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _result_digest(value: Any) -> str:
    payload = json.dumps(
        _canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _median_mad(samples: list[float]) -> tuple[float, float]:
    median = float(statistics.median(samples))
    mad = float(statistics.median(abs(value - median) for value in samples))
    return median, mad


def _timed_bundle(fn, bundle_size: int) -> tuple[int, list[str], Any]:
    values: list[Any] = []
    started = time.perf_counter_ns()
    for _ in range(bundle_size):
        values.append(fn())
    elapsed = time.perf_counter_ns() - started
    return elapsed, [_result_digest(value) for value in values], values[-1]


def _paired_measure(
    reference_fn,
    packed_fn,
    repeats: int,
    *,
    min_sample_ns: int = _COMPLETION_MIN_SAMPLE_NS,
) -> dict[str, Any]:
    """Measure alternating paired arms and verify every invocation."""
    reference_pilot = reference_fn()
    packed_pilot = packed_fn()
    if _result_digest(reference_pilot) != _result_digest(packed_pilot):
        raise AssertionError("completion benchmark pilot parity mismatch")

    bundle_size = 1
    while True:
        reference_ns, reference_digests, reference_value = _timed_bundle(
            reference_fn, bundle_size
        )
        packed_ns, packed_digests, packed_value = _timed_bundle(
            packed_fn, bundle_size
        )
        if reference_digests != packed_digests:
            raise AssertionError("completion benchmark bundle parity mismatch")
        if min(reference_ns, packed_ns) >= min_sample_ns:
            break
        bundle_size *= 2

    pairs: list[dict[str, Any]] = []
    reference_samples: list[float] = []
    packed_samples: list[float] = []
    pair_count = max(1, int(repeats))
    for index in range(pair_count):
        order = ("reference", "packed") if index % 2 == 0 else ("packed", "reference")
        measured: dict[str, tuple[int, list[str], Any]] = {}
        for arm in order:
            measured[arm] = _timed_bundle(
                reference_fn if arm == "reference" else packed_fn,
                bundle_size,
            )
        reference_ns, reference_digests, reference_value = measured["reference"]
        packed_ns, packed_digests, packed_value = measured["packed"]
        if reference_digests != packed_digests:
            raise AssertionError(
                f"completion benchmark pair {index} parity mismatch"
            )
        reference_ns_op = reference_ns / bundle_size
        packed_ns_op = packed_ns / bundle_size
        reference_samples.append(reference_ns_op)
        packed_samples.append(packed_ns_op)
        pairs.append(
            {
                "index": index,
                "order": list(order),
                "reference_ns_per_op": round(reference_ns_op, 3),
                "packed_ns_per_op": round(packed_ns_op, 3),
                "packed_over_reference": round(
                    packed_ns_op / reference_ns_op, 6
                ),
                "output_digest": reference_digests[-1],
            }
        )

    reference_median, reference_mad = _median_mad(reference_samples)
    packed_median, packed_mad = _median_mad(packed_samples)
    reference_relative_mad = reference_mad / reference_median if reference_median else 0
    packed_relative_mad = packed_mad / packed_median if packed_median else 0
    unstable = max(reference_relative_mad, packed_relative_mad) > (
        _COMPLETION_MAX_RELATIVE_MAD
    )
    return {
        "pair_count": pair_count,
        "bundle_size": bundle_size,
        "pilot_excluded": True,
        "minimum_sample_ns": min_sample_ns,
        "pairs": pairs,
        "reference": {
            "median_ns_per_op": round(reference_median, 3),
            "mad_ns_per_op": round(reference_mad, 3),
            "min_ns_per_op": round(min(reference_samples), 3),
            "max_ns_per_op": round(max(reference_samples), 3),
            "relative_mad": round(reference_relative_mad, 6),
        },
        "packed": {
            "median_ns_per_op": round(packed_median, 3),
            "mad_ns_per_op": round(packed_mad, 3),
            "min_ns_per_op": round(min(packed_samples), 3),
            "max_ns_per_op": round(max(packed_samples), 3),
            "relative_mad": round(packed_relative_mad, 6),
        },
        "packed_over_reference": round(packed_median / reference_median, 6),
        "speedup": round(reference_median / packed_median, 6),
        "unstable": unstable,
        "output_digest": _result_digest(reference_value),
        "reference_value": reference_value,
        "packed_value": packed_value,
    }


def _measurement_evidence(measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in measurement.items()
        if key not in {"reference_value", "packed_value"}
    }


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
    _openui_completion_domain(packed_request)

    def packed_graph_query():
        session = getattr(packed_request.state, "completion_session", None)
        if session is not None:
            session._results.clear()  # benchmark the graph/DP, not result replay
        return _openui_completion_domain(packed_request)

    measurement = _paired_measure(
        lambda: _openui_completion_domain_reference(reference_request),
        packed_graph_query,
        repeats,
    )
    reference = measurement.pop("reference_value")
    packed = measurement.pop("packed_value")
    session = getattr(packed_request.state, "completion_session", None)
    counters = session.stats() if session is not None else {}
    correct = packed == reference and len(packed.candidates) == 12
    work_ok = counters.get("general_forest_fallbacks", 0) == 0
    return {
        "n": 1,
        "prefix": "root = Card([b1,",
        "measurement": _measurement_evidence(measurement),
        "work_counters": counters,
        "correct": correct,
        "gate": (
            "exact 12-path parity, stable paired speedup >= 10, "
            "and zero supported-path general forest fallbacks"
        ),
        "pass": bool(
            correct
            and work_ok
            and not measurement["unstable"]
            and measurement["speedup"] >= 10
        ),
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
    warm = _paired_measure(
        lambda: [
            _openui_completion_domain_reference(row) for row in reference_requests
        ],
        lambda: [_openui_completion_domain(row) for row in packed_requests],
        repeats,
    )
    reference = warm.pop("reference_value")
    packed = warm.pop("packed_value")
    # Separate cold guardrail: fresh row state and no stateless forest cache.
    # Other immutable tokenizer/schema tables stay warm for both arms.
    compiler_draft._official_schema()

    def cold(provider, selected=texts):
        compiler_draft._STATELESS_FOREST_CACHE.clear()
        rows = requests(selected)
        return [provider(row) for row in rows]

    # Discard one cold-row warmup per arm so immutable tokenizer/schema memo
    # population is not charged to whichever implementation happens to run
    # first, then alternate measured order.
    cold_rows = []
    cold_ratios: list[float] = []
    cold_correct = True
    for text in texts[:3]:
        measurement = _paired_measure(
            lambda text=text: cold(
                _openui_completion_domain_reference, (text,)
            ),
            lambda text=text: cold(_openui_completion_domain, (text,)),
            repeats,
        )
        cold_reference = measurement.pop("reference_value")
        cold_packed = measurement.pop("packed_value")
        cold_correct = cold_correct and cold_packed == cold_reference
        cold_ratios.append(measurement["packed_over_reference"])
        cold_rows.append(
            {
                "prefix": text,
                "measurement": _measurement_evidence(measurement),
            }
        )
    cold_ratio = max(cold_ratios)
    unstable = warm["unstable"] or any(
        row["measurement"]["unstable"] for row in cold_rows
    )
    return {
        "n": len(packed_requests),
        "warm_measurement": _measurement_evidence(warm),
        "cold_simple_prefixes": cold_rows,
        "cold_packed_over_reference_max": round(cold_ratio, 6),
        "correct": packed == reference and cold_correct,
        "gate": "exact parity and cold packed time <= 1.15x reference",
        "pass": bool(
            packed == reference
            and cold_correct
            and not unstable
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
    warm_cache_ms, _ = _timed_ms(lambda: state.allowed_ids(remaining), repeats)

    def packed_query():
        tokenizer.allowed_cache.clear()
        return state.allowed_ids(remaining)

    measurement = _paired_measure(
        lambda: _legacy_choice_allowed(state, remaining),
        packed_query,
        repeats,
    )
    reference = measurement.pop("reference_value")
    packed = measurement.pop("packed_value")
    return {
        "n": 1,
        "remaining_tokens": remaining,
        "warm_allowed_cache_ms": round(warm_cache_ms, 4),
        "measurement": _measurement_evidence(measurement),
        "correct": packed == reference,
        "gate": "exact parity and stable paired speedup >= 3",
        "pass": bool(
            packed == reference
            and not measurement["unstable"]
            and measurement["speedup"] >= 3
        ),
    }


def _completion_solver_bench(repeats: int) -> dict[str, Any]:
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        build_completion_forest,
    )
    from slm_training.dsl.grammar.fastpath.token_map import decode_prefix
    from slm_training.dsl.solver.adapters import completion_forest_state
    from slm_training.dsl.solver.openui_support import OpenUIForestExpander
    from slm_training.dsl.solver.state import SolverBounds
    from slm_training.dsl.solver.support import (
        EnumerativeSupportOracle,
        ExpandStatus,
        SupportQuery,
        replay_support_certificate,
    )
    from slm_training.dsl.solver.openui_support import OpenUIWellFormedVerifier
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

    def state_payload(state):
        if state is None:
            return None
        return {
            "fingerprint": state.fingerprint,
            "holes": [
                {
                    "hole_id": hole.hole_id,
                    "values": [value.payload for value in hole.values],
                    "metadata": hole.metadata,
                }
                for hole in state.holes
            ],
        }

    def step_payload(step):
        return {
            "status": step.status.value,
            "coverage": step.coverage,
            "program_digest": (
                _result_digest(step.program) if step.program is not None else None
            ),
            "next_state": state_payload(step.next_state),
            "detail": step.detail,
        }

    def packed():
        return [
            step_payload(
                packed_expander.successor(
                    packed_root, packed_root.holes[0].hole_id, value
                )
            )
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
            kind = str(value.payload.get("kind", ""))
            if kind == "eos" or token_ids == (int(tokenizer.eos_id),):
                rows.append(
                    {
                        "status": ExpandStatus.TERMINAL.value,
                        "coverage": "complete",
                        "program_digest": _result_digest(
                            decode_prefix(tokenizer, list(prefix))
                        ),
                        "next_state": None,
                        "detail": f"prefix_len={len(prefix)}",
                    }
                )
                continue
            forest = build_completion_forest(
                tokenizer, list(prefix + token_ids), max_path_tokens=8
            )
            if forest.coverage == "none":
                rows.append(
                    {
                        "status": ExpandStatus.DEAD.value,
                        "coverage": "none",
                        "program_digest": None,
                        "next_state": None,
                        "detail": "illegal_prefix",
                    }
                )
            elif not forest.paths:
                rows.append(
                    {
                        "status": ExpandStatus.INCOMPLETE.value,
                        "coverage": forest.coverage,
                        "program_digest": None,
                        "next_state": None,
                        "detail": "no_enumerated_actions",
                    }
                )
            else:
                child = completion_forest_state(
                    prefix_ids=prefix + token_ids,
                    forest=forest,
                    pack_id="openui",
                    constraint_version="bench",
                    bounds=bounds,
                )
                rows.append(
                    {
                        "status": ExpandStatus.CONTINUE.value,
                        "coverage": forest.coverage,
                        "program_digest": None,
                        "next_state": state_payload(child),
                        "detail": None,
                    }
                )
        return rows

    # Populate each request-local successor once; timed packed calls are cache hits.
    packed()
    measurement = _paired_measure(reference, packed, repeats)
    reference_rows = measurement.pop("reference_value")
    packed_rows = measurement.pop("packed_value")

    verifier = OpenUIWellFormedVerifier()
    oracle = EnumerativeSupportOracle(packed_expander, verifier)
    certificates = []
    replay_failures = 0
    for value in packed_root.holes[0].values:
        query = SupportQuery(
            state_fingerprint=packed_root.fingerprint,
            hole_id=packed_root.holes[0].hole_id,
            candidate=value,
        )
        support = oracle.check(packed_root, query)
        replay = replay_support_certificate(
            support.certificate,
            state=packed_root,
            expander=OpenUIForestExpander(
                tokenizer,
                prefix,
                pack_id="openui",
                constraint_version="bench",
                bounds=bounds,
            ),
            verifier=verifier,
        )
        replay_failures += int(not replay.ok)
        certificates.append(
            {
                "verdict": support.verdict.value,
                "certificate_digest": support.certificate.digest,
                "replay_ok": replay.ok,
                "replay_violations": list(replay.violations),
            }
        )

    correct = reference_rows == packed_rows
    unique_successors = len(packed_root.holes[0].values)
    return {
        "n": len(reference_rows),
        "measurement": _measurement_evidence(measurement),
        "reference_nonempty": bool(reference_rows),
        "packed_nonempty": bool(packed_rows),
        "correct": correct,
        "cached_successors": len(packed_expander._successors),
        "certificates": certificates,
        "certificate_replay_failures": replay_failures,
        "gate": (
            "exact successor parity, replay-valid certificates, one expansion "
            "per unique successor, and packed no slower than reference"
        ),
        "pass": bool(
            reference_rows
            and packed_rows
            and correct
            and replay_failures == 0
            and len(packed_expander._successors) == unique_successors
            and not measurement["unstable"]
            and measurement["packed_over_reference"] <= 1.0
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

    stats_by_arm: dict[str, list[DecodeStats]] = {"reference": [], "packed": []}

    def run(provider, arm: str):
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
            stats_by_arm[arm].append(stats)
            text = model._decode_ids(ids)
            certified = model._canonical_valid_openui(text)
            return {
                "token_ids": tuple(int(v) for v in ids.tolist()),
                "text": text,
                "certified_text": certified,
                "certified": certified is not None,
            }
        finally:
            register_pack(pack)

    def reference_run():
        compiler_draft._STATELESS_FOREST_CACHE.clear()
        return run(_openui_completion_domain_reference, "reference")

    def packed_run():
        compiler_draft._STATELESS_FOREST_CACHE.clear()
        return run(_openui_completion_domain, "packed")

    measurement = _paired_measure(reference_run, packed_run, repeats)
    reference_value = measurement.pop("reference_value")
    packed_value = measurement.pop("packed_value")
    measured_invocations = measurement["pair_count"] * measurement["bundle_size"]
    reference_stats = stats_by_arm["reference"][-measured_invocations:]
    packed_stats_rows = stats_by_arm["packed"][-measured_invocations:]
    reference_compiler_ms = statistics.median(
        row.compiler_ms for row in reference_stats
    )
    packed_compiler_ms = statistics.median(
        row.compiler_ms for row in packed_stats_rows
    )
    reference_stats_last = reference_stats[-1]
    packed_stats = packed_stats_rows[-1]
    compiler_speedup = (
        reference_compiler_ms / packed_compiler_ms
        if packed_compiler_ms
        else None
    )
    work_counters = {
        "completion_session_starts": packed_stats.completion_session_starts,
        "completion_full_prefix_lex_bytes": (
            packed_stats.completion_full_prefix_lex_bytes
        ),
        "completion_candidate_engine_allocations": (
            packed_stats.completion_candidate_engine_allocations
        ),
        "completion_general_forest_builds": getattr(
            packed_stats, "completion_general_forest_builds", 0
        ),
        "completion_value_tree_clones": getattr(
            packed_stats, "completion_value_tree_clones", 0
        ),
        "completion_ast_bridge_calls": getattr(
            packed_stats, "completion_ast_bridge_calls", 0
        ),
        "completion_edge_replays": getattr(
            packed_stats, "completion_edge_replays", 0
        ),
    }
    work_ok = (
        work_counters["completion_session_starts"] == 1
        and work_counters["completion_general_forest_builds"] == 0
        and work_counters["completion_full_prefix_lex_bytes"] == 0
        and work_counters["completion_candidate_engine_allocations"] == 0
        and work_counters["completion_value_tree_clones"] == 0
        and work_counters["completion_ast_bridge_calls"] == 0
        and work_counters["completion_edge_replays"] == 0
    )
    correct = reference_value == packed_value
    return {
        "n": 1,
        "device": device,
        "measurement": _measurement_evidence(measurement),
        "reference_compiler_ms": round(reference_compiler_ms, 4),
        "packed_compiler_ms": round(packed_compiler_ms, 4),
        "compiler_speedup": (
            round(compiler_speedup, 3) if compiler_speedup is not None else None
        ),
        "correct": correct,
        "reference_forwards": reference_stats_last.forwards_count,
        "packed_forwards": packed_stats.forwards_count,
        "work_counters": work_counters,
        "gate": (
            "identical ids/text/certification, stable compiler_ms speedup >= 5, "
            "no extra forwards, one session, and zero supported-path rebuild work"
        ),
        "pass": bool(
            correct
            and work_ok
            and not measurement["unstable"]
            and compiler_speedup is not None
            and compiler_speedup >= 5
            and packed_stats.forwards_count <= reference_stats_last.forwards_count
        ),
    }


_COMPLETION_WORKLOADS = {
    "direct": _completion_direct_bench,
    "corpus": _completion_corpus_bench,
    "choice": _completion_choice_bench,
    "solver": _completion_solver_bench,
}


def _completion_recipe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "execution": "independently capped workload shards",
        "honesty_mode": "fixture_perf_not_ship",
        "hard_cap_minutes": MAX_RUN_MINUTES,
        "pair_count": args.completion_repeats,
        "pair_order": "alternating_ab_ba",
        "pilot": "one excluded pair",
        "minimum_sample_ns": _COMPLETION_MIN_SAMPLE_NS,
        "relative_mad_limit": _COMPLETION_MAX_RELATIVE_MAD,
    }


def _completion_fixture_digest(name: str) -> str:
    return _result_digest(
        {
            "schema": "completion-kernel-perf/v1",
            "workload": name,
            "fixtures": {
                "direct_prefix": "root = Card([b1,",
                "corpus_prefixes": [
                    "",
                    "root",
                    "root =",
                    "root = Card([",
                    "root = Card([b1",
                    "root = Card([b1,",
                    'root = TextContent(":slot_0")',
                ],
                "slot_contract": [":slot_0", ":slot_1"],
                "remaining_tokens": {"direct": 32, "choice": 12},
                "decode_tokens": 16,
            },
        }
    )


def _completion_row_compatibility(row: dict[str, Any]) -> dict[str, Any]:
    stamp = row.get("version_stamp", {})
    return {
        "run_id": row.get("run_id"),
        "code_commit": stamp.get("code_commit"),
        "code_dirty": stamp.get("code_dirty"),
        "components": stamp.get("components"),
        "recipe": row.get("recipe"),
        "host": row.get("host"),
    }


def _merge_completion_shards(args: argparse.Namespace) -> int:
    shard_root = args.completion_shard_root / args.completion_run_id
    rows: dict[str, dict[str, Any]] = {}
    missing = []
    for name in _COMPLETION_REQUIRED_WORKLOADS:
        path = shard_root / f"{name}.json"
        if not path.is_file():
            missing.append(name)
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("workload") != name:
            raise ValueError(f"{path}: workload does not match filename")
        if row.get("fixture_manifest_digest") != _completion_fixture_digest(name):
            raise ValueError(f"{path}: stale or mismatched fixture manifest")
        rows[name] = row
    if missing:
        raise ValueError(f"completion shards missing workloads: {missing}")

    first = _completion_row_compatibility(rows[_COMPLETION_REQUIRED_WORKLOADS[0]])
    if first["run_id"] != args.completion_run_id:
        raise ValueError("completion shard run_id does not match merge run_id")
    if first["code_dirty"]:
        raise ValueError("accepted completion evidence requires code_dirty=false")
    for name, row in rows.items():
        compatibility = _completion_row_compatibility(row)
        if compatibility != first:
            raise ValueError(
                f"completion shard {name} has mixed run/commit/stamp/recipe/host"
            )

    overall_pass = all(bool(row.get("pass")) for row in rows.values())
    board = {
        "schema_version": "completion-kernel-perf/v1",
        "run_id": args.completion_run_id,
        "overall_status": "pass" if overall_pass else "fail",
        "overall_pass": overall_pass,
        "recipe": first["recipe"],
        "source": {
            "code_commit": first["code_commit"],
            "code_dirty": first["code_dirty"],
            "components": first["components"],
            "host": first["host"],
        },
        "workloads": rows,
        "version_stamp": rows[_COMPLETION_REQUIRED_WORKLOADS[0]][
            "version_stamp"
        ],
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(board, indent=2))
    return 0 if overall_pass else 1


def _run_completion_kernel_mode(args: argparse.Namespace) -> int:
    if args.completion_merge:
        return _merge_completion_shards(args)
    names = [
        name.strip().lower()
        for name in args.completion_kernel_workload.split(",")
        if name.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError("completion-kernel workload list contains duplicates")
    unknown = set(names) - (set(_COMPLETION_WORKLOADS) | {"decode"})
    if unknown:
        raise ValueError(f"unknown completion-kernel workloads: {sorted(unknown)}")
    if not names:
        raise ValueError("completion-kernel mode needs a workload or --completion-merge")
    stamp = build_version_stamp(*_COMPLETION_STAMP_COMPONENTS)
    recipe = _completion_recipe(args)
    shard_root = args.completion_shard_root / args.completion_run_id
    shard_root.mkdir(parents=True, exist_ok=True)
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
        result["schema_version"] = "completion-kernel-workload/v1"
        result["run_id"] = args.completion_run_id
        result["workload"] = name
        result["fixture_manifest_digest"] = _completion_fixture_digest(name)
        result["recipe"] = recipe
        result["version_stamp"] = stamp
        results[name] = result
        (shard_root / f"{name}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"run_id": args.completion_run_id, "workloads": results}, indent=2))
    return 0 if all(row["pass"] for row in results.values()) else 1


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
        default=7,
        help="Alternating paired samples per completion-kernel workload.",
    )
    parser.add_argument(
        "--completion-run-id",
        default="",
        help="Shared run id for completion workload shards and their merge.",
    )
    parser.add_argument(
        "--completion-shard-root",
        type=Path,
        default=Path("outputs/runs/completion_kernel"),
        help="Ignored root for independently capped completion workload shards.",
    )
    parser.add_argument(
        "--completion-merge",
        action="store_true",
        help="Merge a complete homogeneous five-workload completion run into docs.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print selected experiment definitions without running them.",
    )
    args = parser.parse_args(argv)
    if args.completion_kernel_workload or args.completion_merge:
        if not args.completion_run_id:
            args.completion_run_id = dt.datetime.now(dt.UTC).strftime(
                "completion-kernel-%Y%m%dT%H%M%SZ"
            )
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
