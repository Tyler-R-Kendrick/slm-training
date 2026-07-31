"""Observability (read) and action (exec) routers for the control-plane app.

Read endpoints are pure filesystem reads (safe everywhere, including Vercel).
The gate/promotion evaluation endpoints are pure math (no subprocess, no FS
writes) so the Checkpoints gate editor is fully live even in read-only mode.
Exec endpoints (jobs, comparisons) require ``capabilities.execution`` and 403
otherwise. State (readers / capabilities / jobs registry) lives on ``app.state``.

``otel_ingest_router`` has no ``/api`` prefix: it serves the standard OTLP/HTTP
paths (``/v1/traces``, ``/v1/logs``) that ``RunTrace._mirror`` derives from a
base endpoint URL, so any app instance can act as a telemetry peer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from slm_training.web.otel_hub import MAX_INGEST_BYTES

from slm_training.autoresearch.run_insights import RunInsightSubmission
from slm_training.harness_core.promotion_engine import (
    PromotionCriteria,
    evaluate_promotion as evaluate_promotion_engine,
)
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)
from slm_training.features.levers import feature_flag_registry_payload

observability_router = APIRouter(prefix="/api")
actions_router = APIRouter(prefix="/api")
otel_ingest_router = APIRouter()
_PROMOTION_HARD_CATEGORIES = ("binding", "structural", "repair")


def _readers(request: Request):
    return request.app.state.readers


def _otel(request: Request):
    return request.app.state.otel


def _require_execution(request: Request):
    caps = request.app.state.capabilities
    if not caps.execution:
        raise HTTPException(
            status_code=403, detail="execution disabled (read-only deployment)"
        )
    return request.app.state.jobs


def _runtime_planning_defaults() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Detect host facts and expose explicit defaults for unknown rate inputs."""

    architecture = os.uname().machine
    logical_cores = os.cpu_count() or 1
    cpu_model = ""
    frequency_ghz: float | None = None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        normalized = key.strip().casefold()
        if not cpu_model and normalized in {"model name", "hardware"}:
            cpu_model = value.strip()
        if frequency_ghz is None and normalized == "cpu mhz":
            try:
                frequency_ghz = float(value.strip()) / 1000
            except ValueError:
                pass
    frequency_source = "runtime /proc/cpuinfo"
    if frequency_ghz is None:
        for path in (
            Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        ):
            try:
                frequency_ghz = float(path.read_text(encoding="utf-8").strip()) / 1e6
                frequency_source = f"runtime {path}"
                break
            except (OSError, ValueError):
                continue
    if frequency_ghz is None:
        frequency_ghz = 2.5
        frequency_source = "assumed default; host frequency is not OS-readable"

    try:
        memory_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (OSError, ValueError):
        memory_gb = None

    # SIMD issue width is not a portable OS fact. The architecture-specific
    # value makes the assumption explicit and editable instead of pretending
    # that cpu_count alone is a measured compute rate.
    ops_per_cycle = 16.0 if architecture in {"x86_64", "amd64"} else 8.0
    peak_compute_gops = logical_cores * frequency_ghz * ops_per_cycle
    machine = {
        "architecture": architecture,
        "cpu_model": cpu_model or architecture,
        "logical_cores": logical_cores,
        "memory_gb": round(memory_gb, 1) if memory_gb is not None else None,
        "frequency_ghz": round(frequency_ghz, 3),
        "frequency_source": frequency_source,
    }
    assumptions = [
        {
            "key": "peak_compute_gops",
            "label": "Peak compute",
            "value": round(peak_compute_gops, 2),
            "unit": "Gop/s",
            "minimum": 0.001,
            "step": 1,
            "source": (
                f"runtime cores × frequency × assumed {ops_per_cycle:g} SIMD ops/cycle"
            ),
        },
        {
            "key": "memory_bandwidth_gbs",
            "label": "Memory bandwidth",
            "value": 25.0,
            "unit": "GB/s",
            "minimum": 0.001,
            "step": 1,
            "source": "assumed default; sustainable bandwidth is not OS-readable",
        },
        {
            "key": "compute_utilization",
            "label": "Compute utilization",
            "value": 0.35,
            "unit": "fraction",
            "minimum": 0.001,
            "maximum": 1,
            "step": 0.05,
            "source": "assumed planning efficiency",
        },
        {
            "key": "bandwidth_utilization",
            "label": "Bandwidth utilization",
            "value": 0.6,
            "unit": "fraction",
            "minimum": 0.001,
            "maximum": 1,
            "step": 0.05,
            "source": "assumed planning efficiency",
        },
        {
            "key": "ops_per_parameter_forward",
            "label": "Work per parameter/forward",
            "value": 2.0,
            "unit": "operations",
            "minimum": 0.001,
            "step": 0.25,
            "source": "dense-forward planning assumption",
        },
        {
            "key": "bytes_per_parameter_forward",
            "label": "Traffic per parameter/forward",
            "value": 2.0,
            "unit": "bytes",
            "minimum": 0.001,
            "step": 0.25,
            "source": "FP16 single-token planning assumption",
        },
        {
            "key": "independent_streams",
            "label": "Independent decode streams",
            "value": float(logical_cores),
            "unit": "streams",
            "minimum": 1,
            "step": 1,
            "source": "runtime logical-core default",
        },
        {
            "key": "dispatch_step_us",
            "label": "Base serial dispatch",
            "value": 25.0,
            "unit": "µs/step",
            "minimum": 0.001,
            "step": 5,
            "source": "assumed optimized dispatch floor",
        },
        {
            "key": "oracle_operation_us",
            "label": "Request-local oracle operation",
            "value": 1.0,
            "unit": "µs/operation",
            "minimum": 0.001,
            "step": 0.25,
            "source": "assumed semantic-mask/branch operation cost",
        },
    ]
    return machine, assumptions


def _completion_kernel_evidence(repo_root: Path) -> dict[str, Any]:
    """Normalize main's committed completion-kernel calculations for the UI."""

    path = repo_root / "docs/design/completion-kernel-perf-results.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"timings": [], "work": [], "gates": [], "provenance": "missing"}

    def timing_row(
        scope: str,
        row: dict[str, Any],
        *,
        baseline_key: str = "v1",
        candidate_key: str = "v2",
        speedup_key: str = "speedup_v1_over_v2",
        row_id: str | None = None,
    ) -> dict[str, Any]:
        baseline = row.get(baseline_key) or {}
        candidate = row.get(candidate_key) or {}
        return {
            "scope": scope,
            "id": row_id or row.get("id"),
            "mode": row.get("mode") or "",
            "candidate_count": row.get("candidate_count"),
            "baseline_median_ms": float(baseline.get("median_ns", 0)) / 1e6,
            "candidate_min_ms": float(candidate.get("min_ns", 0)) / 1e6,
            "candidate_median_ms": float(candidate.get("median_ns", 0)) / 1e6,
            "candidate_max_ms": float(candidate.get("max_ns", 0)) / 1e6,
            "candidate_mad_ms": float(candidate.get("mad_ns", 0)) / 1e6,
            "speedup": row.get(speedup_key),
            "equivalent": row.get("equivalent", row.get("v1_authority_preserved")),
        }

    timings = [
        timing_row("completion domain", row)
        for row in payload.get("rows") or []
        if isinstance(row, dict)
    ]
    choice = payload.get("choice_codec") or {}
    timings.extend(
        timing_row("choice codec", row)
        for row in choice.get("rows") or []
        if isinstance(row, dict)
    )
    solver = payload.get("solver_fixture")
    if isinstance(solver, dict):
        timings.append(timing_row("solver", solver))
    compiler = payload.get("compiler_decode_fixture")
    if isinstance(compiler, dict):
        timings.extend(
            (
                timing_row("compiler decode", compiler),
                timing_row(
                    "compiler phase",
                    compiler,
                    baseline_key="v1_compiler",
                    candidate_key="v2_compiler",
                    speedup_key="compiler_ms_speedup_v1_over_v2",
                    row_id=f"{compiler.get('id')}_compiler",
                ),
            )
        )
    batch = payload.get("compiler_batch_fixture")
    if isinstance(batch, dict):
        timings.append(
            timing_row(
                "compiler batch",
                batch,
                baseline_key="control",
                candidate_key="compact",
                speedup_key="speedup_control_over_compact",
            )
        )

    warm_row = next(
        (
            row
            for row in payload.get("rows") or []
            if isinstance(row, dict) and isinstance(row.get("v2_work_delta"), dict)
        ),
        {},
    )
    work = [
        {"metric": key, "value": value, "scope": warm_row.get("id")}
        for key, value in sorted((warm_row.get("v2_work_delta") or {}).items())
    ]
    if isinstance(batch, dict):
        control = batch.get("control_payload") or {}
        compact = batch.get("compact_payload") or {}
        work.extend(
            {
                "metric": metric,
                "value": compact.get(key),
                "baseline": control.get(key),
                "scope": batch.get("id"),
            }
            for metric, key in (
                ("forwards_count", "forwards_count"),
                ("denoiser_rows_evaluated", "denoiser_rows_evaluated"),
                ("ambiguous_rows_forwarded", "ambiguous_rows_forwarded"),
            )
        )

    gates = [
        {"gate": key, "pass": value}
        for key, value in (payload.get("gates") or {}).items()
    ]
    return {
        "path": str(path.relative_to(repo_root)),
        "kind": payload.get("kind"),
        "claim_class": payload.get("claim_class"),
        "device": payload.get("device"),
        "repetitions": payload.get("repetitions"),
        "environment": payload.get("environment") or {},
        "timings": timings,
        "work": work,
        "gates": gates,
        "provenance": "committed",
    }


def _completion_artifact_evidence(repo_root: Path) -> dict[str, Any]:
    """Expose immutable artifact metadata without claiming a runtime check."""

    artifact = (
        repo_root / "src/slm_training/resources/decode/openui_completion_v1.safetensors"
    )
    manifest_path = artifact.with_suffix(".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError):
        return {
            "status": "missing",
            "path": str(artifact.relative_to(repo_root)),
            "manifest": str(manifest_path.relative_to(repo_root)),
        }
    digest_matches = digest == manifest.get("artifact_sha256")
    return {
        "status": "hash_verified" if digest_matches else "hash_mismatch",
        "path": str(artifact.relative_to(repo_root)),
        "manifest": str(manifest_path.relative_to(repo_root)),
        "schema": manifest.get("schema"),
        "checker_schema": manifest.get("checker_schema"),
        "sha256": digest,
        "lalr_state_count": manifest.get("lalr_state_count"),
        "lalr_edge_count": manifest.get("lalr_edge_count"),
        "direct_token_count": manifest.get("direct_token_count"),
        "request_dependent_authority_excluded": manifest.get(
            "request_dependent_authority_excluded", []
        ),
        "runtime_policy": (
            "The decoder independently reconstructs grammar/tokenizer authority; "
            "missing, stale, corrupt, or unequal artifacts fall back to the exact "
            "live path. Scope and semantic legality stay request-local."
        ),
    }


def _dsl_pack_payload(_request: Request, pack_id: str) -> dict[str, Any]:
    """Source-backed pack details and calculated experiment-cap metrics."""
    from dataclasses import fields

    from slm_training.data.progspec.generate import GeneratorConfig
    from slm_training.dsl.grammar.backends.types import REPO_ROOT
    from slm_training.dsl.pack import get_pack, list_packs
    from slm_training.harnesses.model_build.config import ModelBuildConfig
    from slm_training.levers import (
        DEFAULT_DECODE_TIMEOUT_SECONDS,
        MAX_HARNESS_WALL_SECONDS,
        MAX_RUN_MINUTES,
    )

    pack = get_pack(pack_id)
    info = pack.backend.info
    grammar_path = info.grammar_path
    grammar_source = (
        grammar_path.read_text(encoding="utf-8")
        if grammar_path is not None and grammar_path.is_file()
        else ""
    )
    try:
        grammar_display_path = (
            str(grammar_path.relative_to(REPO_ROOT)) if grammar_path else "external"
        )
    except ValueError:
        grammar_display_path = str(grammar_path)

    authority = pack.grammar_capability_authority
    productions = tuple(authority.productions or ()) if authority else ()
    recursive_nonterminals = 0
    if authority and productions:
        from slm_training.dsl.grammar_capabilities import GrammarCapabilityAdapterV1

        analysis = GrammarCapabilityAdapterV1(pack).nonterminal_analysis
        if isinstance(analysis, tuple):
            recursive_nonterminals = sum(item.recursive for item in analysis)

    # Introspection must stay a static read. The active OpenUI lang-core backend
    # may start its Node bridge for library schema calls, while the paired Lark
    # backend exposes the same committed grammar metadata in-process.
    metadata_backend = pack.backend
    if pack.pack_id == "openui":
        from slm_training.dsl.grammar.backends import get_backend

        metadata_backend = get_backend("openui-lark")

    def safe_set(method_name: str) -> frozenset[str]:
        if metadata_backend.info.kind not in {"lark"}:
            return frozenset()
        try:
            return frozenset(getattr(metadata_backend, method_name)())
        except Exception:  # noqa: BLE001 - optional pack backends may be unavailable
            return frozenset()

    components = safe_set("component_names") | frozenset(
        (*pack.leaf_components, *pack.container_components)
    )
    structural_tokens = safe_set("structural_tokens")
    all_fields = {item.name for item in fields(pack)}
    filled = set(pack.filled_slots())
    missing = sorted(all_fields - filled)

    constraints = [
        {
            "constraint": "Backend availability",
            "value": "available" if pack.backend.available() else "unavailable",
            "authority": info.id,
        },
        {
            "constraint": "Required root",
            "value": info.root_component or "backend-defined",
            "authority": "grammar backend",
        },
        {
            "constraint": "Placeholder syntax",
            "value": pack.placeholder_policy.placeholder_re.pattern,
            "authority": "placeholder policy",
        },
        {
            "constraint": "Routed content props",
            "value": ", ".join(sorted(pack.placeholder_policy.content_props)) or "none",
            "authority": "placeholder policy",
        },
        {
            "constraint": "Validity reward",
            "value": pack.reward_label,
            "authority": "pack oracle contract",
        },
        {
            "constraint": "Missing optional slots",
            "value": ", ".join(missing) or "none",
            "authority": "DslPack",
        },
    ]

    templates = [
        {"kind": kind, "syntax": syntax, "authority": "pack statement_templates"}
        for kind, syntax in pack.statement_templates
    ]
    if not templates:
        templates = [
            {
                "kind": "none declared",
                "syntax": "This pack does not declare additional statement templates.",
                "authority": "pack statement_templates",
            }
        ]

    defaults = ModelBuildConfig.__dataclass_fields__
    default = lambda name: defaults[name].default  # noqa: E731
    boundaries = [
        {
            "scope": "pack fact",
            "boundary": "AST nodes",
            "minimum": "1",
            "maximum": "unbounded" if recursive_nonterminals else "backend-defined",
            "authority": f"{recursive_nonterminals} recursive nonterminal(s)",
        },
        {
            "scope": "training",
            "boundary": "Topology AST nodes",
            "minimum": "1",
            "maximum": str(default("topology_max_nodes")),
            "authority": "ModelBuildConfig.topology_max_nodes",
        },
        {
            "scope": "training",
            "boundary": "Topology active nodes",
            "minimum": "1",
            "maximum": str(default("topology_max_active")),
            "authority": "ModelBuildConfig.topology_max_active",
        },
        {
            "scope": "training",
            "boundary": "Model hidden width",
            "minimum": str(default("d_model")),
            "maximum": str(default("d_model")),
            "authority": "ModelBuildConfig.d_model",
        },
        {
            "scope": "training",
            "boundary": "Topology depth",
            "minimum": "1",
            "maximum": str(default("topology_max_depth")),
            "authority": "ModelBuildConfig.topology_max_depth",
        },
        {
            "scope": "training",
            "boundary": "Topology arity",
            "minimum": "0",
            "maximum": str(default("topology_max_arity")),
            "authority": "ModelBuildConfig.topology_max_arity",
        },
        {
            "scope": "training",
            "boundary": "Topology phases",
            "minimum": "1",
            "maximum": str(default("topology_max_phases")),
            "authority": "ModelBuildConfig.topology_max_phases",
        },
        {
            "scope": "training",
            "boundary": "LTR output tokens",
            "minimum": "1",
            "maximum": str(default("grammar_ltr_max_tokens")),
            "authority": "ModelBuildConfig.grammar_ltr_max_tokens",
        },
        {
            "scope": "training",
            "boundary": "Decode attempts",
            "minimum": "1",
            "maximum": str(default("generate_max_attempts")),
            "authority": "ModelBuildConfig.generate_max_attempts",
        },
        {
            "scope": "experiment policy",
            "boundary": "Decode timeout",
            "minimum": str(DEFAULT_DECODE_TIMEOUT_SECONDS),
            "maximum": str(DEFAULT_DECODE_TIMEOUT_SECONDS),
            "authority": "levers.DEFAULT_DECODE_TIMEOUT_SECONDS",
        },
        {
            "scope": "experiment policy",
            "boundary": "Harness work wall",
            "minimum": str(MAX_HARNESS_WALL_SECONDS),
            "maximum": str(MAX_HARNESS_WALL_SECONDS),
            "authority": "levers.MAX_HARNESS_WALL_SECONDS",
        },
        {
            "scope": "experiment policy",
            "boundary": "Hard command wall",
            "minimum": str(MAX_RUN_MINUTES * 60),
            "maximum": str(MAX_RUN_MINUTES * 60),
            "authority": "levers.MAX_RUN_MINUTES",
        },
    ]
    if pack.pack_id == "openui":
        generator = GeneratorConfig()
        boundaries[1:1] = [
            {
                "scope": "training",
                "boundary": "ProgramSpec depth",
                "minimum": "1",
                "maximum": str(generator.max_depth),
                "authority": "GeneratorConfig.max_depth",
            },
            {
                "scope": "training",
                "boundary": "ProgramSpec width",
                "minimum": "1",
                "maximum": str(generator.max_width),
                "authority": "GeneratorConfig.max_width",
            },
        ]

    maximum_nodes = int(default("topology_max_nodes"))
    maximum_arity = int(default("topology_max_arity"))
    maximum_depth = int(default("topology_max_depth"))
    maximum_phases = int(default("topology_max_phases"))
    maximum_active = max(1, int(default("topology_max_active")))

    # Encode the declared grammar as a canonical production-entry stream. Its
    # empirical zero-order entropy is a Shannon/MDL description target; the
    # fixed-width stream is a constructive upper target given the same symbol
    # dictionary. Neither number reads a checkpoint or an observed run.
    from collections import Counter

    grammar_stream: list[str] = []
    alternatives_by_lhs = Counter(production.lhs for production in productions)
    for production in productions:
        grammar_stream.extend((f"lhs:{production.lhs}", f"len:{len(production.rhs)}"))
        grammar_stream.extend(
            f"{symbol.kind}:{symbol.name}" for symbol in production.rhs
        )
    grammar_counts = Counter(grammar_stream)
    grammar_entries = len(grammar_stream)
    grammar_alphabet = len(grammar_counts)
    entropy_bits: int | None = None
    fixed_width_bits: int | None = None
    if grammar_entries and grammar_alphabet:
        entropy_bits = math.ceil(
            sum(
                count * -math.log2(count / grammar_entries)
                for count in grammar_counts.values()
            )
        )
        fixed_width_bits = grammar_entries * max(
            1, math.ceil(math.log2(grammar_alphabet))
        )

    # Collins et al. report 3–6 task-information bits/parameter across trained
    # recurrent architectures; Morris et al. independently estimate 3.6 for
    # GPT-style models. Use the former interval only as a grammar-memory
    # capacity prior, never as a whole-model or generalization estimate.
    minimum_memory_params = (
        math.ceil(entropy_bits / 6) if entropy_bits is not None else None
    )
    maximum_memory_params = (
        math.ceil(fixed_width_bits / 3) if fixed_width_bits is not None else None
    )

    # Work/span target for resolving one structural choice per possible AST
    # node with at most topology_max_active choices in parallel. The low span
    # is the shallowest max-arity tree that can contain N nodes; the high span
    # is the deepest permitted tree. Brent's scheduling bound supplies the high
    # turn target. This covers structure, not semantic correctness.
    if maximum_nodes <= 1:
        compact_span = 1
    elif maximum_arity <= 1:
        compact_span = maximum_nodes
    else:
        compact_span = math.ceil(
            math.log(maximum_nodes * (maximum_arity - 1) + 1, maximum_arity)
        )
    deep_span = max(1, min(maximum_depth, maximum_nodes))
    minimum_structural_turns = max(
        compact_span, math.ceil(maximum_nodes / maximum_active)
    )
    maximum_structural_turns = deep_span + math.ceil(
        max(0, maximum_nodes - deep_span) / maximum_active
    )
    turn_cap_delta = maximum_structural_turns - maximum_phases
    turn_cap_signal = (
        f"{turn_cap_delta} above configured phase cap"
        if turn_cap_delta > 0
        else f"{abs(turn_cap_delta)} turns headroom"
    )

    maximum_alternatives = max(alternatives_by_lhs.values(), default=1)
    maximum_choice_bits = math.ceil(maximum_nodes * math.log2(maximum_alternatives))
    machine, planning_assumptions = _runtime_planning_defaults()

    state = getattr(getattr(_request, "app", None), "state", None)
    readers = getattr(state, "readers", None)
    if readers is None:
        from slm_training.web.observability import Readers

        readers = Readers(REPO_ROOT)
    perf_board = readers.scoreboard("perf")
    quality_board = readers.scoreboard("quality")
    performance = readers.performance_insights()
    completion_kernel = _completion_kernel_evidence(REPO_ROOT)
    completion_artifact = _completion_artifact_evidence(REPO_ROOT)
    checkpoint_references = [
        {
            key: row.get(key)
            for key in (
                "role",
                "run_id",
                "parameters",
                "model_size",
                "throughput",
                "evaluation_status",
                "gate_pass",
                "provenance",
                "resource_basis",
                "status",
            )
        }
        for row in (performance.get("references") or [])
        if isinstance(row, dict)
    ]
    from slm_training.runtime.performance_target import build_performance_target

    performance_target = build_performance_target(
        checkpoint_references=checkpoint_references,
        assumptions=planning_assumptions,
        output_tokens=int(default("grammar_ltr_max_tokens")),
        structural_turn_range=(
            minimum_structural_turns,
            maximum_structural_turns,
        ),
        grammar_inputs={
            "productions": len(productions),
            "components": len(components),
            "structural_tokens": len(structural_tokens),
            "maximum_alternatives": maximum_alternatives,
            "maximum_nodes": maximum_nodes,
            "maximum_active": maximum_active,
            "artifact_lalr_states": int(
                completion_artifact.get("lalr_state_count") or 0
            ),
            "artifact_lalr_edges": int(completion_artifact.get("lalr_edge_count") or 0),
            "artifact_direct_tokens": int(
                completion_artifact.get("direct_token_count") or 0
            ),
        },
    )
    profile_floors = {
        str(profile.get("id")): float(profile["floor_tps"])
        for profile in performance_target.get("profiles", [])
        if profile.get("floor_tps") is not None
    }

    def target_gap(row: dict[str, Any]) -> dict[str, Any]:
        profile = str(row.get("throughput_profile") or "undeclared")
        observed = row.get("tokens_per_sec")
        try:
            observed_value = float(observed)
        except (TypeError, ValueError):
            observed_value = 0.0
        if profile in profile_floors:
            floor = profile_floors[profile]
            factor = floor / observed_value if observed_value > 0 else None
            return {
                "target_profile": profile,
                "target_floor_tps_min": floor,
                "target_floor_tps_max": floor,
                "target_gap_factor_min": factor,
                "target_gap_factor_max": factor,
                "target_gap_relation": (
                    "below" if factor is not None and factor >= 1 else "above"
                ),
                "target_gap_status": "comparable",
            }
        floors = list(profile_floors.values())
        factor_min = (
            min(floors) / observed_value if floors and observed_value > 0 else None
        )
        factor_max = (
            max(floors) / observed_value if floors and observed_value > 0 else None
        )
        return {
            "target_profile": "undeclared",
            "target_floor_tps_min": min(floors) if floors else None,
            "target_floor_tps_max": max(floors) if floors else None,
            "target_gap_factor_min": factor_min,
            "target_gap_factor_max": factor_max,
            "target_gap_relation": (
                "below"
                if factor_min is not None and factor_min >= 1
                else "above"
                if factor_max is not None and factor_max < 1
                else "spans"
            ),
            "target_gap_status": (
                "range only: evidence does not declare single_stream or aggregate"
            ),
        }

    perf_evidence = [
        {
            "id": row.get("id") or row.get("run_id"),
            "run_id": row.get("run_id") or row.get("id"),
            "description": row.get("description") or "",
            "n": row.get("n"),
            "tokens_per_sec": row.get("tokens_per_sec"),
            "latency_ms_p50": row.get("latency_ms_p50"),
            "parse_rate": row.get("parse_rate"),
            "placeholder_fidelity": row.get("placeholder_fidelity"),
            "provenance": row.get("provenance") or perf_board.get("provenance"),
            "evidence_class": "diagnostic perf/smoke; not ship evidence",
            **target_gap(row),
        }
        for row in (perf_board.get("results") or [])
        if isinstance(row, dict)
    ]
    smoke_evidence = []
    for row in quality_board.get("results") or []:
        suites = row.get("suites") if isinstance(row, dict) else None
        smoke = suites.get("smoke") if isinstance(suites, dict) else None
        if not isinstance(smoke, dict):
            continue
        smoke_evidence.append(
            {
                "id": row.get("id") or row.get("run_id"),
                "run_id": row.get("run_id") or row.get("id"),
                "n": smoke.get("n"),
                "meaningful_program_rate": smoke.get("meaningful_program_rate"),
                "placeholder_fidelity": smoke.get("placeholder_fidelity"),
                "reward_score": smoke.get("reward_score"),
                "pass": row.get("pass"),
                "provenance": row.get("provenance") or quality_board.get("provenance"),
                "evidence_class": "smoke suite; diagnostic, not ship evidence",
            }
        )

    calculated_rows = [
        {
            "metric": "Grammar description target",
            "minimum": str(entropy_bits or "unavailable"),
            "maximum": str(fixed_width_bits or "unavailable"),
            "unit": "bits",
            "formula": "Σ −nᵢ log₂(nᵢ/N) ↔ N⌈log₂|Σ|⌉",
            "inputs": "production LHS, RHS lengths, symbol frequencies",
            "cap_use": "grammar-memory floor",
        },
        {
            "metric": "Estimated optimal grammar-memory parameters",
            "minimum": str(minimum_memory_params or "unavailable"),
            "maximum": str(maximum_memory_params or "unavailable"),
            "unit": "parameters",
            "formula": "description bits ÷ 6 ↔ description bits ÷ 3",
            "inputs": "grammar description target, 3–6 task bits/parameter prior",
            "cap_use": "grammar-only model-size sweep",
        },
        {
            "metric": "AST legal-choice information",
            "minimum": "0",
            "maximum": str(maximum_choice_bits),
            "unit": "bits/record",
            "formula": "AST nodes × log₂(max alternatives/nonterminal)",
            "inputs": "topology_max_nodes, production alternatives by LHS",
            "cap_use": "ambiguity budget",
        },
        {
            "metric": "Estimated optimal structural turns",
            "minimum": str(minimum_structural_turns),
            "maximum": str(maximum_structural_turns),
            "unit": "turns/record",
            "formula": "max(critical span, ⌈work/active⌉) ↔ span + ⌈(work−span)/active⌉",
            "inputs": (
                "topology_max_nodes, topology_max_active, topology_max_arity, "
                "topology_max_depth"
            ),
            "cap_use": f"generation-turn sweep; {turn_cap_signal}",
        },
    ]
    calculated_tiles = [
        {
            "label": "Grammar information",
            "value": (
                f"{entropy_bits:,}–{fixed_width_bits:,} bits"
                if entropy_bits is not None and fixed_width_bits is not None
                else "—"
            ),
            "sub": "entropy code → fixed-width code",
            "accent": "",
        },
        {
            "label": "Grammar-memory target",
            "value": (
                f"{minimum_memory_params:,}–{maximum_memory_params:,}"
                if minimum_memory_params is not None
                and maximum_memory_params is not None
                else "—"
            ),
            "sub": "parameters; grammar storage only",
            "accent": "moss",
        },
        {
            "label": "Legal-choice information",
            "value": f"0–{maximum_choice_bits:,} bits",
            "sub": f"up to {maximum_alternatives} alternatives/node",
            "accent": "",
        },
        {
            "label": "Structural turns",
            "value": f"{minimum_structural_turns}–{maximum_structural_turns}",
            "sub": "work/span scheduling target",
            "accent": "ember" if turn_cap_delta > 0 else "",
        },
        {
            "label": "Phase-cap signal",
            "value": f"{turn_cap_delta:+d}" if turn_cap_delta else "aligned",
            "sub": turn_cap_signal,
            "accent": "ember" if turn_cap_delta > 0 else "moss",
        },
    ]
    calculated_basis = (
        "These are planning targets from the declared grammar and configured AST "
        "envelope only. Grammar memory uses a Shannon description-length interval "
        "and the published 3–6 task-bits/parameter capacity range. Turns use a "
        "work/span schedule that resolves each possible AST node once. No current "
        "model, checkpoint, run, telemetry, timeout, or observed throughput is an input."
    )
    unavailable_targets = [
        {
            "metric": "Optimal whole-model parameter size",
            "reason": "Grammar storage does not determine prompt mapping or generalization.",
            "required": "task distribution, target error, trainability/scaling law",
            "formula": "not identifiable from pack parameters",
        },
        {
            "metric": "Semantic correctness turns",
            "reason": "Structural coverage is not a semantic error-contraction model.",
            "required": "declared correctness target and a justified error-per-turn law",
            "formula": "not identifiable from grammar topology",
        },
        {
            "metric": "Actual attainable throughput",
            "reason": (
                "Pack structure plus machine assumptions yields a target, not "
                "implementation utilization or end-to-end overhead."
            ),
            "required": "measured implementation evidence",
            "formula": "observed TPS is compared independently below",
        },
    ]
    research_basis = [
        {
            "method": "Description length",
            "reference": "Shannon (1948), A Mathematical Theory of Communication",
            "supports": "entropy-coded grammar target",
        },
        {
            "method": "Parameter capacity",
            "reference": "Collins et al. (ICLR 2017), 3–6 task bits/parameter",
            "supports": "grammar-memory parameter interval",
        },
        {
            "method": "Capacity cross-check",
            "reference": "Morris et al. (2025), ≈3.6 bits/parameter for GPT-style models",
            "supports": "capacity prior lies inside an independent estimate",
        },
        {
            "method": "Parallel scheduling",
            "reference": "Brent (JACM 1974), work/span scheduling bound",
            "supports": "structural turn interval",
        },
        {
            "method": "Throughput identifiability",
            "reference": "Williams et al. (CACM 2009), Roofline model",
            "supports": "TPS requires compute and bandwidth rates",
        },
    ]

    return {
        "packs": list_packs(),
        "selected": pack.pack_id,
        "provenance": "source",
        "summary": {
            "backend": info.id,
            "backend_kind": info.kind,
            "description": info.description,
            "grammar_path": grammar_display_path,
            "production_count": len(productions),
            "recursive_nonterminals": recursive_nonterminals,
            "component_count": len(components),
            "structural_token_count": len(structural_tokens),
            "filled_slots": len(filled),
            "total_slots": len(all_fields),
        },
        "tiles": [
            {
                "label": "Backend",
                "value": info.id,
                "sub": info.kind,
                "accent": "moss",
            },
            {
                "label": "Productions",
                "value": str(len(productions)),
                "sub": f"{recursive_nonterminals} recursive nonterminal(s)",
                "accent": "",
            },
            {
                "label": "Components",
                "value": str(len(components)),
                "sub": "backend component symbols",
                "accent": "",
            },
            {
                "label": "Structural tokens",
                "value": str(len(structural_tokens)),
                "sub": "decoder-visible grammar surface",
                "accent": "",
            },
        ],
        "grammar": {
            "path": grammar_display_path,
            "source": grammar_source,
            "rows": [
                {"id": index, "run_id": line or " "}
                for index, line in enumerate(grammar_source.splitlines(), start=1)
            ],
        },
        "constraints": constraints,
        "templates": templates,
        "boundaries": boundaries,
        "calculated": {
            "tiles": calculated_tiles,
            "rows": calculated_rows,
            "basis": calculated_basis,
            "unavailable": unavailable_targets,
            "references": research_basis,
        },
        "planning": {
            "machine": machine,
            "assumptions": planning_assumptions,
            "harness_wall_seconds": MAX_HARNESS_WALL_SECONDS,
            "performance_target": performance_target,
        },
        "evidence": {
            "basis": (
                "Canonical committed/live scoreboards and current references. "
                "Evidence never feeds the target formula; it only reports the gap."
            ),
            "perf_provenance": perf_board.get("provenance"),
            "perf": perf_evidence,
            "smoke_provenance": quality_board.get("provenance"),
            "smoke": smoke_evidence,
            "checkpoints": checkpoint_references,
            "completion_kernel": completion_kernel,
            "completion_artifact": completion_artifact,
        },
    }


# --------------------------------------------------------------------------- #
# Read / observability
# --------------------------------------------------------------------------- #
@observability_router.get("/capabilities")
def capabilities(request: Request) -> dict[str, Any]:
    from slm_training.web.jobs import catalog

    caps = request.app.state.capabilities.to_dict()
    caps["jobs"] = catalog()
    caps["run_insights"] = {
        "browser": True,
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
    }
    hub = getattr(request.app.state, "otel", None)
    caps["otel"] = {
        "hub": bool(hub and hub.enabled),
        "peers_configured": bool(hub and hub.peers),
        "auth_mode": hub.auth_mode if hub else "open",
    }
    features = getattr(request.app.state, "features", None)
    caps["features"] = {
        "openfeature": bool(features),
        "provider": features.provider if features else None,
    }
    flag_client = getattr(request.app.state, "flag_client", None)
    caps["openfeature"] = {
        "enabled": flag_client is not None,
        "provider": getattr(flag_client, "provider_name", "InMemoryProvider"),
        "evaluate": "/api/flags/ofrep/v1/evaluate",
        "levers": "/api/flags/levers",
        "docs": "docs/design/openfeature-research-levers.md",
    }
    return caps


@observability_router.get("/features/bootstrap")
def features_bootstrap(
    request: Request,
    targeting_key: str = Query(default="anonymous", max_length=320),
) -> dict[str, Any]:
    features = getattr(request.app.state, "features", None)
    if features is None:
        raise HTTPException(status_code=503, detail="feature runtime unavailable")
    return features.bootstrap_payload(targeting_key=targeting_key)


@observability_router.get("/features/levers")
def features_levers() -> dict[str, Any]:
    return feature_flag_registry_payload()


@observability_router.get("/overview")
def overview(request: Request) -> dict[str, Any]:
    return _readers(request).overview()


@observability_router.get("/dsl-packs")
def dsl_packs(
    request: Request, pack: str = Query(default="openui", min_length=1, max_length=64)
) -> dict[str, Any]:
    try:
        return _dsl_pack_payload(request, pack)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@observability_router.get("/scoreboards")
def scoreboards(request: Request) -> dict[str, Any]:
    return {"scoreboards": _readers(request).scoreboards()}


@observability_router.get("/experiment-flags")
def experiment_flags(request: Request) -> dict[str, Any]:
    return _readers(request).experiment_flags()


@observability_router.get("/experiment-flags/{key}")
def experiment_flag(request: Request, key: str) -> dict[str, Any]:
    detail = _readers(request).experiment_flag(key)
    if detail is None:
        raise HTTPException(status_code=404, detail="unknown experiment feature flag")
    return detail


@observability_router.get("/scoreboards/{kind}")
def scoreboard(request: Request, kind: str) -> dict[str, Any]:
    board = _readers(request).scoreboard(kind)
    if board["provenance"] == "unknown":
        raise HTTPException(status_code=404, detail=f"unknown scoreboard: {kind}")
    return board


@observability_router.get("/runs")
def runs(request: Request) -> dict[str, Any]:
    return _readers(request).runs()


@observability_router.get("/runs/{run_id}")
def run(request: Request, run_id: str) -> dict[str, Any]:
    return _readers(request).run(run_id)


@observability_router.get("/runs/{run_id}/data")
def run_training_data(request: Request, run_id: str) -> dict[str, Any]:
    return _readers(request).run_training_data(run_id)


@observability_router.get("/runs/{run_id}/rl-traces")
def rl_traces(
    request: Request,
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    return _readers(request).rl_traces(run_id, offset=offset, limit=limit)


@observability_router.get("/lineage/champions")
def champions(request: Request) -> dict[str, Any]:
    return _readers(request).champions()


@observability_router.get("/lineage/deployments")
def deployments(request: Request) -> dict[str, Any]:
    return _readers(request).deployment_state()


@observability_router.get("/checkpoints")
def checkpoints(request: Request) -> dict[str, Any]:
    return _readers(request).checkpoints()


@observability_router.get("/checkpoints/{run_id}/gates")
def checkpoint_gates(request: Request, run_id: str) -> dict[str, Any]:
    return _readers(request).gates_for_run(run_id)


@observability_router.get("/data/train")
def data_train(
    request: Request, version: str | None = Query(default=None)
) -> dict[str, Any]:
    return _readers(request).train_data(version)


@observability_router.get("/data/train/{version}/records")
def data_train_records(
    request: Request,
    version: str,
    split: str | None = Query(default=None),
    source: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return _readers(request).train_records(
        version, split=split, source=source, query=q, offset=offset, limit=limit
    )


@observability_router.get("/data/train/{version}/quality")
def data_train_quality(request: Request, version: str) -> dict[str, Any]:
    return _readers(request).train_quality(version)


@observability_router.get("/data/train/{version}/rejected")
def data_train_rejected(
    request: Request,
    version: str,
    stage: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return _readers(request).train_rejected(
        version, stage=stage, offset=offset, limit=limit
    )


@observability_router.get("/data/preference")
def data_preference(request: Request) -> dict[str, Any]:
    return _readers(request).preference_data()


@observability_router.get("/data/test")
def data_test(request: Request) -> dict[str, Any]:
    return _readers(request).test_data()


@observability_router.get("/data/test/records")
def data_test_records(
    request: Request,
    suite: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return _readers(request).test_records(suite, query=q, offset=offset, limit=limit)


@observability_router.get("/annotations/summary")
def annotations_summary(request: Request) -> dict[str, Any]:
    return _readers(request).annotations_summary()


@observability_router.get("/comparisons/metrics")
def comparison_metrics(
    request: Request, candidate_run_id: str = Query(...)
) -> dict[str, Any]:
    return _readers(request).comparison_metrics(candidate_run_id)


@observability_router.get("/system")
def system(request: Request) -> dict[str, Any]:
    return _readers(request).system()


@observability_router.get("/dispatches")
def dispatches(request: Request) -> dict[str, Any]:
    return _readers(request).dispatches()


# --------------------------------------------------------------------------- #
# OTel hub: active runs across peers + lazy per-run event streams
# --------------------------------------------------------------------------- #
@observability_router.get("/otel/runs")
def otel_runs(request: Request, local: int = Query(default=0)) -> dict[str, Any]:
    # Sync handler on purpose: peer fetches are blocking urllib calls and must
    # run on the threadpool, never on the event loop. ``local=1`` is the
    # federation contract — peers only ever serve their own ingested state.
    return _otel(request).merged_runs(local_only=bool(local))


@observability_router.get("/otel/runs/{run_id}/events")
def otel_events(
    request: Request,
    run_id: str,
    since: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    return _otel(request).events(run_id, since=since, limit=limit)


@observability_router.get("/otel/runs/{run_id}/stream")
async def otel_stream(
    request: Request, run_id: str, since: int = Query(default=0, ge=0)
) -> StreamingResponse:
    hub = _otel(request)
    if hub.enabled and hub.has_local(run_id):
        generator = hub.stream(run_id, since=since)
    else:
        loop = asyncio.get_running_loop()
        peer = await loop.run_in_executor(None, hub.find_peer_for, run_id)
        if peer is not None:
            generator = hub.stream_remote(run_id, peer, since=since)
        elif hub.enabled:
            # Not seen anywhere yet: subscribe locally so a run that starts
            # after the viewer opens the page streams from its first event.
            generator = hub.stream(run_id, since=since)
        else:
            raise HTTPException(
                status_code=503,
                detail="otel streaming unavailable on this deployment",
            )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@otel_ingest_router.post("/v1/{signal}")
async def otel_ingest(signal: str, request: Request) -> dict[str, Any]:
    if signal not in {"traces", "logs"}:
        raise HTTPException(status_code=404, detail=f"unknown OTLP signal: {signal}")
    hub = _otel(request)
    if not hub.enabled:
        raise HTTPException(
            status_code=503, detail="otel hub disabled on this deployment"
        )
    ok, user = await hub.authorize(request.headers.get("authorization"))
    if not ok:
        raise HTTPException(status_code=401, detail="valid bearer token required")
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_INGEST_BYTES:
        raise HTTPException(status_code=413, detail="payload too large")
    # Stream with an incremental cap instead of request.body(): an oversized
    # or lying upload 413s at the threshold rather than buffering fully first.
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_INGEST_BYTES:
            raise HTTPException(status_code=413, detail="payload too large")
    try:
        payload = json.loads(bytes(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    return hub.ingest(signal, payload, user=user)


# --------------------------------------------------------------------------- #
# Pure-compute gate/promotion evaluation (read-only-safe; powers gate editor)
# --------------------------------------------------------------------------- #
class GateEvalRequest(BaseModel):
    suites: dict[str, dict[str, Any]]
    thresholds: dict[str, dict[str, float]] | None = None


@observability_router.get("/gates/policy")
def gates_policy() -> dict[str, Any]:
    return {"policy": DEFAULT_SHIP_GATES}


@observability_router.post("/gates/evaluate")
def gates_evaluate(payload: GateEvalRequest) -> dict[str, Any]:
    return evaluate_ship_gates(payload.suites, thresholds=payload.thresholds)


class PromotionEvalRequest(BaseModel):
    ship_suites: dict[str, dict[str, Any]] | None = None
    integrity: dict[str, Any] | None = None
    rankings: dict[str, list[str]] | None = None
    eg_time_by_seed: list[float] | None = None
    eg_params_by_seed: list[float] | None = None
    baseline_trainable_params: int | None = None
    candidate_trainable_params: int | None = None
    category_regression_tolerance: float = 0.02
    eg_time_lcb_min: float = 1.0
    eg_params_lcb_min: float = 1.0
    require_rank_stable_top2: bool = True
    campaign_manifest: dict[str, Any] | None = None
    campaign_result: dict[str, Any] | None = None
    campaign_store_root: Path | None = None
    campaign_artifact_root: Path | None = None
    verify_locked_manifest_digest: bool = False


@observability_router.post("/promotion/evaluate")
def promotion_evaluate(payload: PromotionEvalRequest) -> dict[str, Any]:
    criteria = PromotionCriteria(
        category_regression_tolerance=payload.category_regression_tolerance,
        require_rank_stable_top2=payload.require_rank_stable_top2,
        eg_time_lcb_min=payload.eg_time_lcb_min,
        eg_params_lcb_min=payload.eg_params_lcb_min,
    )
    campaign_store = None
    if (
        payload.campaign_manifest is not None
        and payload.campaign_store_root is not None
    ):
        from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
        from slm_training.autoresearch.storage import CampaignStore

        manifest = ExperimentCampaignV1.model_validate(payload.campaign_manifest)
        campaign_store = CampaignStore(
            manifest.campaign_id, payload.campaign_store_root
        )
    result = evaluate_promotion_engine(
        integrity=payload.integrity,
        rankings=payload.rankings,
        eg_time_by_seed=payload.eg_time_by_seed,
        eg_params_by_seed=payload.eg_params_by_seed,
        baseline_trainable_params=payload.baseline_trainable_params,
        candidate_trainable_params=payload.candidate_trainable_params,
        ship_suites=payload.ship_suites,
        criteria=criteria,
        hard_categories=_PROMOTION_HARD_CATEGORIES,
        gate_evaluator=lambda suites, policy: evaluate_ship_gates(
            suites, thresholds=policy or DEFAULT_SHIP_GATES
        ),
    )
    if payload.campaign_manifest is None or payload.campaign_result is None:
        governance_failures = ("campaign_governance_missing",)
        manifest_sha = None
    else:
        from slm_training.autoresearch.experiment_campaign import (
            CampaignResultV1,
            ExperimentCampaignV1,
            campaign_manifest_sha256,
            validate_result_claim,
        )

        manifest = ExperimentCampaignV1.model_validate(payload.campaign_manifest)
        governed_result = CampaignResultV1.model_validate(payload.campaign_result)
        locked_manifest_path = None
        if payload.verify_locked_manifest_digest:
            # SLM-431: deferred import -- see the identical note on
            # validate_result_claim's own locked_manifest_path import, this
            # module must not import slm_training.data.locked_eval_manifest
            # (or anything that transitively imports harnesses.experiments)
            # at module scope.
            from slm_training.data.locked_eval_manifest import canonical_manifest_path

            locked_manifest_path = canonical_manifest_path()
        governance_failures = validate_result_claim(
            manifest,
            governed_result,
            artifact_root=payload.campaign_artifact_root,
            locked_manifest_path=locked_manifest_path,
            baseline_trainable_params=payload.baseline_trainable_params,
            candidate_trainable_params=payload.candidate_trainable_params,
            eg_params_by_seed=payload.eg_params_by_seed,
        )
        if campaign_store is None or payload.campaign_artifact_root is None:
            governance_failures = (*governance_failures, "campaign_store_missing")
        else:
            governance_failures = (
                *governance_failures,
                *campaign_store.validate_campaign_result(
                    governed_result, artifact_root=payload.campaign_artifact_root
                ),
            )
        if governed_result.claim_class not in {"promotion_candidate", "ship_gate"}:
            governance_failures = (*governance_failures, "claim_class_not_promotable")
        governance_failures = tuple(dict.fromkeys(governance_failures))
        manifest_sha = campaign_manifest_sha256(manifest)
    result.setdefault("checks", {})["campaign_governance"] = {
        "pass": not governance_failures,
        "failures": list(governance_failures),
        "manifest_sha256": manifest_sha,
    }
    if not governance_failures and result.get("failures") == ["sufficient_evidence"]:
        result["checks"].pop("sufficient_evidence", None)
        result["checks"]["governed_campaign_evidence"] = {"pass": True}
        result["failures"] = []
        result["promotable"] = True
    if governance_failures:
        result["promotable"] = False
        result.setdefault("failures", []).extend(governance_failures)
    return result


# --------------------------------------------------------------------------- #
# Research OpenFeature / OFREP-shaped flag evaluation (read-only)
# --------------------------------------------------------------------------- #
class OfrepEvaluateRequest(BaseModel):
    context: dict[str, Any] = Field(default_factory=dict)
    flags: list[str] | None = None


@observability_router.get("/flags/levers")
def flags_levers() -> dict[str, Any]:
    from slm_training.flags.levers import LEVER_FLAGS

    return {
        "provider": "InMemoryProvider",
        "standard": "openfeature",
        "docs": "docs/design/openfeature-research-levers.md",
        "levers": [
            {
                "key": spec.key,
                "type": spec.value_type.value,
                "default": spec.default,
                "description": spec.description,
            }
            for spec in LEVER_FLAGS
        ],
    }


@observability_router.post("/flags/ofrep/v1/evaluate")
def flags_ofrep_evaluate(
    request: Request, payload: OfrepEvaluateRequest
) -> dict[str, Any]:
    from slm_training.flags.ofrep import evaluate_ofrep

    client = getattr(request.app.state, "flag_client", None)
    if client is None:
        from slm_training.flags import FlagClient, InMemoryProvider
        from slm_training.flags.in_memory import ruleset_from_defaults

        client = FlagClient(InMemoryProvider(ruleset_from_defaults()))
    return evaluate_ofrep(client, context_payload=payload.context, flags=payload.flags)


# --------------------------------------------------------------------------- #
# Action / exec (require execution)
# --------------------------------------------------------------------------- #
class JobRequest(BaseModel):
    job: str = Field(..., min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


@actions_router.post("/runs/{run_id}/insights")
def persist_run_insights(
    request: Request, run_id: str, payload: RunInsightSubmission
) -> dict[str, Any]:
    _require_execution(request)
    try:
        return _readers(request).save_run_insights(run_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail="run insights are not writable"
        ) from exc


@actions_router.post("/runs/{run_id}/insights/openai")
def enrich_run_insights_openai(request: Request, run_id: str) -> dict[str, Any]:
    _require_execution(request)
    try:
        return _readers(request).enrich_run_with_openai(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail="run insights are not writable"
        ) from exc


@actions_router.post("/jobs")
def submit_job(request: Request, payload: JobRequest) -> dict[str, Any]:
    registry = _require_execution(request)
    try:
        job = registry.submit(payload.job, payload.params)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown job: {payload.job}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return job.to_dict()


@actions_router.get("/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    caps = request.app.state.capabilities
    if not caps.execution:
        return {"execution": False, "jobs": []}
    return {
        "execution": True,
        "jobs": [j.to_dict() for j in request.app.state.jobs.list()],
    }


@actions_router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict[str, Any]:
    registry = _require_execution(request)
    job = registry.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {**job.to_dict(), "tail": registry.tail(job_id, lines=40)}


@actions_router.get("/jobs/{job_id}/logs")
def job_logs(request: Request, job_id: str) -> StreamingResponse:
    registry = _require_execution(request)
    if job_id not in registry.jobs:
        raise HTTPException(status_code=404, detail="unknown job")
    return StreamingResponse(
        registry.stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@actions_router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict[str, Any]:
    registry = _require_execution(request)
    if job_id not in registry.jobs:
        raise HTTPException(status_code=404, detail="unknown job")
    return registry.cancel(job_id).to_dict()


class ComparisonCreate(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    champion_run_id: str = Field(..., max_length=128)
    candidate_run_id: str = Field(..., max_length=128)
    champion_openui: str = Field(..., max_length=20000)
    candidate_openui: str = Field(..., max_length=20000)


@actions_router.post("/comparisons")
def create_comparison(request: Request, payload: ComparisonCreate) -> dict[str, Any]:
    _require_execution(request)
    return request.app.state.readers.comparisons.create(
        prompt=payload.prompt,
        champion_run_id=payload.champion_run_id,
        candidate_run_id=payload.candidate_run_id,
        champion_openui=payload.champion_openui,
        candidate_openui=payload.candidate_openui,
    )


class ComparisonVote(BaseModel):
    winner: str = Field(..., pattern="^(left|right|tie)$")
    reviewer_id: str = Field(default="anon", max_length=128)


@actions_router.post("/comparisons/{pair_id}/vote")
def vote_comparison(
    request: Request, pair_id: str, payload: ComparisonVote
) -> dict[str, Any]:
    _require_execution(request)
    try:
        return request.app.state.readers.comparisons.vote(
            pair_id,
            payload.winner,
            reviewer_id=payload.reviewer_id,  # type: ignore[arg-type]
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown comparison")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
