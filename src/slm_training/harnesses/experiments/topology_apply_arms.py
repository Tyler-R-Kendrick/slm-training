"""DSH3-16 topology-apply arm comparison (SLM-384).

Tests whether applying compiler-verified operator edits directly to typed
topology nodes (``dsl.operators.topology_apply``) reduces decode work at
matched quality, compared against two controls:

- ``RECOMPUTE_ONLY``: ordinary lowering; the typed topology tree is fully
  re-materialized from the authoritative new source after every edit.
- ``HIERARCHICAL_HEAD``: the SLM-383 hierarchical operator action head in its
  default ``off`` mode shadow-scores every non-singleton decision (two scorer
  forwards per decision), always DEFERs, and ordinary lowering applies --
  "hierarchical-head with ordinary lowering", measuring head overhead without
  letting it change the workload.
- ``TOPOLOGY_APPLY``: the same deterministic ordinary choice, but the verified
  ``ActionEffectV1`` transition is mapped to a topology expand/contract/move/
  rewrite edit and applied directly to the typed nodes; exact fallback
  (non-exact coverage, unmapped operator) defers to ordinary lowering.

Workload contract: every arm executes the *identical* deterministic action
sequence per request (forced singletons commit with zero model forwards in
every arm; otherwise the first legal action -- semantic-id order -- whose
result state was not already visited in this request), so the legal terminal
set and per-step typed trees must be identical across arms. That is the
matched-quality bar; work counters (active nodes, node passes, remask/rewrite
phases, model forwards, compiler expansions, verifier calls, latency, memory)
are the measured difference.

Scope: fixture/wiring-scale evidence (CPU, seconds, small deterministic
request set). Node-pass counters model the production decode path; the
harness additionally re-materializes authoritative trees every step to verify
matched quality, so wall-clock here is conservative *against* topology-apply
and is never reported as an inference-speed claim.
"""

from __future__ import annotations

import hashlib
import resource
import time
from dataclasses import replace
from enum import Enum
from typing import Any, Sequence

import torch

from slm_training.dsl.operators import (
    ApplicationProvenanceV1,
    OpenUITemplateAliasV1,
    OperatorStateV1,
    branch_fingerprint,
    build_openui_topology_operator_context,
    build_openui_topology_operator_library,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.hierarchical_head import (
    HierarchicalDecisionKind,
    HierarchicalOperatorActionHead,
    HierarchicalOperatorHeadConfigV1,
    build_operator_action_features,
)
from slm_training.dsl.operators.topology_apply import (
    TopologyApplyConfigV1,
    TopologyApplySession,
    materialize_topology_tree,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.production_codec import parse_statement_bindings

__all__ = [
    "TopologyApplyArm",
    "TopologyApplyRequestV1",
    "build_fixture_requests",
    "run_topology_apply_arms",
]


class TopologyApplyArm(str, Enum):
    RECOMPUTE_ONLY = "RECOMPUTE_ONLY"
    HIERARCHICAL_HEAD = "HIERARCHICAL_HEAD"
    TOPOLOGY_APPLY = "TOPOLOGY_APPLY"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _FixtureAlias:
    """Small builders mirroring tests/test_dsl/test_topology_operators.py."""

    EXPANDED = {
        "type": "element",
        "typeName": "Card",
        "props": {
            "children": [
                {
                    "type": "element",
                    "typeName": "TextContent",
                    "props": {"text": ":expanded.value"},
                }
            ]
        },
    }
    COMPACT = {
        "type": "element",
        "typeName": "TextContent",
        "props": {"text": ":compact.value"},
    }
    WRAPPER = {
        "type": "element",
        "typeName": "Stack",
        "props": {"children": [], "direction": "column"},
    }

    @classmethod
    def alias(
        cls,
        expanded: dict[str, Any],
        contracted: dict[str, Any],
        *,
        child_role: str | None = None,
        digest: str,
    ) -> OpenUITemplateAliasV1:
        return OpenUITemplateAliasV1(
            pack_id="openui",
            expanded=expanded,
            contracted=contracted,
            source_artifact_digest=digest,
            child_role=child_role,
        )


class TopologyApplyRequestV1:
    """One deterministic fixture request: seed source plus pack-owned aliases."""

    def __init__(
        self,
        *,
        request_id: str,
        source: str,
        template_aliases: tuple[OpenUITemplateAliasV1, ...] = (),
        values: tuple[Any, ...] = (),
        seed: int = 11,
    ) -> None:
        self.request_id = request_id
        self.source = source
        self.template_aliases = template_aliases
        self.values = values
        self.seed = seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source": self.source,
            "template_aliases": [a.fingerprint for a in self.template_aliases],
            "values": [repr(v) for v in self.values],
            "seed": self.seed,
        }


def build_fixture_requests() -> tuple[TopologyApplyRequestV1, ...]:
    """Small deterministic request set exercising move/expand/contract/rewrite."""
    expand_alias = _FixtureAlias.alias(
        _FixtureAlias.EXPANDED, _FixtureAlias.COMPACT, digest="e" * 64
    )
    wrap_alias = _FixtureAlias.alias(
        _FixtureAlias.WRAPPER,
        _FixtureAlias.COMPACT,
        child_role="children",
        digest="a" * 64,
    )
    return (
        TopologyApplyRequestV1(
            request_id="dsh3-16-singleton",
            source='root = Card([], "clear")',
            values=("scroll",),
            seed=7,
        ),
        TopologyApplyRequestV1(
            request_id="dsh3-16-move",
            source='root = Card([TextContent(":source.a"), TextContent(":source.b")], "clear")',
            values=("clear", "scroll", [1, 0]),
            seed=11,
        ),
        TopologyApplyRequestV1(
            request_id="dsh3-16-expand",
            source='root = TextContent(":compact.value")',
            template_aliases=(expand_alias,),
            values=(),
            seed=23,
        ),
        TopologyApplyRequestV1(
            request_id="dsh3-16-contract",
            source='root = Card([TextContent(":expanded.value")])',
            template_aliases=(expand_alias, wrap_alias),
            values=(),
            seed=37,
        ),
    )


def _new_counters() -> dict[str, int]:
    return {
        "model_forwards": 0,
        "compiler_expansions": 0,
        "verifier_calls": 0,
        "singleton_bypasses": 0,
        "head_defers": 0,
    }


def _run_request(
    spec: TopologyApplyRequestV1,
    arm: TopologyApplyArm,
    *,
    max_steps: int,
    head_d_model: int,
    head_hidden_dim: int,
    max_combinations_per_operator: int,
) -> dict[str, Any]:
    base = get_pack("openui")
    state = OperatorStateV1.from_source(base, spec.source)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="dsh3-16-topology-apply-arms",
        compiler_version="v1",
        source_artifact_digest=_sha(spec.source),
        request_id=spec.request_id,
    )
    head = (
        HierarchicalOperatorActionHead(
            HierarchicalOperatorHeadConfigV1(
                mode="off", d_model=head_d_model, hidden_dim=head_hidden_dim
            )
        )
        if arm is TopologyApplyArm.HIERARCHICAL_HEAD
        else None
    )
    session = TopologyApplySession(
        TopologyApplyConfigV1(
            mode="topology_apply" if arm is TopologyApplyArm.TOPOLOGY_APPLY else "off"
        ),
        request_id=spec.request_id,
        branch_digest=branch_fingerprint(state.state_digest, _sha(spec.request_id)),
    )
    counters = _new_counters()
    visited = {state.state_digest}
    bindings = parse_statement_bindings(state.source, dsl="openui")
    session.begin(state, bindings)

    steps = 0
    matched_steps = 0
    action_sequence: list[str] = []
    edit_kinds: dict[str, int] = {}
    for _ in range(max_steps):
        branch = branch_fingerprint(state.state_digest, _sha(spec.request_id))
        context = build_openui_topology_operator_context(
            base,
            state,
            request_id=spec.request_id,
            branch_digest=branch,
            seed=spec.seed,
            template_aliases=spec.template_aliases,
            values=spec.values,
        )
        library = build_openui_topology_operator_library(context)
        pack = replace(base, operator_library=library)
        legal_set = enumerate_operator_legal_set(
            pack=pack,
            library=library,
            state=state,
            reference_table=context.local.reference_table,
            provenance=provenance,
            max_combinations_per_operator=max_combinations_per_operator,
        )
        counters["compiler_expansions"] += sum(
            entry.evaluated_combinations for entry in legal_set.entries
        )
        candidates = sorted(
            legal_set.operator_actions, key=lambda action: action.semantic_id
        )
        if not candidates:
            break

        forced = legal_set.forced_action
        if forced is not None:
            counters["singleton_bypasses"] += 1
            chosen = next(a for a in candidates if a.serialized == forced)
            result = library.apply(
                pack, state, chosen.operator_id, chosen.arguments, provenance
            )
            counters["compiler_expansions"] += 1
            counters["verifier_calls"] += 1
            if not result.succeeded:
                break
        else:
            if head is not None:
                features = build_operator_action_features(
                    legal_set=legal_set,
                    library=library,
                    pack=pack,
                    state=state,
                    provenance=provenance,
                )
                counters["compiler_expansions"] += len(features)
                decision = head.decide(
                    torch.zeros(head_d_model), legal_set, features
                )
                counters["model_forwards"] += 2  # stage-one + stage-two scorer
                if decision.decision is not HierarchicalDecisionKind.DEFER:
                    raise AssertionError("default-off head must always DEFER")
                counters["head_defers"] += 1
            chosen = None
            result = None
            for candidate in candidates:
                probe = library.apply(
                    pack, state, candidate.operator_id, candidate.arguments, provenance
                )
                counters["compiler_expansions"] += 1
                counters["verifier_calls"] += 1
                if probe.succeeded and probe.state is not None and (
                    probe.state.state_digest not in visited
                ):
                    chosen = candidate
                    result = probe
                    break
            if chosen is None or result is None or result.state is None:
                break

        visited.add(result.state.state_digest)
        new_bindings = parse_statement_bindings(result.state.source, dsl="openui")
        if arm is TopologyApplyArm.TOPOLOGY_APPLY:
            direct = session.apply_direct(
                result=result,
                action=chosen,
                legal_set=legal_set,
                context=context,
                new_bindings=new_bindings,
            )
            if direct:
                kind = session.edits[-1].kind.value
                edit_kinds[kind] = edit_kinds.get(kind, 0) + 1
        else:
            session.begin(result.state, new_bindings)
        # Matched-quality check: the session tree must equal the authoritative
        # full materialization of the compiler-validated new source, every step.
        _, authoritative_roots, _ = materialize_topology_tree(new_bindings)
        if authoritative_roots != session.roots:
            raise AssertionError("typed topology tree drifted from authoritative source")
        matched_steps += 1
        action_sequence.append(chosen.application_id)
        state = result.state
        steps += 1

    return {
        "request_id": spec.request_id,
        "steps": steps,
        "matched_steps": matched_steps,
        "terminal_source": state.source,
        "terminal_digest": state.state_digest,
        "action_sequence": action_sequence,
        "edit_kinds": edit_kinds,
        **session.stats.to_dict(),
        **counters,
    }


def _totals(runs: Sequence[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "steps",
        "matched_steps",
        "active_nodes",
        "node_passes",
        "remask_phases",
        "rewrite_phases",
        "model_forwards",
        "compiler_expansions",
        "verifier_calls",
        "singleton_bypasses",
        "direct_applies",
        "deferred_applies",
        "head_defers",
    )
    return {key: sum(int(run[key]) for run in runs) for key in keys}


def run_topology_apply_arms(
    *,
    requests: Sequence[TopologyApplyRequestV1] | None = None,
    max_steps: int = 8,
    head_d_model: int = 16,
    head_hidden_dim: int = 16,
    max_combinations_per_operator: int = 128,
) -> dict[str, Any]:
    """Run the three matched arms and publish the execution-sparsity disposition."""
    specs = tuple(requests) if requests is not None else build_fixture_requests()
    if not specs:
        raise ValueError("topology-apply arm comparison requires requests")

    per_arm: dict[str, Any] = {}
    for arm in TopologyApplyArm:
        started = time.perf_counter()
        peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        runs = [
            _run_request(
                spec,
                arm,
                max_steps=max_steps,
                head_d_model=head_d_model,
                head_hidden_dim=head_hidden_dim,
                max_combinations_per_operator=max_combinations_per_operator,
            )
            for spec in specs
        ]
        peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        per_arm[arm.value] = {
            "requests": runs,
            "totals": _totals(runs),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": max(peak_before, peak_after) * 1024,
        }

    terminal_sets = {
        arm: {run["request_id"]: run["terminal_digest"] for run in data["requests"]}
        for arm, data in per_arm.items()
    }
    terminal_sets_identical = (
        terminal_sets[TopologyApplyArm.RECOMPUTE_ONLY.value]
        == terminal_sets[TopologyApplyArm.HIERARCHICAL_HEAD.value]
        == terminal_sets[TopologyApplyArm.TOPOLOGY_APPLY.value]
    )
    sequences = {
        arm: [run["action_sequence"] for run in data["requests"]]
        for arm, data in per_arm.items()
    }
    workload_identical = (
        sequences[TopologyApplyArm.RECOMPUTE_ONLY.value]
        == sequences[TopologyApplyArm.HIERARCHICAL_HEAD.value]
        == sequences[TopologyApplyArm.TOPOLOGY_APPLY.value]
    )
    topology = per_arm[TopologyApplyArm.TOPOLOGY_APPLY.value]["totals"]
    recompute = per_arm[TopologyApplyArm.RECOMPUTE_ONLY.value]["totals"]
    head = per_arm[TopologyApplyArm.HIERARCHICAL_HEAD.value]["totals"]
    every_step_matched = all(
        run["matched_steps"] == run["steps"]
        for data in per_arm.values()
        for run in data["requests"]
    )
    matched_quality = bool(
        terminal_sets_identical and workload_identical and every_step_matched
    )
    node_pass_improvement = bool(
        matched_quality
        and topology["steps"] > 0
        and topology["direct_applies"] > 0
        and topology["node_passes"] < recompute["node_passes"]
    )
    acceptance = {
        "legal_terminal_set_unchanged": terminal_sets_identical,
        "matched_workload_identical_action_sequences": workload_identical,
        "matched_quality_every_step": every_step_matched,
        "exact_singleton_authority_preserved": (
            topology["singleton_bypasses"] == recompute["singleton_bypasses"]
            == head["singleton_bypasses"]
        ),
        "head_always_defers_off_mode": (
            head["head_defers"] * 2 == head["model_forwards"]
            and head["head_defers"] >= head["steps"] - head["singleton_bypasses"]
        ),
        "zero_model_forwards_without_head": (
            topology["model_forwards"] == 0 and recompute["model_forwards"] == 0
        ),
        "node_pass_improvement_at_matched_quality": node_pass_improvement,
    }
    disposition = (
        "execution_sparsity_supported"
        if node_pass_improvement
        else "execution_sparsity_rejected"
    )
    return {
        "schema": "topology_apply_arms/v1",
        "issue": "SLM-384",
        "experiment_id": "DSH3-16",
        "requests": [spec.to_dict() for spec in specs],
        "max_steps": max_steps,
        "max_combinations_per_operator": max_combinations_per_operator,
        "enumeration_budget_note": (
            "Operator legal-set enumeration is budget-capped; truncated "
            "operators stay UNKNOWN (never UNSUPPORTED) and forced-singleton "
            "detection requires COMPLETE coverage, exactly as in "
            "enumerate_operator_legal_set."
        ),
        "arms": per_arm,
        "acceptance": acceptance,
        "disposition": disposition,
        "evidence_tier": "fixture_wiring",
        "systems_claim": False,
        "scope_note": (
            "Fixture/wiring-scale arm comparison (CPU, deterministic small "
            "request set). Node-pass counters model the production decode "
            "path; the harness re-materializes authoritative trees every step "
            "to certify matched quality, so arm wall-clock here is "
            "conservative against topology-apply. No wall-clock or systems "
            "claim is made; serialized target compression is not reported as "
            "inference speed."
        ),
    }
