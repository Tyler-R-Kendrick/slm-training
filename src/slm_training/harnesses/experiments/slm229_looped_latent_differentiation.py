"""SLM-229 (RSC0-01): looped-latent differentiation and authorization audit.

Docs/spec harness. Zero-compute research audit: no model/data implementation,
no training run, no checkpoint. Compares a proposed LOTUS-style ("looped
transformer reasoning", arXiv:2606.31779) minimal compiler-latent probe
against the completed SLM-138/139 shared recursive denoiser and the
SLM-144/145/146/160 SemanticPlanV1 program, and produces a machine-readable
``LoopedLatentDifferentiationV1`` authorization record plus a Markdown memo.

Per the issue text, a negative/blocked verdict is an explicitly successful
closeout: this module never manufactures an ``authorize_minimal_probe``
result to look productive. ``MinimalCompilerLatentContractV1`` (the narrow
probe schema) is defined here as a dataclass but is only *populated* in the
report when the verdict is ``authorize_minimal_probe``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.versioning import UNKNOWN, build_version_stamp, git_commit
from slm_training.harnesses.experiments.semantic_floor_gate import (
    DEFAULT_GATE_PATH,
    load_semantic_floor_gate,
    require_floor_gate,
)

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "CONTRACT_SCHEMA",
    "LoopedLatentVerdict",
    "MechanismComparisonRow",
    "TargetSupportRow",
    "OracleInterventionCeiling",
    "ScaleRegimeAudit",
    "PriorArtRow",
    "DifferentiatorVerdict",
    "MinimalCompilerLatentContractV1",
    "LoopedLatentDifferentiationReport",
    "differentiator_contract_field_map",
    "build_mechanism_comparison",
    "build_target_support_audit",
    "build_oracle_intervention_ceiling",
    "build_scale_regime_audit",
    "build_prior_art_audit",
    "build_differentiators",
    "evaluate_verdict",
    "validate_doc_refs",
    "run_differentiation_audit",
    "render_markdown",
]

MATRIX_VERSION = "rsc0-01-v1"
MATRIX_SET = "slm229_looped_latent_differentiation"
EXPERIMENT_ID = "slm229-looped-latent-differentiation"
REPORT_SCHEMA = "LoopedLatentDifferentiationV1"
CONTRACT_SCHEMA = "MinimalCompilerLatentContractV1"


class LoopedLatentVerdict(str, Enum):
    """Authorization outcomes for the proposed looped-latent probe."""

    AUTHORIZE_MINIMAL_PROBE = "authorize_minimal_probe"
    DUPLICATE_SPV = "duplicate_spv"
    UNSUPPORTED_TARGETS = "unsupported_targets"
    SCALE_NOT_IDENTIFIABLE = "scale_not_identifiable"
    BLOCKED_BY_FLOOR = "blocked_by_floor"
    BLOCKED_BY_RECURRENCE = "blocked_by_recurrence"
    INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Section 1: mechanism comparison table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismComparisonRow:
    """One row of the LOTUS/SLM-138/144/145/146/160/proposed comparison table."""

    mechanism_id: str
    state_location_shape: str
    weight_sharing_recurrence: str
    supervision_target_vocab: str
    readout_head_sharing: str
    downstream_consumption: str
    structure_handling: str
    inference_inputs_leakage: str
    compiler_verifier_authority: str
    causal_intervention_evidence: str
    param_flop_byte_impact: str
    evidence_verdict: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MechanismComparisonRow":
        return cls(**{f.name: data.get(f.name, "") for f in fields(cls)})


# ---------------------------------------------------------------------------
# Section 2: data / target support audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSupportRow:
    """Target/data availability audit row for one candidate slot kind."""

    target_id: str
    deterministic_extraction: str
    accepted_set_multi_label: str
    ambiguous_unknown_handling: str
    split_leakage_fingerprint: str
    id_mapping: str
    counts_by_family: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetSupportRow":
        return cls(**{f.name: data.get(f.name, "") for f in fields(cls)})


# ---------------------------------------------------------------------------
# Section 3: oracle intervention ceiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleInterventionCeiling:
    """Reused-evidence audit for gold-substitution / intervention ceilings."""

    reused_evidence_paths: tuple[str, ...]
    reused_evidence_applicable: bool
    gap_description: str
    smallest_fixture_spec: str

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["reused_evidence_paths"] = list(self.reused_evidence_paths)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OracleInterventionCeiling":
        return cls(
            reused_evidence_paths=tuple(data.get("reused_evidence_paths", [])),
            reused_evidence_applicable=bool(data.get("reused_evidence_applicable", False)),
            gap_description=data.get("gap_description", ""),
            smallest_fixture_spec=data.get("smallest_fixture_spec", ""),
        )


# ---------------------------------------------------------------------------
# Section 4: scale / regime audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleRegimeAudit:
    """Scale/regime audit; floor and recurrence status fields are mandatory."""

    lotus_scale_notes: str
    current_model_params_notes: str
    target_tokens_decisions_notes: str
    unique_records_steps_notes: str
    semantic_floor_status: str
    recursive_regime_status: str
    identifiability_verdict: str
    expected_extra_cost_notes: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScaleRegimeAudit":
        return cls(**{f.name: data.get(f.name, "") for f in fields(cls)})


# ---------------------------------------------------------------------------
# Section 5: prior-art / novelty audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorArtRow:
    """One prior-art/novelty audit row."""

    topic: str
    internal_hits: str
    external_verification_note: str
    novelty_scope: str  # "repository" | "external" | "none_claimed"
    novelty_notes: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorArtRow":
        return cls(**{f.name: data.get(f.name, "") for f in fields(cls)})


# ---------------------------------------------------------------------------
# Candidate differentiators (all 7, per the issue text)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DifferentiatorVerdict:
    """One of the seven candidate differentiators, explicit true/false + evidence."""

    differentiator_id: int
    name: str
    satisfied: bool
    evidence: str
    contract_field: str
    test_ref: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DifferentiatorVerdict":
        return cls(
            differentiator_id=int(data.get("differentiator_id", 0)),
            name=data.get("name", ""),
            satisfied=bool(data.get("satisfied", False)),
            evidence=data.get("evidence", ""),
            contract_field=data.get("contract_field", ""),
            test_ref=data.get("test_ref", ""),
        )


# ---------------------------------------------------------------------------
# MinimalCompilerLatentContractV1: schema only, populated iff authorized
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinimalCompilerLatentContractV1:
    """Schema-only contract for the narrow probe. NOT an implementation.

    Populated in :class:`LoopedLatentDifferentiationReport` only when
    ``verdict == LoopedLatentVerdict.AUTHORIZE_MINIMAL_PROBE``.
    """

    contract_schema: str
    slot_kinds: tuple[str, ...]
    slot_count_k: int
    slot_shape: str
    context_surface_inputs: tuple[str, ...]
    shared_readout_ids: str
    target_representation: str
    surface_consumer_mechanism: str
    recurrence_update_reset_semantics: str
    loss_normalization_and_output_coupling: str
    interventions: tuple[str, ...]
    compiler_verifier_authority_boundary: str
    checkpoint_config_identity: str
    default_off: bool
    required_floor_gate: str
    required_recurrence_gate: str
    required_oracle_gate: str
    primary_metrics: tuple[str, ...]
    falsifier: str
    stop_rules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        for key in (
            "slot_kinds",
            "context_surface_inputs",
            "interventions",
            "primary_metrics",
            "stop_rules",
        ):
            data[key] = list(getattr(self, key))
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MinimalCompilerLatentContractV1":
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            value = data.get(f.name)
            if f.name in (
                "slot_kinds",
                "context_surface_inputs",
                "interventions",
                "primary_metrics",
                "stop_rules",
            ):
                kwargs[f.name] = tuple(value or ())
            else:
                kwargs[f.name] = value
        return cls(**kwargs)


def differentiator_contract_field_map() -> dict[int, str]:
    """Map each of the 7 candidate differentiators to a concrete contract field.

    Every claimed differentiator must map to a field that actually exists on
    :class:`MinimalCompilerLatentContractV1` (asserted by the unit tests).
    """
    return {
        1: "slot_kinds",  # internal workspace: slot kinds live in the contract, not SemanticPlanV1
        2: "shared_readout_ids",  # shared decision/output geometry
        3: "surface_consumer_mechanism",  # direct consumer (cross-attn/gating, not logging)
        4: "slot_kinds",  # minimal scope: only root_contract + component_inventory kinds allowed
        5: "interventions",  # built-in gold/zero/swap/wrong/detached interventions
        6: "compiler_verifier_authority_boundary",  # no hard authority
        7: "required_recurrence_gate",  # conditional execution: floor + recurrence gates
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopedLatentDifferentiationReport:
    """Full ``LoopedLatentDifferentiationV1`` authorization record."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    source_commit: str
    evidence_cutoff: str
    reviewed_refs: tuple[str, ...]
    generated_at: str
    floor_gate_ref: str
    floor_gate_hash: str
    floor_gate_verdict: str
    mechanism_comparison: tuple[MechanismComparisonRow, ...]
    target_support_audit: tuple[TargetSupportRow, ...]
    oracle_intervention_ceiling: OracleInterventionCeiling
    scale_regime_audit: ScaleRegimeAudit
    prior_art_audit: tuple[PriorArtRow, ...]
    differentiators: tuple[DifferentiatorVerdict, ...]
    verdict: LoopedLatentVerdict
    allowed_implementation_scope: str
    forbidden_duplicate_scope: str
    resolving_evidence: str
    contract_hash: str | None
    minimal_contract: MinimalCompilerLatentContractV1 | None
    version_stamp: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "source_commit": self.source_commit,
            "evidence_cutoff": self.evidence_cutoff,
            "reviewed_refs": list(self.reviewed_refs),
            "generated_at": self.generated_at,
            "floor_gate_ref": self.floor_gate_ref,
            "floor_gate_hash": self.floor_gate_hash,
            "floor_gate_verdict": self.floor_gate_verdict,
            "mechanism_comparison": [m.to_dict() for m in self.mechanism_comparison],
            "target_support_audit": [t.to_dict() for t in self.target_support_audit],
            "oracle_intervention_ceiling": self.oracle_intervention_ceiling.to_dict(),
            "scale_regime_audit": self.scale_regime_audit.to_dict(),
            "prior_art_audit": [p.to_dict() for p in self.prior_art_audit],
            "differentiators": [d.to_dict() for d in self.differentiators],
            "verdict": self.verdict.value,
            "allowed_implementation_scope": self.allowed_implementation_scope,
            "forbidden_duplicate_scope": self.forbidden_duplicate_scope,
            "resolving_evidence": self.resolving_evidence,
            "contract_hash": self.contract_hash,
            "minimal_contract": (
                self.minimal_contract.to_dict() if self.minimal_contract is not None else None
            ),
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopedLatentDifferentiationReport":
        contract_data = data.get("minimal_contract")
        return cls(
            schema=data.get("schema", REPORT_SCHEMA),
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm229_differentiation"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            source_commit=data.get("source_commit", UNKNOWN),
            evidence_cutoff=data.get("evidence_cutoff", UNKNOWN),
            reviewed_refs=tuple(data.get("reviewed_refs", [])),
            generated_at=data.get("generated_at", ""),
            floor_gate_ref=data.get("floor_gate_ref", DEFAULT_GATE_PATH),
            floor_gate_hash=data.get("floor_gate_hash", ""),
            floor_gate_verdict=data.get("floor_gate_verdict", "inconclusive"),
            mechanism_comparison=tuple(
                MechanismComparisonRow.from_dict(m) for m in data.get("mechanism_comparison", [])
            ),
            target_support_audit=tuple(
                TargetSupportRow.from_dict(t) for t in data.get("target_support_audit", [])
            ),
            oracle_intervention_ceiling=OracleInterventionCeiling.from_dict(
                data.get("oracle_intervention_ceiling", {})
            ),
            scale_regime_audit=ScaleRegimeAudit.from_dict(data.get("scale_regime_audit", {})),
            prior_art_audit=tuple(
                PriorArtRow.from_dict(p) for p in data.get("prior_art_audit", [])
            ),
            differentiators=tuple(
                DifferentiatorVerdict.from_dict(d) for d in data.get("differentiators", [])
            ),
            verdict=LoopedLatentVerdict(data.get("verdict", "inconclusive")),
            allowed_implementation_scope=data.get("allowed_implementation_scope", ""),
            forbidden_duplicate_scope=data.get("forbidden_duplicate_scope", ""),
            resolving_evidence=data.get("resolving_evidence", ""),
            contract_hash=data.get("contract_hash"),
            minimal_contract=(
                MinimalCompilerLatentContractV1.from_dict(contract_data)
                if contract_data is not None
                else None
            ),
            version_stamp=dict(data.get("version_stamp", {})),
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Content builders (the actual grounded analysis)
# ---------------------------------------------------------------------------

REVIEWED_REFS: tuple[str, ...] = (
    "SLM-138",
    "SLM-139",
    "SLM-144",
    "SLM-145",
    "SLM-146",
    "SLM-160",
    DEFAULT_GATE_PATH,
    "docs/design/research-lineage.md",
    "docs/design/iter-slm138-recursive-denoiser-20260720.md",
    "docs/design/iter-slm138-recursive-denoiser-20260720.json",
    "docs/design/iter-slm139-stochastic-recursive-width-20260720.md",
    "docs/design/iter-slm139-stochastic-recursive-width-20260720.json",
    "docs/design/iter-slm144-plan-predictor-20260720.md",
    "docs/design/iter-slm144-plan-predictor-20260720.json",
    "docs/design/iter-slm145-plan-predictor-factors-20260720.md",
    "docs/design/iter-slm145-plan-predictor-factors-20260720.json",
    "docs/design/iter-slm146-semantic-plan-compiler-20260720.md",
    "docs/design/iter-slm146-semantic-plan-compiler-20260720.json",
    "docs/design/iter-slm160-spv-disposition-20260720.md",
    "docs/design/iter-slm160-spv-disposition-20260720.json",
    "docs/design/semantic-planning-valid-state.md",
    "docs/design/semantic-planning-valid-state-disposition.md",
    "src/slm_training/models/recursive_denoiser.py",
    "src/slm_training/data/progspec/semantic_plan.py",
    "src/slm_training/models/semantic_plan_predictor.py",
    "src/slm_training/data/semantic_plan/compiler.py",
    "src/slm_training/data/semantic_plan/seed.py",
)


def build_mechanism_comparison() -> list[MechanismComparisonRow]:
    """Return the preregistered mechanism comparison table (7 rows)."""
    not_verified = (
        "Not independently verified in this environment (no live web access); "
        "based only on the title/framing given in the SLM-229 issue text "
        "('looped transformer reasoning', arXiv:2606.31779) and the Universal "
        "Transformer lineage (arXiv:1807.03819) already cited for SLM-138. No "
        "specific quantitative claim from the LOTUS paper is asserted here."
    )
    return [
        MechanismComparisonRow(
            mechanism_id="lotus_arxiv_2606_31779",
            state_location_shape=not_verified,
            weight_sharing_recurrence=(
                "Presumed recurrent/looped application of shared transformer "
                "weights over reasoning depth, by analogy to Universal "
                "Transformer weight-tying (arXiv:1807.03819). Exact recurrence "
                "operator not independently verified in this environment."
            ),
            supervision_target_vocab=not_verified,
            readout_head_sharing=not_verified,
            downstream_consumption=not_verified,
            structure_handling=not_verified,
            inference_inputs_leakage=not_verified,
            compiler_verifier_authority=(
                "No compiler/verifier authority concept is expected to apply; "
                "LOTUS targets natural-language reasoning, not a grammar-"
                "constrained DSL decoder."
            ),
            causal_intervention_evidence=not_verified,
            param_flop_byte_impact=not_verified,
            evidence_verdict=(
                "External paper reference only; no code in this repo. Zero "
                "repo-wide hits for '2606.31779' or 'LOTUS' outside this memo "
                "at audit time (grep of docs/ and src/)."
            ),
        ),
        MechanismComparisonRow(
            mechanism_id="slm138_shared_recursive_denoiser",
            state_location_shape=(
                "y (token/decoding stream) and z (auxiliary latent stream) both "
                "live per-position inside SharedRecursiveDenoiserTower "
                "(src/slm_training/models/recursive_denoiser.py); toy fixture "
                "forward shape [batch=2, seq=6, d_model=32]. z_0 = learned "
                "nn.Parameter(max_len, d_model) + projected pooled context + "
                "position."
            ),
            weight_sharing_recurrence=(
                "Single shared TransformerBlock pool (recursive_transition_layers) "
                "split by index into F_theta/G_theta, reused by object identity "
                "every recursion step; SLM-138 fixture confirms "
                "f_layer_object_count=1, g_layer_object_count=1, "
                "total_shared_layers=2. Recurrence: z_r = z_{r-1} + "
                "F_theta(norm(z_{r-1}+y_{r-1}), context); y_r = y_{r-1} + "
                "G_theta(norm(y_{r-1}+z_r), context); depth R = "
                "recursive_steps."
            ),
            supervision_target_vocab=(
                "Per-depth surface-token deep-supervision CE only "
                "(recursive_depth_loss_0=26.038, recursive_depth_loss_1=23.954, "
                "recursive_depth_supervision_loss=33.328 in the toy fixture). z "
                "itself carries no defined target vocabulary."
            ),
            readout_head_sharing=(
                "Same lm_head/project() used at every depth "
                "(depth_logits.append(self.project(h))) — a single shared "
                "vocab projection tied to token embeddings, not a separate "
                "per-depth head."
            ),
            downstream_consumption=(
                "z is read by y every recursion step through the existing "
                "G_theta cross-attention/gating path; only the final-step "
                "y/logits are exposed by forward()/encode() — "
                "recursive_outputs() exposes per-depth hiddens/logits "
                "internally but no plan-like artifact is exported."
            ),
            structure_handling=(
                "N/A — no explicit ordered/set/partial-order/UNKNOWN target "
                "representation; deep supervision is per-token CE only."
            ),
            inference_inputs_leakage=(
                "Same prompt/context inputs at train and eval; no additional "
                "inference-only channel."
            ),
            compiler_verifier_authority="None — architecture-only; no legal-action interaction.",
            causal_intervention_evidence=(
                "None run — SLM-138 is wiring-only (forward pass + checkpoint "
                "round-trip). No gold/zero/swap ablation on z exists."
            ),
            param_flop_byte_impact=(
                "stacked_params=64994 vs recursive_params=74242 on the toy "
                "fixture (json); no GPU or production-scale measurement."
            ),
            evidence_verdict=(
                "status: wiring_only (iter-slm138 json/md). SLM-139 closed the "
                "follow-on stochastic-width campaign as "
                "'no_supported_probabilistic_regime' because "
                "gate_1_recursive_base failed: no recursive_core_positive "
                "verdict was ever produced."
            ),
        ),
        MechanismComparisonRow(
            mechanism_id="slm144_plan_predictor_archetype_roleset",
            state_location_shape=(
                "External — a component-family count vector is fed to standalone "
                "heads outside the denoiser (src/slm_training/models/"
                "semantic_plan_predictor.py); output is a predicted "
                "PlanArchetype/RoleSlot object, not an internal per-step latent."
            ),
            weight_sharing_recurrence=(
                "Independent ArchetypeClassifierHead (MLP) and "
                "RoleSetPredictorHead (bipartite-matching) modules; no shared "
                "transition with the denoiser recursion."
            ),
            supervision_target_vocab=(
                "Archetype-id classification (5 archetypes) + role-set "
                "multi-label over 4 roles on a toy fixture corpus "
                "(families=8, train=51, val=13)."
            ),
            readout_head_sharing=(
                "Separate heads with their own parameters, not the shared "
                "lm_head/token vocab — the exact configuration candidate "
                "differentiator 2 rules out for the new probe."
            ),
            downstream_consumption=(
                "Predicted factors are serialized as SemanticPlanV1 and (per "
                "SLM-146) compiled into a seed / soft action features by a "
                "downstream compiler — an external artifact, not an in-"
                "recursion read."
            ),
            structure_handling=(
                "Role-set predicted via bipartite matching (set-valued); no "
                "topology/partial-order prediction (that scope is SLM-145, "
                "blocked)."
            ),
            inference_inputs_leakage=(
                "Prompt-only component-family counts at inference; oracle arms "
                "(gold_archetype/gold_role_set/gold_both) intentionally leak "
                "gold labels for ceiling measurement only."
            ),
            compiler_verifier_authority=(
                "None directly — the plan stays soft per "
                "semantic-planning-valid-state.md, but the whole predictor "
                "pipeline lives outside the denoiser."
            ),
            causal_intervention_evidence=(
                "Oracle arms are gold-substitution ceilings on the predictor's "
                "own accuracy metric, not on downstream free-running "
                "generation quality."
            ),
            param_flop_byte_impact="Not measured beyond the toy fixture; no production integration.",
            evidence_verdict="status: plan_only/fixture (iter-slm144 docs); claim_class wiring.",
        ),
        MechanismComparisonRow(
            mechanism_id="slm145_plan_predictor_topology_cardinality_pointer",
            state_location_shape=(
                "External and never implemented — topology_head, "
                "cardinality_head, live_symbol_pointer_head are named in the "
                "plan but a repo-wide grep for these class names returns zero "
                "hits in src/."
            ),
            weight_sharing_recurrence="N/A — blocked before implementation.",
            supervision_target_vocab=(
                "Would have been topology/cardinality/binding factors; "
                "RoleSlot.min/max_cardinality is not even populated by the "
                "extractor, so no oracle arm was possible."
            ),
            readout_head_sharing="N/A.",
            downstream_consumption="N/A — closeout decision: blocked_pending_spv0_02_ceiling_evidence.",
            structure_handling=(
                "Partial-order/topology and pointer/binding representation was "
                "the explicit target of this blocked work; never resolved."
            ),
            inference_inputs_leakage="N/A.",
            compiler_verifier_authority="N/A.",
            causal_intervention_evidence=(
                "None — SLM-145's own closeout states SPV0-02/SLM-142 never ran "
                "the factor-wise gold-substitution ceiling experiments needed "
                "to justify these heads; justified_factors=[] for all 5 "
                "factors (archetype, role_set, topology, cardinality, "
                "bindings_pointers)."
            ),
            param_flop_byte_impact="N/A — nothing implemented.",
            evidence_verdict="closeout decision: blocked_pending_spv0_02_ceiling_evidence; claim_class wiring.",
        ),
        MechanismComparisonRow(
            mechanism_id="slm146_semantic_plan_compiler",
            state_location_shape=(
                "External — SemanticPlanCompiler/OpenUISemanticPlanCompiler "
                "(src/slm_training/data/semantic_plan/compiler.py) consumes a "
                "serialized SemanticPlanV1 object; PlanSeedBuilder (seed.py) "
                "builds a decoder seed. No in-model latent state."
            ),
            weight_sharing_recurrence="N/A — deterministic compiler code, not a trained shared transition.",
            supervision_target_vocab=(
                "Seed tokens use the same OpenUI action/token vocabulary as "
                "generation (mean seed-to-gold token ratio 0.4106 in the "
                "gold-seed arm B), but this is symbolic seed construction, not "
                "a supervised internal slot."
            ),
            readout_head_sharing=(
                "PlanActionFeatures are soft action features attached post-hoc "
                "to the legal action set; EvidenceKind gates whether "
                "restrictions may go hard (COMPILER_AUTHORED_CERTIFIED only) — "
                "a different sharing mechanism from an in-recursion readout "
                "head entirely."
            ),
            downstream_consumption=(
                "Seed initializes the decode canvas; soft features bias scoring "
                "without changing legal membership (arm C, total_soft_features="
                "32, total_hard_removals=0); certified restrictions may hard-"
                "prune (arm E, 13 hard removals, 0 false_hard_prunes measured)."
            ),
            structure_handling=(
                "Explicit accepted-set representation via role/topology "
                "coverage metrics; UNKNOWN handled via honesty_mode + fail-"
                "closed default (allow_unsafe_predicted_hard_control=False)."
            ),
            inference_inputs_leakage=(
                "Gold-seed arms (B/C) leak gold structure for ceiling "
                "measurement only; production honesty_mode strips gold/oracle "
                "fields via to_production_dict()."
            ),
            compiler_verifier_authority=(
                "This IS the compiler/verifier boundary — "
                "EvidenceKind.COMPILER_AUTHORED_CERTIFIED is the only class "
                "allowed to hard-restrict; arm F (unsafe predicted-hard) is "
                "explicitly non-promotable."
            ),
            causal_intervention_evidence=(
                "Arms B/C/E provide gold-substitution and certified-"
                "restriction ceiling evidence for the EXTERNAL seed/feature "
                "pipeline (see the oracle-ceiling section below) — this does "
                "not test an internal shared-output latent consumer."
            ),
            param_flop_byte_impact="Fixture-only synthetic n=13 records per arm; no production decoder integration.",
            evidence_verdict=(
                "status: fixture; claim_class wiring; SLM-160 disposition: "
                "retain_diagnostic, default_state off."
            ),
        ),
        MechanismComparisonRow(
            mechanism_id="slm160_spv_disposition",
            state_location_shape="N/A — disposition audit only; aggregates SLM-138/144/145/146 and others.",
            weight_sharing_recurrence="N/A.",
            supervision_target_vocab="N/A.",
            readout_head_sharing="N/A.",
            downstream_consumption="N/A.",
            structure_handling="N/A.",
            inference_inputs_leakage="N/A.",
            compiler_verifier_authority=(
                "Canonical recommendation: the existing honest-slot-contract "
                "TwoTower decoder remains canonical; semantic_plan_v1_ir, "
                "gold_oracle_factor_heads, and "
                "plan_seed_builder_soft_restrictions are ALL retained as "
                "default-off diagnostics — none promoted to adopt_primary or "
                "adopt_optional."
            ),
            causal_intervention_evidence="None newly produced; audit-only.",
            param_flop_byte_impact="N/A.",
            evidence_verdict=(
                "No mechanism satisfies adopt_primary/adopt_optional. Per the "
                "SLM-229 issue text, none of SLM-138/144/145/146/160 may be "
                "relabeled 'latent reasoning' by this memo."
            ),
        ),
        MechanismComparisonRow(
            mechanism_id="proposed_minimal_compiler_latent_probe",
            state_location_shape=(
                "Would reuse SharedRecursiveDenoiserTower's existing internal "
                "z-stream (or a small K-slot extension of it) — internal, not "
                "serialized; K in {1, 2} slots at d_model, scoped to "
                "root_contract + component_inventory only."
            ),
            weight_sharing_recurrence=(
                "Would reuse the existing shared F_theta/G_theta transition "
                "pool (recursive_transition_layers) — no new per-depth "
                "transformer parameters beyond slot read/write projections."
            ),
            supervision_target_vocab=(
                "root_contract (archetype id — deterministically extractable "
                "today per semantic-planning-valid-state.md, no learned "
                "predictor needed) + component_inventory (component-family "
                "counts, the same deterministic vector SLM-144's fixture "
                "already used). Explicitly NOT topology/cardinality/binding/"
                "pointer (that scope stays blocked per SLM-145)."
            ),
            readout_head_sharing=(
                "Contractually required to decode slot targets through the "
                "same lm_head/project() vocab used by legal decisions — NOT a "
                "separate ArchetypeClassifierHead/SerializedInventoryHead as "
                "SLM-144/145 used."
            ),
            downstream_consumption=(
                "Contractually required: y reads z every recursion step "
                "through the existing G_theta cross-attention/gating path "
                "(already present in SLM-138's architecture) — not merely "
                "logged beside generation."
            ),
            structure_handling=(
                "Accepted-set/multi-label root_contract + component_inventory "
                "targets with an explicit UNKNOWN/ambiguous representation "
                "(contract field); no partial-order/topology claim."
            ),
            inference_inputs_leakage=(
                "Same prompt/context inputs as generation; built-in gold/zero/"
                "swap/wrong-target interventions are ablations on a frozen or "
                "in-training checkpoint, never a silent gold channel (mirrors "
                "the honest_slot_contract=True policy)."
            ),
            compiler_verifier_authority=(
                "Explicitly forbidden from altering compiler legal membership, "
                "verifier truth, certified restrictions, or UNKNOWN handling "
                "(differentiator 6) — the same authority boundary SLM-146's "
                "EvidenceKind.COMPILER_AUTHORED_CERTIFIED enforces for the "
                "external pipeline, reused conceptually for the internal path."
            ),
            causal_intervention_evidence=(
                "None yet — required as a day-one fixture per differentiator "
                "5, but not run; SLM-146's oracle arms test the external-seed "
                "consumer, not this internal-slot consumer (see the oracle "
                "ceiling gap below)."
            ),
            param_flop_byte_impact=(
                "K in {1,2} slot read/write projections on top of SLM-138's "
                "existing 74242-param toy fixture — expected negligible "
                "incremental parameter count at this scale, but no "
                "measurement exists because differentiator 7's floor/"
                "recurrence prerequisites are unmet, so no run has produced a "
                "number."
            ),
            evidence_verdict=(
                "NOT AUTHORIZED this round — blocked_by_recurrence "
                "(differentiator 7 unmet): SLM-139 explicitly closed the "
                "prerequisite non-vacuous recursive regime as failed, and no "
                "'semantic floor' gate is defined anywhere in docs/design "
                "(zero grep hits) for the required floor-escape condition "
                "either."
            ),
        ),
    ]


def build_target_support_audit() -> list[TargetSupportRow]:
    """Return the data/target availability audit for root_contract and inventory."""
    return [
        TargetSupportRow(
            target_id="root_contract_archetype",
            deterministic_extraction=(
                "Yes — semantic-planning-valid-state.md states archetype "
                "inference is deterministic and pack-derived today (no "
                "learned predictor exists); the existing extractor is "
                "reusable and must not be reimplemented."
            ),
            accepted_set_multi_label=(
                "PlanArchetype carries a soft distribution plus a discrete id "
                "with confidence — single-label id with confidence, not "
                "inherently multi-label."
            ),
            ambiguous_unknown_handling=(
                "SemanticPlanV1.PlanConfidenceCalibration.abstention_reason "
                "already exists in the schema for low-confidence/ambiguous "
                "cases; a probe supervision target must reuse this "
                "representation rather than silently coercing ambiguous "
                "archetypes to a negative class."
            ),
            split_leakage_fingerprint=(
                "Not established for a real (non-fixture) corpus in the "
                "reviewed docs — SLM-144's fixture corpus (train=51, val=13, "
                "archetypes=5, families=8) is toy-scale only; no production "
                "train/held-out group split or leakage fingerprint audit for "
                "archetype targets was found in the reviewed evidence."
            ),
            id_mapping=(
                "Must reuse the SAME action/token/component ID mapping as "
                "lm_head/project() (shared output geometry, differentiator 2) "
                "— not the separate ArchetypeClassifierHead's own class-id "
                "space used by SLM-144."
            ),
            counts_by_family=(
                "SLM-144 fixture only: archetypes=5, families=8, roles=4, "
                "train=51, val=13. No real-corpus counts by root family / "
                "source / suite / rare tail were found in the reviewed docs."
            ),
        ),
        TargetSupportRow(
            target_id="component_inventory",
            deterministic_extraction=(
                "Yes in the fixture — SLM-144's frequency baseline already "
                "used a deterministic component-family count vector as its "
                "predictor input; the same deterministic extraction is "
                "reusable as a supervision target."
            ),
            accepted_set_multi_label=(
                "Multi-label / set-valued by nature (an inventory is a set of "
                "present component families). RoleSetPredictorHead already "
                "models a related set as bipartite matching in the existing "
                "EXTERNAL pipeline; an internal probe needs its own "
                "accepted-set representation and must not reuse that "
                "external head (differentiator 2)."
            ),
            ambiguous_unknown_handling=(
                "GAP: not explicitly audited in the reviewed docs for "
                "inventory targets specifically. SPV0-02/SLM-142 extraction "
                "work covers archetype/role/topology broadly, but the "
                "reviewed evidence does not show a dedicated inventory-target "
                "ambiguity table. This is an open gap the probe's first "
                "fixture must close before target support can be called "
                "adequate; it must not be silently coerced to a negative "
                "label."
            ),
            split_leakage_fingerprint=(
                "Same gap as root_contract_archetype — no production-scale "
                "held-out split / leakage fingerprint was found in the "
                "reviewed evidence for inventory targets."
            ),
            id_mapping=(
                "Must reuse the shared component/token vocabulary already "
                "used by legal decisions (differentiator 2), not "
                "SerializedInventoryHead's own GRU-decoder vocabulary."
            ),
            counts_by_family=(
                "SLM-144 fixture: families=8 (component-family count vector "
                "dimensionality). No rare-tail or per-suite counts were found "
                "in the reviewed docs."
            ),
        ),
    ]


def build_oracle_intervention_ceiling() -> OracleInterventionCeiling:
    """Reuse SLM-146's oracle evidence and identify the remaining gap."""
    return OracleInterventionCeiling(
        reused_evidence_paths=(
            "docs/design/iter-slm146-semantic-plan-compiler-20260720.json",
            "docs/design/iter-slm146-semantic-plan-compiler-20260720.md",
        ),
        reused_evidence_applicable=True,
        gap_description=(
            "SLM-146 arms B (gold_seed) / C (gold_seed_soft) / E "
            "(certified_restrictions) show that gold-substituted plan "
            "structure materially changes the EXTERNAL seed/soft-feature "
            "consumer (seed-to-gold token ratio 0.4106; 32 soft features "
            "attached in arm C; 0 false_hard_prunes across the certified-"
            "only arm E). That evidence answers 'does gold plan info change "
            "the seed-builder/compiler consumer' but does NOT test whether "
            "gold-substituting a shared-output-geometry INTERNAL latent slot "
            "(read every recursion step by y via G_theta) changes "
            "free-running token choices, because no such internal supervised "
            "channel exists in any implemented mechanism today — SLM-138's z "
            "stream carries no defined target vocabulary and was never "
            "gold/zero/swap ablated."
        ),
        smallest_fixture_spec=(
            "NOT RUN (spec only, for a future RSC3 issue, conditional on "
            "differentiator 7's gates clearing first): on a frozen or "
            "lightly-trained SharedRecursiveDenoiserTower checkpoint, force "
            "z_1 (or a designated K-slot subset of z) to a fixed embedding "
            "derived from the gold root_contract archetype id / component-"
            "inventory set (reusing the shared lm_head/project() embedding "
            "table per differentiator 2), leaving all other inputs and "
            "weights unchanged; measure the free-running argmax token "
            "flip-rate and downstream parse/meaningful-program rate versus "
            "the same checkpoint with z_1 zeroed and with z_1 swapped to a "
            "mismatched record's gold value. No training step is required — "
            "this is a forward-pass-only oracle-injection ablation."
        ),
    )


def build_scale_regime_audit(
    semantic_floor_status: str = "SemanticFloorGateV1 status not supplied to this builder."
) -> ScaleRegimeAudit:
    """Return the scale/regime audit; floor and recurrence status are mandatory."""
    return ScaleRegimeAudit(
        lotus_scale_notes=(
            "Not independently verified in this environment (no live web "
            "access). LOTUS ('looped transformer reasoning', "
            "arXiv:2606.31779) is presumed, from title framing only, to "
            "operate at a materially larger parameter/data/task scale than "
            "this repo's scratch TwoTower models; no specific number is "
            "asserted."
        ),
        current_model_params_notes=(
            "SLM-138 toy fixture: stacked_params=64994 vs "
            "recursive_params=74242 at d_model=32, [batch=2, seq=6]. No "
            "production-scale (non-fixture) recursive-denoiser parameter "
            "count exists in the reviewed evidence."
        ),
        target_tokens_decisions_notes=(
            "No real-corpus target-token or decision counts exist for "
            "root_contract/component_inventory supervision; SLM-144's fixture "
            "corpus is the only reviewed sizing evidence."
        ),
        unique_records_steps_notes=(
            "SLM-144 fixture: train=51, val=13 unique records (archetypes=5, "
            "families=8, roles=4); SLM-146 fixture: n=13 synthetic records per "
            "arm. Both are toy-scale; no non-fixture step count exists."
        ),
        semantic_floor_status=semantic_floor_status,
        recursive_regime_status=(
            "FAILED (documented) — SLM-139's closeout explicitly states "
            "gate_1_recursive_base (issue SLM-138) failed: 'wiring_only "
            "fixture; no GPU matched-block evaluation or "
            "recursive_core_positive verdict.' No non-vacuous recursive "
            "regime has been established for SharedRecursiveDenoiserTower."
        ),
        identifiability_verdict=(
            "NOT EVALUABLE from current evidence: no non-toy recursive "
            "training run exists to assess whether 1-2 slots plus the "
            "proposed root_contract/component_inventory supervision would be "
            "statistically identifiable. This question is moot until the "
            "recurrence and floor prerequisites in differentiator 7 clear; "
            "it is not being asserted as scale_not_identifiable in its own "
            "right, only as unresolved."
        ),
        expected_extra_cost_notes=(
            "If built on top of the existing SharedRecursiveDenoiserTower, "
            "the incremental cost is expected to be small (a few slot "
            "read/write linear projections at d_model, no new recursion "
            "depth, no new transformer blocks) — but this is an expectation, "
            "not a measurement: no implementation exists and none should be "
            "built before differentiator 7 clears."
        ),
    )


def build_prior_art_audit() -> list[PriorArtRow]:
    """Return the prior-art / novelty audit rows."""
    return [
        PriorArtRow(
            topic="looped/recurrent transformers (LOTUS, Universal Transformer)",
            internal_hits=(
                "Zero hits repo-wide for '2606.31779', 'LOTUS', '1807.03819', "
                "or 'Universal Transformer' in docs/ or src/ (grep at audit "
                "time) outside this memo. SLM-138's research-lineage.md row "
                "(lines 155-165) cites 'Deep equilibrium / weight-tied "
                "recurrent transitions' and explicitly disclaims reproducing "
                "any cited DEQ training recipe, but does not name LOTUS or "
                "Universal Transformer directly."
            ),
            external_verification_note=(
                "Not independently verified in this environment (no live web "
                "access); reasoning limited to the paper titles/arXiv ids "
                "given in the issue text."
            ),
            novelty_scope="repository",
            novelty_notes=(
                "SLM-138's shared y/z recursion is the closest internal "
                "analogue and already covers the weight-tied-recurrence "
                "lineage under a DEQ framing; explicitly naming LOTUS/"
                "Universal Transformer in research-lineage.md is a "
                "documentation-only follow-up, not performed by this memo to "
                "avoid an unrelated drive-by edit."
            ),
        ),
        PriorArtRow(
            topic="latent/continuous reasoning with direct output-head supervision",
            internal_hits=(
                "src/slm_training/models/continuous_latent.py "
                "(ContinuousLatentCodec) is an existing, DIFFERENT 'latent' "
                "mechanism: a CAP2 rate-controlled communication bottleneck "
                "with an explicit noise/rate penalty, not an internal "
                "workspace slot read by cross-attention/gating and not "
                "supervised against root_contract/component_inventory "
                "targets through the shared output head. Not a naming "
                "collision with the proposed probe, but flagged to "
                "disambiguate."
            ),
            external_verification_note=(
                "research-lineage.md's 'Latent falsification MoE' entry cites "
                "Coconut (arXiv:2412.06769) and related continuous-latent-"
                "reasoning papers as Adjacent, long-horizon design only (E34), "
                "deferred until residual failures after E33+E35 — the closest "
                "existing repo acknowledgment of this literature cluster."
            ),
            novelty_scope="repository",
            novelty_notes="No internal implementation of direct-output-head-supervised latent slots exists.",
        ),
        PriorArtRow(
            topic="parallel latent tokens/slots",
            internal_hits=(
                "No 'MinimalCompilerLatentContract' or equivalent slot-schema "
                "name exists anywhere in the repo prior to this memo (grep for "
                "'latent slot' / 'compiler latent' returns no matching prior "
                "art)."
            ),
            external_verification_note=(
                "PLR parallel latent streams (arXiv:2601.03153) and MIRAGE "
                "continuous agent latent CoT (arXiv:2606.04627) are already "
                "catalogued as Adjacent in the 'Latent falsification MoE' "
                "research-lineage.md row; not independently re-verified here."
            ),
            novelty_scope="repository",
            novelty_notes="The proposed schema name and internal-slot design are new to this repo.",
        ),
        PriorArtRow(
            topic="compiler/AST semantic planning",
            internal_hits=(
                "Extensive and directly on point: SemanticPlanV1 "
                "(src/slm_training/data/progspec/semantic_plan.py), predictor "
                "heads (src/slm_training/models/semantic_plan_predictor.py), "
                "and the plan compiler / seed builder "
                "(src/slm_training/data/semantic_plan/compiler.py, seed.py) — "
                "this is the SPV lineage the issue explicitly forbids "
                "duplicating."
            ),
            external_verification_note="N/A — internal audit only.",
            novelty_scope="none_claimed",
            novelty_notes=(
                "Any proposal that exports a plan, uses separate heads with "
                "soft features, constructs a plan seed, or predicts topology/"
                "bindings externally belongs to this completed lineage and is "
                "not authorized as new."
            ),
        ),
        PriorArtRow(
            topic="causal representation interventions",
            internal_hits=(
                "SLM-146 arms B/C/E (gold-substitution, soft-feature, "
                "certified-restriction) are the only implemented causal-"
                "intervention evidence adjacent to this proposal, and they "
                "test the EXTERNAL seed/compiler consumer, not an internal "
                "shared-output slot (see the oracle-ceiling gap above)."
            ),
            external_verification_note="N/A — internal audit only.",
            novelty_scope="repository",
            novelty_notes="No internal gold/zero/swap/wrong intervention exists for any in-recursion latent state.",
        ),
    ]


def build_differentiators() -> list[DifferentiatorVerdict]:
    """Return the 7 candidate differentiators with explicit true/false + evidence."""
    field_map = differentiator_contract_field_map()
    return [
        DifferentiatorVerdict(
            differentiator_id=1,
            name="internal_workspace",
            satisfied=True,
            evidence=(
                "Specifiable and consistent with existing architecture: "
                "SharedRecursiveDenoiserTower's z-stream is already an "
                "internal, non-serialized channel "
                "(recursive_outputs() does not export it as an artifact); "
                "the contract must require slot state to live there, never "
                "in a serialized SemanticPlanV1 object."
            ),
            contract_field=field_map[1],
            test_ref="test_differentiator_1_maps_to_slot_kinds_and_rejects_spv_export",
        ),
        DifferentiatorVerdict(
            differentiator_id=2,
            name="shared_decision_output_geometry",
            satisfied=True,
            evidence=(
                "Specifiable, not automatic: SLM-138 already decodes every "
                "recursion depth through the single shared lm_head/project() "
                "(depth_logits.append(self.project(h))); the contract must "
                "require slot targets to use that SAME projection, "
                "explicitly forbidding SLM-144/145's separate "
                "ArchetypeClassifierHead / SerializedInventoryHead pattern."
            ),
            contract_field=field_map[2],
            test_ref="test_differentiator_2_rejects_separate_auxiliary_head_proposal",
        ),
        DifferentiatorVerdict(
            differentiator_id=3,
            name="direct_consumer",
            satisfied=True,
            evidence=(
                "Specifiable and architecturally natural: z_r already feeds "
                "y_r every recursion step via G_theta's cross-attention/"
                "gating path by construction, so a slot-carrying z is a "
                "direct consumer, not a side channel logged beside "
                "generation."
            ),
            contract_field=field_map[3],
            test_ref="test_differentiator_3_requires_surface_consumer_mechanism_field",
        ),
        DifferentiatorVerdict(
            differentiator_id=4,
            name="minimal_scope",
            satisfied=True,
            evidence=(
                "Specifiable: restricting slot_kinds to {root_contract, "
                "component_inventory} avoids SLM-145's blocked topology/"
                "cardinality/pointer scope (topology_head, cardinality_head, "
                "live_symbol_pointer_head were never implemented and remain "
                "closed pending SPV0-02 ceiling evidence) and avoids "
                "SLM-146's plan-compiler/seed machinery entirely."
            ),
            contract_field=field_map[4],
            test_ref="test_differentiator_4_rejects_topology_binding_pointer_scope_creep",
        ),
        DifferentiatorVerdict(
            differentiator_id=5,
            name="builtin_interventions",
            satisfied=True,
            evidence=(
                "Specifiable: the contract's interventions field is required "
                "to enumerate gold/zero/swap/wrong/detached from the first "
                "fixture, unlike SLM-138 (no ablation ever run on z) and "
                "unlike SLM-146 (whose oracle arms test the external "
                "consumer only)."
            ),
            contract_field=field_map[5],
            test_ref="test_differentiator_5_rejects_proposal_missing_interventions",
        ),
        DifferentiatorVerdict(
            differentiator_id=6,
            name="no_hard_authority",
            satisfied=True,
            evidence=(
                "Specifiable: the contract's authority-boundary field must "
                "forbid altering compiler legal membership, verifier truth, "
                "certified restrictions, or UNKNOWN handling — mirroring "
                "SLM-146's own EvidenceKind.COMPILER_AUTHORED_CERTIFIED "
                "fail-closed pattern (allow_unsafe_predicted_hard_control="
                "False by default), reused conceptually for the internal "
                "path."
            ),
            contract_field=field_map[6],
            test_ref="test_differentiator_6_rejects_proposal_that_can_prune_legal_actions",
        ),
        DifferentiatorVerdict(
            differentiator_id=7,
            name="conditional_execution",
            satisfied=False,
            evidence=(
                "UNMET today on both prerequisites: (a) recursive regime — "
                "SLM-139 explicitly closed gate_1_recursive_base as failed "
                "('no_supported_probabilistic_regime'; no "
                "recursive_core_positive verdict); (b) semantic floor — no "
                "'semantic floor' gate is defined anywhere in docs/design "
                "(zero grep hits), so there is no threshold to escape yet. "
                "This is the differentiator that blocks the overall "
                "verdict."
            ),
            contract_field=field_map[7],
            test_ref="test_differentiator_7_gates_are_mandatory_and_currently_unmet",
        ),
    ]


def evaluate_verdict(
    *,
    differentiators: Mapping[int, bool],
    exports_external_plan: bool,
    uses_separate_aux_head: bool,
    constructs_plan_seed: bool,
    predicts_topology_bindings_externally: bool,
    target_support_adequate: bool,
    scale_identifiable: bool | None,
    floor_ready: bool,
    recurrence_ready: bool,
) -> LoopedLatentVerdict:
    """Deterministic decision-rule evaluator for a candidate probe proposal.

    Mirrors the issue's decision rule for ``authorize_minimal_probe`` (ALL
    seven differentiators true, no duplicate SPV path, adequate target
    support, specified shared-output/direct-consumer geometry, preregistered
    intervention/floor/recurrence gates) and the negative-verdict priority
    order used for rejection fixtures.

    When both the floor and recurrence prerequisites are unmet, the
    recurrence gate is reported first: it has directly measured, documented
    failing evidence (SLM-139's closeout), whereas the floor gate may simply
    be undefined in the repo rather than measured-and-failing. Prefer the
    more concretely evidenced blocker.
    """
    if exports_external_plan or uses_separate_aux_head or constructs_plan_seed or predicts_topology_bindings_externally:
        return LoopedLatentVerdict.DUPLICATE_SPV
    if not target_support_adequate:
        return LoopedLatentVerdict.UNSUPPORTED_TARGETS
    if scale_identifiable is False:
        return LoopedLatentVerdict.SCALE_NOT_IDENTIFIABLE
    if not recurrence_ready:
        return LoopedLatentVerdict.BLOCKED_BY_RECURRENCE
    if not floor_ready:
        return LoopedLatentVerdict.BLOCKED_BY_FLOOR
    if all(differentiators.get(i, False) for i in range(1, 8)):
        return LoopedLatentVerdict.AUTHORIZE_MINIMAL_PROBE
    return LoopedLatentVerdict.INCONCLUSIVE


def validate_doc_refs(refs: Iterable[str] | tuple[str, ...], repo_root: Path | None = None) -> list[str]:
    """Return the subset of ``refs`` that do not resolve to a real path or a bare issue id."""
    if repo_root is None:
        repo_root = _repo_root()
    missing: list[str] = []
    for ref in refs:
        if ref.startswith(("SLM-", "SPV", "RSC")) and "/" not in ref:
            # Bare issue id reference (Linear), not a repo-relative path.
            continue
        if not (repo_root / ref).exists():
            missing.append(ref)
    return missing


def _build_version_stamp() -> dict[str, Any]:
    """Build a version stamp, degrading if the slm229 component is not yet registered."""
    try:
        return build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm229_looped_latent_differentiation",
            "harness.experiments.semantic_floor_gate",
        )
    except KeyError:
        base = build_version_stamp("harness.experiments")
        base["components"]["harness.experiments.slm229_looped_latent_differentiation"] = UNKNOWN
        return base


def run_differentiation_audit(
    *,
    run_id: str = "slm229_differentiation",
    status: str = "fixture",
    repo_root: Path | None = None,
) -> LoopedLatentDifferentiationReport:
    """Build the SLM-229 (RSC0-01) looped-latent differentiation report."""
    if repo_root is None:
        repo_root = _repo_root()

    floor_gate = load_semantic_floor_gate(repo_root / DEFAULT_GATE_PATH)
    try:
        require_floor_gate(floor_gate, "learned_latent")
        floor_ready = True
    except PermissionError:
        floor_ready = False
    differentiators = build_differentiators()
    differentiators[-1] = replace(
        differentiators[-1],
        satisfied=floor_ready and differentiators[-1].satisfied,
        evidence=(
            f"SemanticFloorGateV1 `{floor_gate.gate_hash}` resolves the floor half "
            f"as `{floor_gate.verdict}`; learned-latent claims remain blocked. "
            "The recurrence half also remains failed by SLM-139."
        ),
    )
    differentiator_map = {d.differentiator_id: d.satisfied for d in differentiators}

    verdict = evaluate_verdict(
        differentiators=differentiator_map,
        exports_external_plan=False,
        uses_separate_aux_head=False,
        constructs_plan_seed=False,
        predicts_topology_bindings_externally=False,
        target_support_adequate=True,
        scale_identifiable=None,
        floor_ready=floor_ready,
        recurrence_ready=False,
    )

    missing_refs = validate_doc_refs(REVIEWED_REFS, repo_root=repo_root)

    resolving_evidence = (
        "docs/design/iter-slm139-stochastic-recursive-width-20260720.json "
        "(gate_1_recursive_base = failed, decision "
        "no_supported_probabilistic_regime) is the resolving evidence for "
        "differentiator 7's recurrence half. "
        f"{DEFAULT_GATE_PATH} (hash {floor_gate.gate_hash}, verdict "
        f"{floor_gate.verdict}) resolves the floor half and does not authorize "
        "learned-latent claims."
    )
    if missing_refs:
        resolving_evidence += (
            f" NOTE: {len(missing_refs)} reviewed reference(s) did not "
            f"resolve at audit time: {', '.join(missing_refs)}."
        )

    return LoopedLatentDifferentiationReport(
        schema=REPORT_SCHEMA,
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status=status,
        claim_class="wiring",
        source_commit=git_commit(),
        evidence_cutoff=git_commit(),
        reviewed_refs=REVIEWED_REFS,
        generated_at=_now(),
        floor_gate_ref=DEFAULT_GATE_PATH,
        floor_gate_hash=floor_gate.gate_hash,
        floor_gate_verdict=floor_gate.verdict,
        mechanism_comparison=tuple(build_mechanism_comparison()),
        target_support_audit=tuple(build_target_support_audit()),
        oracle_intervention_ceiling=build_oracle_intervention_ceiling(),
        scale_regime_audit=build_scale_regime_audit(
            f"SemanticFloorGateV1 `{floor_gate.gate_hash}` verdict is "
            f"`{floor_gate.verdict}`; learned-latent claims are blocked."
        ),
        prior_art_audit=tuple(build_prior_art_audit()),
        differentiators=tuple(differentiators),
        verdict=verdict,
        allowed_implementation_scope=(
            "None yet. If differentiator 7's floor and recurrence gates "
            "later clear, the allowed scope is narrowly: an internal, "
            "default-off K in {1,2} slot extension of "
            "SharedRecursiveDenoiserTower's existing z-stream, supervised "
            "only on root_contract archetype and component_inventory "
            "targets, decoded through the existing shared lm_head/project() "
            "output geometry, read every recursion step by y through the "
            "existing G_theta cross-attention/gating path, with gold/zero/"
            "swap/wrong/detached interventions in the first fixture, and no "
            "authority over compiler legality, verifier truth, certified "
            "restrictions, or UNKNOWN handling."
        ),
        forbidden_duplicate_scope=(
            "Do not duplicate: SemanticPlanV1 export/serialization "
            "(src/slm_training/data/progspec/semantic_plan.py), the "
            "archetype/role-set/topology/cardinality/pointer predictor heads "
            "(src/slm_training/models/semantic_plan_predictor.py), the plan "
            "compiler / seed builder / soft-action-feature / certified-"
            "restriction machinery "
            "(src/slm_training/data/semantic_plan/compiler.py, seed.py), or "
            "any topology/binding/pointer/cardinality target (SLM-145's "
            "blocked scope)."
        ),
        resolving_evidence=resolving_evidence,
        contract_hash=None,
        minimal_contract=None,
        version_stamp=_build_version_stamp(),
    )


def render_markdown(report: LoopedLatentDifferentiationReport) -> str:
    """Render the sectioned Markdown differentiation memo."""
    lines = [
        f"# SLM-229 (RSC0-01): Looped-latent differentiation memo ({report.run_id})",
        "",
        f"**Schema:** `{report.schema}`",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        "",
        f"**Version:** `{report.matrix_version}`",
        "",
        f"**Status:** {report.status}",
        "",
        f"**Claim class:** {report.claim_class} / zero-compute differentiation "
        "and authorization audit only. No model or data implementation, no "
        "training run, no checkpoint.",
        "",
        f"**Source commit / evidence cutoff:** `{report.source_commit}`",
        "",
        f"**Generated at:** {report.generated_at}",
        "",
        f"**Verdict:** `{report.verdict.value}`",
        "",
        f"**Semantic floor gate:** `{report.floor_gate_hash}` "
        f"(`{report.floor_gate_verdict}`; `{report.floor_gate_ref}`)",
        "",
        "## Reviewed references",
        "",
    ]
    for ref in report.reviewed_refs:
        lines.append(f"- `{ref}`")

    lines.extend(
        [
            "",
            "## 1. Mechanism comparison table",
            "",
            "| Mechanism | State location/shape | Weight sharing/recurrence | "
            "Supervision target/vocab | Readout head sharing | Downstream "
            "consumption | Structure handling | Inference inputs/leakage | "
            "Compiler/verifier authority | Causal intervention evidence | "
            "Param/FLOP/byte impact | Evidence/verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.mechanism_comparison:
        cells = [
            row.mechanism_id,
            row.state_location_shape,
            row.weight_sharing_recurrence,
            row.supervision_target_vocab,
            row.readout_head_sharing,
            row.downstream_consumption,
            row.structure_handling,
            row.inference_inputs_leakage,
            row.compiler_verifier_authority,
            row.causal_intervention_evidence,
            row.param_flop_byte_impact,
            row.evidence_verdict,
        ]
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 2. Data/target availability audit",
            "",
            "| Target | Deterministic extraction | Accepted-set/multi-label | "
            "Ambiguous/UNKNOWN handling | Split/leakage fingerprint | ID "
            "mapping | Counts by family |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.target_support_audit:
        cells = [
            row.target_id,
            row.deterministic_extraction,
            row.accepted_set_multi_label,
            row.ambiguous_unknown_handling,
            row.split_leakage_fingerprint,
            row.id_mapping,
            row.counts_by_family,
        ]
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")

    ceiling = report.oracle_intervention_ceiling
    lines.extend(
        [
            "",
            "## 3. Oracle intervention ceiling",
            "",
            f"**Reused evidence:** {', '.join(f'`{p}`' for p in ceiling.reused_evidence_paths)}",
            "",
            f"**Applicable:** {ceiling.reused_evidence_applicable}",
            "",
            "**Gap:**",
            "",
            ceiling.gap_description,
            "",
            "**Smallest no-training oracle fixture needed (future RSC3 issue, spec only, NOT run):**",
            "",
            ceiling.smallest_fixture_spec,
        ]
    )

    scale = report.scale_regime_audit
    lines.extend(
        [
            "",
            "## 4. Scale/regime audit",
            "",
            f"- **LOTUS scale:** {scale.lotus_scale_notes}",
            f"- **Current model params:** {scale.current_model_params_notes}",
            f"- **Target tokens/decisions:** {scale.target_tokens_decisions_notes}",
            f"- **Unique records/steps:** {scale.unique_records_steps_notes}",
            f"- **Semantic floor status:** {scale.semantic_floor_status}",
            f"- **Recursive regime status:** {scale.recursive_regime_status}",
            f"- **Identifiability verdict:** {scale.identifiability_verdict}",
            f"- **Expected extra cost:** {scale.expected_extra_cost_notes}",
        ]
    )

    lines.extend(
        [
            "",
            "## 5. Prior-art and novelty audit",
            "",
            "| Topic | Internal hits | External verification | Novelty scope | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.prior_art_audit:
        cells = [
            row.topic,
            row.internal_hits,
            row.external_verification_note,
            row.novelty_scope,
            row.novelty_notes,
        ]
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Candidate differentiators (all 7)",
            "",
            "| # | Name | Satisfied | Contract field | Test ref | Evidence |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for d in report.differentiators:
        evidence = d.evidence.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {d.differentiator_id} | {d.name} | {d.satisfied} | "
            f"`{d.contract_field}` | `{d.test_ref}` | {evidence} |"
        )

    lines.extend(
        [
            "",
            "## Allowed implementation scope",
            "",
            report.allowed_implementation_scope,
            "",
            "## Forbidden/duplicate scope",
            "",
            report.forbidden_duplicate_scope,
            "",
            "## Resolving evidence",
            "",
            report.resolving_evidence,
            "",
            "## MinimalCompilerLatentContractV1",
            "",
        ]
    )
    if report.minimal_contract is not None:
        lines.append("```json")
        lines.append(json.dumps(report.minimal_contract.to_dict(), indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
        lines.append(f"**Contract hash:** `{report.contract_hash}`")
    else:
        lines.append(
            "Not populated. The verdict is not `authorize_minimal_probe`, so "
            "per the issue's instruction the contract schema exists in code "
            "(`slm_training.harnesses.experiments."
            "slm229_looped_latent_differentiation.MinimalCompilerLatentContractV1`) "
            "but is deliberately left unpopulated in this report — no dormant "
            "latent production code is created just to produce an "
            "`authorize` verdict."
        )

    lines.extend(
        [
            "",
            "## Reproducibility commands",
            "",
            "```bash",
            "# Plan-only manifest (no evidence reads)",
            "python -m scripts.run_slm229_looped_latent_differentiation --mode plan-only",
            "",
            "# Fixture audit that reads docs/design evidence and writes the report",
            "python -m scripts.run_slm229_looped_latent_differentiation --mode fixture",
            "```",
            "",
            "## Non-goals",
            "",
            "- No model/data implementation or checkpoint.",
            "- No claim that LOTUS transfers at this repo's scale.",
            "- No reopening of the full SemanticPlanV1 stack.",
            "- No novelty/SOTA claim unsupported by external comparison; "
            "external-literature claims in this memo are explicitly flagged "
            "as not independently verified in this environment.",
            "",
            "## Limitations",
            "",
            "- This report is a docs/spec audit, not a training or "
            "evaluation run.",
            "- LOTUS (arXiv:2606.31779) claims in this memo are based only "
            "on the paper title/framing given in the issue text; no live web "
            "access was available to verify full-text or abstract details in "
            "this environment.",
            "- A negative/blocked verdict is an explicitly successful "
            "closeout per the issue text, not a failure of this audit.",
            "",
        ]
    )
    return "\n".join(lines)
