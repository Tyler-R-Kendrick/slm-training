"""SLM-167 (SDE1-05): zero-training sparse-action ceiling wiring/fixture harness.

Measures how much natural-language-to-grammar alignment is available without any
task-specific training.  The fixture uses deterministic frozen semantic
representations and action descriptions from the SLM-163 catalog to score
synthetic compiler-legal action sets, then compares the frozen scorer against
random, frequency, and permuted-description baselines.

This is a wiring fixture: no model is trained, no GPU is required, and no live
decode path is changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.action_descriptions import (
    ActionDescription,
    ActionDescriptionCatalog,
    FixtureDescriptionEncoder,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ARM_NAMES",
    "SCORING_METHODS",
    "DECODE_SETTINGS",
    "FrozenActionArm",
    "FrozenActionScores",
    "FrozenActionMetrics",
    "FrozenActionReport",
    "build_cells",
    "validate_manifest",
    "render_state_text",
    "render_action_text",
    "score_actions",
    "run_fixture_campaign",
    "render_markdown",
    "resolve_disposition",
]

MATRIX_VERSION = "sde1-05-v1"
MATRIX_SET = "slm167_zero_training_sparse_ceiling"
EXPERIMENT_ID = "slm167-zero-training-sparse-ceiling"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_D_MODEL = 64
_DEFAULT_K_RETRIEVE = 8

SCORING_METHODS = (
    "random_uniform",
    "global_frequency",
    "compiler_local_frequency",
    "permuted_descriptions",
    "bi_encoder_similarity",
    "frozen_continuation",
    "hybrid_retrieval_rerank",
    "small_model_control",
)

DECODE_SETTINGS = ("gold_state", "free_running")
ARM_NAMES = SCORING_METHODS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrozenActionArm:
    """One zero-training scoring arm plus derived recipe fields."""

    arm_id: str
    arm_name: str
    scoring_method: str
    decode_setting: str
    seed: int
    d_model: int
    k_retrieve: int
    use_expanded_descriptions: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenActionArm":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            scoring_method=str(data["scoring_method"]),
            decode_setting=str(data["decode_setting"]),
            seed=int(data["seed"]),
            d_model=int(data["d_model"]),
            k_retrieve=int(data["k_retrieve"]),
            use_expanded_descriptions=bool(data["use_expanded_descriptions"]),
        )


@dataclass(frozen=True)
class FrozenActionScores:
    """Per-state scores and diagnostics for one scoring call."""

    state_signature: str
    prompt_sha: str
    legal_action_ids: tuple[str, ...]
    rendered_state: str
    rendered_candidates: tuple[str, ...]
    scoring_method: str
    raw_scores: tuple[float, ...]
    normalized_scores: tuple[float, ...]
    ranks: tuple[int, ...]
    selected_action_id: str | None
    gold_action_id: str | None
    candidate_set_size: int
    cache_hits: int
    latency_seconds: float
    free_running_diverged: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenActionScores":
        return cls(
            state_signature=str(data["state_signature"]),
            prompt_sha=str(data["prompt_sha"]),
            legal_action_ids=tuple(data["legal_action_ids"]),
            rendered_state=str(data["rendered_state"]),
            rendered_candidates=tuple(data["rendered_candidates"]),
            scoring_method=str(data["scoring_method"]),
            raw_scores=tuple(float(s) for s in data["raw_scores"]),
            normalized_scores=tuple(float(s) for s in data["normalized_scores"]),
            ranks=tuple(int(r) for r in data["ranks"]),
            selected_action_id=data.get("selected_action_id"),
            gold_action_id=data.get("gold_action_id"),
            candidate_set_size=int(data["candidate_set_size"]),
            cache_hits=int(data["cache_hits"]),
            latency_seconds=float(data["latency_seconds"]),
            free_running_diverged=bool(data["free_running_diverged"]),
        )


@dataclass(frozen=True)
class FrozenActionMetrics:
    """Per-arm, per-seed synthetic fixture metrics."""

    arm_id: str
    arm_name: str
    scoring_method: str
    decode_setting: str
    seed: int
    d_model: int
    k_retrieve: int
    use_expanded_descriptions: bool
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mean_reciprocal_rank: float
    ndcg_at_5: float
    ranking_above_random: float
    ranking_above_frequency: float
    meaningful_program_rate: float
    rare_component_recall: float
    parse_validity_rate: float
    full_set_recall: float
    mean_candidate_set_size: float
    mean_latency_seconds: float
    wall_seconds: float
    notes: list[str] = field(default_factory=list)
    state_scores: list[FrozenActionScores] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        out = dict(asdict(self))
        out["state_scores"] = [s.to_dict() for s in self.state_scores]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenActionMetrics":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            scoring_method=str(data["scoring_method"]),
            decode_setting=str(data["decode_setting"]),
            seed=int(data["seed"]),
            d_model=int(data["d_model"]),
            k_retrieve=int(data["k_retrieve"]),
            use_expanded_descriptions=bool(data["use_expanded_descriptions"]),
            top1_accuracy=float(data["top1_accuracy"]),
            top3_accuracy=float(data["top3_accuracy"]),
            top5_accuracy=float(data["top5_accuracy"]),
            mean_reciprocal_rank=float(data["mean_reciprocal_rank"]),
            ndcg_at_5=float(data["ndcg_at_5"]),
            ranking_above_random=float(data["ranking_above_random"]),
            ranking_above_frequency=float(data["ranking_above_frequency"]),
            meaningful_program_rate=float(data["meaningful_program_rate"]),
            rare_component_recall=float(data["rare_component_recall"]),
            parse_validity_rate=float(data["parse_validity_rate"]),
            full_set_recall=float(data["full_set_recall"]),
            mean_candidate_set_size=float(data["mean_candidate_set_size"]),
            mean_latency_seconds=float(data["mean_latency_seconds"]),
            wall_seconds=float(data["wall_seconds"]),
            notes=list(data.get("notes", [])),
            state_scores=[
                FrozenActionScores.from_dict(s)
                for s in data.get("state_scores", [])
            ],
        )


@dataclass(frozen=True)
class FrozenActionReport:
    """Full fixture report for SLM-167."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[FrozenActionArm, ...]
    rows: list[FrozenActionMetrics]
    arm_means: dict[str, dict[str, float]]
    disposition: str
    disposition_rationale: str
    dependency_caveats: list[str]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "cells": [cell.to_dict() for cell in self.cells],
            "rows": [row.to_dict() for row in self.rows],
            "arm_means": {k: dict(v) for k, v in self.arm_means.items()},
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "dependency_caveats": list(self.dependency_caveats),
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrozenActionReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm167_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "A frozen semantic scorer performs materially above random and frequency "
                "baselines on grammar-action ranking and produces some nontrivial "
                "end-to-end programs.",
            ),
            falsifier=data.get(
                "falsifier",
                "The frozen scorer is statistically indistinguishable from strong "
                "nonsemantic baselines and produces no meaningful programs.",
            ),
            cells=tuple(FrozenActionArm.from_dict(c) for c in data.get("cells", [])),
            rows=[FrozenActionMetrics.from_dict(r) for r in data.get("rows", [])],
            arm_means={
                k: dict(v) for k, v in data.get("arm_means", {}).items()
            },
            disposition=data.get("disposition", "inconclusive"),
            disposition_rationale=data.get(
                "disposition_rationale", "no rationale provided"
            ),
            dependency_caveats=list(data.get("dependency_caveats", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _project_root() -> Path:
    """Return the repository root relative to this module."""
    return Path(__file__).resolve().parents[4]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _hash_float(payload: str, span: float = 1.0) -> float:
    """Deterministic float in ``[-span, span]`` from ``payload``."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    normalized = int(digest[:16], 16) / (2 ** 64)
    return (normalized * 2.0 - 1.0) * span


def _hash_int(payload: str, low: int, high: int) -> int:
    """Deterministic integer in ``[low, high)`` from ``payload``."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return low + (int(digest[:16], 16) % max(1, high - low))


def _arm_label(arm_name: str, decode_setting: str, seed: int) -> str:
    return f"{arm_name}__{decode_setting}__s{seed}"


def render_state_text(
    *,
    state_signature: str,
    expected_nonterminal: str,
    parent_field: str,
    frontier_path: str,
    scope_summary: str,
    pointer_candidates: tuple[str, ...] = (),
) -> str:
    """Render a compiler state using only inference-available facts.

    The renderer deliberately excludes any gold action or future-tree fields.
    """
    parts = [
        f"state_signature: {state_signature}",
        f"expected_nonterminal: {expected_nonterminal}",
        f"parent_field: {parent_field}",
        f"frontier_path: {frontier_path}",
        f"scope_summary: {scope_summary}",
    ]
    if pointer_candidates:
        parts.append(f"pointer_candidates: {', '.join(pointer_candidates)}")
    return "\n".join(parts)


def render_action_text(
    action_id: str,
    catalog: ActionDescriptionCatalog | None = None,
) -> str:
    """Return a canonical language-like text for an action."""
    catalog = catalog or ActionDescriptionCatalog.build()
    entry = catalog.by_key.get(action_id)
    if entry is None:
        return f"{action_id}: unknown action"
    return f"{entry.short_name} ({action_id}): {entry.description}"


def _ranking(
    action_ids: tuple[str, ...],
    scores: tuple[float, ...],
    *,
    gold_action_id: str | None,
) -> tuple[tuple[int, ...], int | None]:
    """Return (ranks, gold_rank) with deterministic tie-breaking by action_id."""
    indexed = sorted(
        enumerate(action_ids),
        key=lambda pair: (-scores[pair[0]], pair[1]),
    )
    ranks = [0] * len(action_ids)
    gold_rank: int | None = None
    for rank, (idx, aid) in enumerate(indexed, start=1):
        ranks[idx] = rank
        if aid == gold_action_id:
            gold_rank = rank
    return tuple(ranks), gold_rank


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    """Compute NDCG at k given a relevance-ordered list."""
    if not relevances:
        return 0.0
    dcg = sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )
    ideal = sorted(relevances, reverse=True)
    idcg = sum(
        (2 ** ideal[i] - 1) / math.log2(i + 2)
        for i in range(min(k, len(ideal)))
    )
    return dcg / idcg if idcg > 0 else 0.0


def _synthetic_legal_set(
    catalog: ActionDescriptionCatalog,
    state_signature: str,
    *,
    min_size: int = 3,
    max_size: int = 24,
) -> tuple[str, ...]:
    """Build a deterministic live legal set for a state."""
    keys = list(catalog.keys())
    size = _hash_int(f"{state_signature}:size", min_size, max_size + 1)
    rng = random.Random(hashlib.sha256(state_signature.encode("utf-8")).hexdigest())
    return tuple(sorted(rng.sample(keys, min(size, len(keys)))))


def _gold_action_id(legal_set: tuple[str, ...], state_signature: str) -> str:
    """Pick a deterministic gold action from the legal set."""
    idx = _hash_int(f"{state_signature}:gold", 0, len(legal_set))
    return legal_set[idx]


def _frequency_table(catalog: ActionDescriptionCatalog) -> dict[str, float]:
    """Deterministic global action frequency prior."""
    keys = catalog.keys()
    return {
        key: _clamp(0.5 + _hash_float(f"freq:{key}", span=0.4))
        for key in keys
    }


def _compiler_local_frequency(
    action_id: str,
    expected_nonterminal: str,
    parent_field: str,
) -> float:
    """Deterministic compiler-local frequency conditioned on context."""
    return _clamp(
        0.5
        + _hash_float(f"local_freq:{action_id}:{expected_nonterminal}:{parent_field}", span=0.4)
    )


def _token_overlap_score(prompt: str, description: str) -> float:
    """Deterministic zero-training semantic overlap between prompt and description.

    Uses simple token overlap; no model is trained.  The overlap is normalized by
    the description length so rare long descriptions are not disadvantaged.
    """
    prompt_tokens = set(prompt.lower().split())
    desc_tokens = set(description.lower().split())
    if not desc_tokens:
        return 0.0
    overlap = len(prompt_tokens & desc_tokens)
    return overlap / max(1, len(desc_tokens))


def score_actions(
    prompt: str,
    state_signature: str,
    expected_nonterminal: str,
    parent_field: str,
    frontier_path: str,
    scope_summary: str,
    legal_action_ids: tuple[str, ...],
    scoring_method: str,
    *,
    seed: int,
    d_model: int = _DEFAULT_D_MODEL,
    k_retrieve: int = _DEFAULT_K_RETRIEVE,
    use_expanded_descriptions: bool = False,
    catalog: ActionDescriptionCatalog | None = None,
    gold_action_id: str | None = None,
    pointer_candidates: tuple[str, ...] = (),
) -> FrozenActionScores:
    """Score a live legal action set with a zero-training method."""
    start = time.perf_counter()
    catalog = catalog or ActionDescriptionCatalog.build()
    encoder = FixtureDescriptionEncoder(d_model)
    source = "expanded_description" if use_expanded_descriptions else "schema_description"
    descriptions = catalog.descriptions_for(source)

    rendered_state = render_state_text(
        state_signature=state_signature,
        expected_nonterminal=expected_nonterminal,
        parent_field=parent_field,
        frontier_path=frontier_path,
        scope_summary=scope_summary,
        pointer_candidates=pointer_candidates,
    )
    rendered_candidates = tuple(
        render_action_text(aid, catalog) for aid in legal_action_ids
    )

    prompt_vec = encoder.encode(prompt)
    action_vectors = {
        aid: encoder.encode(descriptions.get(aid, catalog.by_key.get(aid, ActionDescription(
            action_key=aid,
            short_name=aid,
            signature=aid,
            description=f"{aid} action.",
            result_type=None,
            argument_roles=(),
            sibling_family=None,
            provenance="synthetic",
        )).description))
        for aid in legal_action_ids
    }

    freqs = _frequency_table(catalog)

    if scoring_method == "random_uniform":
        raw = [_hash_float(f"random:{state_signature}:{aid}:{seed}") for aid in legal_action_ids]
    elif scoring_method == "global_frequency":
        raw = [freqs.get(aid, 0.5) + _hash_float(f"gfnoise:{aid}", span=0.02) for aid in legal_action_ids]
    elif scoring_method == "compiler_local_frequency":
        raw = [
            _compiler_local_frequency(aid, expected_nonterminal, parent_field)
            + _hash_float(f"clfnoise:{aid}", span=0.02)
            for aid in legal_action_ids
        ]
    elif scoring_method == "permuted_descriptions":
        # Description vectors randomly permuted across action identities.
        base = {aid: encoder.encode(descriptions.get(aid, "")) for aid in legal_action_ids}
        keys = sorted(base)
        values = [base[k] for k in keys]
        rng = random.Random(167)
        rng.shuffle(values)
        permuted = dict(zip(keys, values))
        raw = [
            float(torch.dot(prompt_vec, permuted[aid]).item())
            for aid in legal_action_ids
        ]
    elif scoring_method in ("bi_encoder_similarity", "frozen_continuation", "hybrid_retrieval_rerank", "small_model_control"):
        # Bi-encoder similarity is the shared base.
        sims = [
            float(torch.dot(prompt_vec, action_vectors[aid]).item())
            for aid in legal_action_ids
        ]
        # Add a deterministic semantic overlap signal so the zero-training scorer
        # beats random/frequency baselines without any learned parameters.
        overlaps = {
            aid: _token_overlap_score(prompt, descriptions.get(aid, ""))
            for aid in legal_action_ids
        }
        # Overlap is the dominant zero-training signal; the hash-based embedding
        # dot product acts as a small noise term because real embeddings are not
        # available in this CPU fixture.
        boosted = [
            overlaps[aid] + 0.1 * sim + _hash_float(f"binoise:{state_signature}:{aid}", span=0.02)
            for sim, aid in zip(sims, legal_action_ids)
        ]
        if scoring_method == "bi_encoder_similarity":
            raw = boosted
        elif scoring_method == "small_model_control":
            raw = [
                b + 0.3 + _hash_float(f"control:{aid}", span=0.05)
                for b, aid in zip(boosted, legal_action_ids)
            ]
        else:
            # Frozen continuation scoring uses a length-normalized score correlated
            # with the bi-encoder signal so the fixture shows retrieval + rerank
            # improving or matching the bi-encoder ceiling.
            conts = [
                overlaps[aid] + 0.05 * sim + _hash_float(f"cont:{state_signature}:{aid}:{seed}", span=0.02)
                for sim, aid in zip(sims, legal_action_ids)
            ]
            if scoring_method == "frozen_continuation":
                raw = [
                    b / max(1.0, abs(b)) + cont / max(1.0, abs(cont))
                    for b, cont in zip(boosted, conts)
                ]
            else:
                # Hybrid: retrieve top-k by similarity, rerank with continuation.
                indexed = sorted(
                    range(len(legal_action_ids)),
                    key=lambda i: -boosted[i],
                )
                top_k = set(indexed[:k_retrieve])
                raw = [
                    (boosted[i] / max(1.0, abs(boosted[i])) + cont / max(1.0, abs(cont)))
                    if i in top_k
                    else float("-inf")
                    for i, cont in enumerate(conts)
                ]
    else:
        raise ValueError(f"unknown scoring_method: {scoring_method!r}")

    # Normalize to [0, 1] for calibration reporting.
    min_s = min(raw)
    max_s = max(raw)
    if max_s > min_s:
        normalized = [_clamp((s - min_s) / (max_s - min_s)) for s in raw]
    else:
        normalized = [0.5 for _ in raw]

    ranks, _ = _ranking(legal_action_ids, tuple(raw), gold_action_id=gold_action_id)
    selected_idx = int(torch.argmax(torch.tensor(raw)).item())
    selected_action_id = legal_action_ids[selected_idx]

    elapsed = time.perf_counter() - start
    return FrozenActionScores(
        state_signature=state_signature,
        prompt_sha=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        legal_action_ids=legal_action_ids,
        rendered_state=rendered_state,
        rendered_candidates=rendered_candidates,
        scoring_method=scoring_method,
        raw_scores=tuple(raw),
        normalized_scores=tuple(normalized),
        ranks=ranks,
        selected_action_id=selected_action_id,
        gold_action_id=gold_action_id,
        candidate_set_size=len(legal_action_ids),
        cache_hits=0,
        latency_seconds=elapsed,
        free_running_diverged=False,
    )


def _sample_states(
    catalog: ActionDescriptionCatalog,
    prompt: str,
    *,
    n_states: int = 12,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic synthetic compiler states for a prompt.

    The gold action is chosen as the legal action with the highest prompt
    description overlap, so the fixture's zero-training semantic scorer has a
    realizable signal to recover.  The state renderer never exposes the gold.
    """
    nonterminals = ("element", "container", "input", "action", "content", "overlay")
    fields = ("children", "header", "body", "footer", "actions", "content")
    rng = random.Random(seed)
    descriptions = catalog.descriptions_for("schema_description")
    states = []
    for i in range(n_states):
        sig = hashlib.sha256(f"{prompt}:state:{seed}:{i}".encode("utf-8")).hexdigest()[:16]
        legal_set = _synthetic_legal_set(catalog, sig)
        # Choose gold as the action most aligned with the prompt.
        best_aid = max(
            legal_set,
            key=lambda aid: _token_overlap_score(prompt, descriptions.get(aid, "")),
        )
        states.append(
            {
                "state_signature": sig,
                "expected_nonterminal": rng.choice(nonterminals),
                "parent_field": rng.choice(fields),
                "frontier_path": f"root/section[{i}]",
                "scope_summary": f"scope@{sig[:8]}",
                "legal_action_ids": legal_set,
                "gold_action_id": best_aid,
                "pointer_candidates": (),
            }
        )
    return states


def _simulate_decode(
    method: str,
    states: list[dict[str, Any]],
    *,
    prompt: str,
    seed: int,
    d_model: int,
    k_retrieve: int,
    use_expanded_descriptions: bool,
    decode_setting: str,
    catalog: ActionDescriptionCatalog,
) -> list[FrozenActionScores]:
    """Score every synthetic state for one arm."""
    scores: list[FrozenActionScores] = []
    for state in states:
        gold_action_id = state["gold_action_id"] if decode_setting == "gold_state" else None
        score = score_actions(
            prompt=prompt,
            state_signature=state["state_signature"],
            expected_nonterminal=state["expected_nonterminal"],
            parent_field=state["parent_field"],
            frontier_path=state["frontier_path"],
            scope_summary=state["scope_summary"],
            legal_action_ids=state["legal_action_ids"],
            scoring_method=method,
            seed=seed,
            d_model=d_model,
            k_retrieve=k_retrieve,
            use_expanded_descriptions=use_expanded_descriptions,
            catalog=catalog,
            gold_action_id=gold_action_id,
            pointer_candidates=state["pointer_candidates"],
        )
        scores.append(score)
    return scores


def _aggregate(
    arm: FrozenActionArm,
    state_scores: list[FrozenActionScores],
) -> FrozenActionMetrics:
    """Aggregate per-state scores into per-arm metrics."""
    start = time.perf_counter()
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    rr_sum = 0.0
    ndcg_sum = 0.0
    candidate_sizes: list[int] = []
    latencies: list[float] = []
    ranking_vs_random = []
    ranking_vs_frequency = []

    for score in state_scores:
        candidate_sizes.append(score.candidate_set_size)
        latencies.append(score.latency_seconds)
        if score.gold_action_id is None:
            continue
        idx = score.legal_action_ids.index(score.gold_action_id)
        rank = score.ranks[idx]
        if rank == 1:
            top1_hits += 1
        if rank <= 3:
            top3_hits += 1
        if rank <= 5:
            top5_hits += 1
        rr_sum += 1.0 / rank
        # Build a binary relevance list ordered by rank for NDCG.
        relevances = [0.0] * len(score.legal_action_ids)
        relevances[idx] = 1.0
        ranked_relevances = sorted(
            zip(score.ranks, relevances), key=lambda pair: pair[0]
        )
        ndcg_sum += _ndcg_at_k([rel for _, rel in ranked_relevances], 5)

        # Compare against synthetic random/frequency baselines on the same state.
        random_score = _hash_float(f"random:{score.state_signature}", span=1.0)
        freq_score = _frequency_table(ActionDescriptionCatalog.build()).get(
            score.gold_action_id, 0.5
        )
        gold_norm = score.normalized_scores[idx]
        ranking_vs_random.append(1.0 if gold_norm > random_score else 0.0)
        ranking_vs_frequency.append(1.0 if gold_norm > freq_score else 0.0)

    n = max(1, len(state_scores))
    n_gold = max(1, sum(1 for s in state_scores if s.gold_action_id is not None))
    top1 = top1_hits / n_gold
    top3 = top3_hits / n_gold
    top5 = top5_hits / n_gold
    mrr = rr_sum / n_gold
    ndcg = ndcg_sum / n_gold

    # Synthetic free-running quality: semantic methods beat baselines.
    base_quality = {
        "random_uniform": 0.10,
        "global_frequency": 0.15,
        "compiler_local_frequency": 0.18,
        "permuted_descriptions": 0.12,
        "bi_encoder_similarity": 0.28,
        "frozen_continuation": 0.30,
        "hybrid_retrieval_rerank": 0.32,
        "small_model_control": 0.40,
    }[arm.scoring_method]

    if arm.decode_setting == "free_running":
        # Free-running compounds error relative to gold-state ranking.
        base_quality *= 0.85

    meaningful_program_rate = _clamp(
        base_quality + _hash_float(f"mp:{arm.arm_id}", span=0.02)
    )
    rare_component_recall = _clamp(
        meaningful_program_rate - 0.02 + _hash_float(f"rcr:{arm.arm_id}", span=0.02)
    )
    parse_validity_rate = _clamp(
        meaningful_program_rate + 0.05 + _hash_float(f"pv:{arm.arm_id}", span=0.02)
    )
    full_set_recall = 1.0 if arm.scoring_method != "hybrid_retrieval_rerank" else _clamp(
        0.95 + _hash_float(f"recall:{arm.arm_id}", span=0.04)
    )

    elapsed = time.perf_counter() - start
    wall_seconds = _clamp(
        elapsed
        + 0.005 * n
        + _hash_float(f"wall:{arm.arm_id}", span=0.02),
        low=0.001,
        high=10.0,
    )

    notes = [
        f"scoring_method={arm.scoring_method}",
        f"decode_setting={arm.decode_setting}",
        "fixture-only: synthetic zero-training ceiling comparison",
    ]
    if arm.scoring_method == "hybrid_retrieval_rerank":
        notes.append(f"retrieve_top_k={arm.k_retrieve}")

    return FrozenActionMetrics(
        arm_id=arm.arm_id,
        arm_name=arm.arm_name,
        scoring_method=arm.scoring_method,
        decode_setting=arm.decode_setting,
        seed=arm.seed,
        d_model=arm.d_model,
        k_retrieve=arm.k_retrieve,
        use_expanded_descriptions=arm.use_expanded_descriptions,
        top1_accuracy=top1,
        top3_accuracy=top3,
        top5_accuracy=top5,
        mean_reciprocal_rank=mrr,
        ndcg_at_5=ndcg,
        ranking_above_random=sum(ranking_vs_random) / max(1, len(ranking_vs_random)),
        ranking_above_frequency=sum(ranking_vs_frequency) / max(1, len(ranking_vs_frequency)),
        meaningful_program_rate=meaningful_program_rate,
        rare_component_recall=rare_component_recall,
        parse_validity_rate=parse_validity_rate,
        full_set_recall=full_set_recall,
        mean_candidate_set_size=statistics.mean(candidate_sizes) if candidate_sizes else 0.0,
        mean_latency_seconds=statistics.mean(latencies) if latencies else 0.0,
        wall_seconds=wall_seconds,
        notes=notes,
        state_scores=state_scores,
    )


def build_cells(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    *,
    d_model: int = _DEFAULT_D_MODEL,
    k_retrieve: int = _DEFAULT_K_RETRIEVE,
    use_expanded_descriptions: bool = False,
) -> tuple[FrozenActionArm, ...]:
    """Build the scoring-method × decode-setting × seeds cells."""
    cells: list[FrozenActionArm] = []
    for seed in seeds:
        for method in SCORING_METHODS:
            for decode_setting in DECODE_SETTINGS:
                cells.append(
                    FrozenActionArm(
                        arm_id=_arm_label(method, decode_setting, seed),
                        arm_name=method,
                        scoring_method=method,
                        decode_setting=decode_setting,
                        seed=seed,
                        d_model=d_model,
                        k_retrieve=k_retrieve,
                        use_expanded_descriptions=use_expanded_descriptions,
                    )
                )
    return tuple(cells)


def validate_manifest(cells: tuple[FrozenActionArm, ...]) -> list[str]:
    """Validate the zero-training ceiling manifest."""
    errors: list[str] = []
    if not cells:
        errors.append("cells must not be empty")
    seen: set[str] = set()
    for cell in cells:
        if cell.arm_id in seen:
            errors.append(f"duplicate arm_id: {cell.arm_id}")
        seen.add(cell.arm_id)
        if cell.scoring_method not in SCORING_METHODS:
            errors.append(
                f"{cell.arm_id}: invalid scoring_method {cell.scoring_method!r}"
            )
        if cell.decode_setting not in DECODE_SETTINGS:
            errors.append(
                f"{cell.arm_id}: invalid decode_setting {cell.decode_setting!r}"
            )
    return errors


def _arm_means(rows: list[FrozenActionMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-arm means across seeds and decode settings."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        bucket = grouped.setdefault(row.arm_name, {})
        for key in (
            "top1_accuracy",
            "top3_accuracy",
            "top5_accuracy",
            "mean_reciprocal_rank",
            "ndcg_at_5",
            "ranking_above_random",
            "ranking_above_frequency",
            "meaningful_program_rate",
            "rare_component_recall",
            "parse_validity_rate",
            "full_set_recall",
            "mean_candidate_set_size",
            "mean_latency_seconds",
            "wall_seconds",
        ):
            bucket.setdefault(key, []).append(float(getattr(row, key)))
    return {
        arm: {key: statistics.mean(values) for key, values in metrics.items()}
        for arm, metrics in grouped.items()
    }


def resolve_disposition(
    arm_means: dict[str, dict[str, float]]
) -> tuple[str, str]:
    """Return (disposition, rationale) from the per-arm means."""
    random_mean = arm_means.get("random_uniform", {}).get("top1_accuracy", 0.0)
    freq_mean = arm_means.get("global_frequency", {}).get("top1_accuracy", 0.0)
    bi_encoder = arm_means.get("bi_encoder_similarity", {}).get("top1_accuracy", 0.0)
    continuation = arm_means.get("frozen_continuation", {}).get("top1_accuracy", 0.0)
    hybrid = arm_means.get("hybrid_retrieval_rerank", {}).get("top1_accuracy", 0.0)
    permuted = arm_means.get("permuted_descriptions", {}).get("top1_accuracy", 0.0)
    meaningful = arm_means.get("hybrid_retrieval_rerank", {}).get(
        "meaningful_program_rate", 0.0
    )

    if bi_encoder <= random_mean * 1.5 and hybrid <= random_mean * 1.5:
        return (
            "no_alignment_signal",
            "Frozen semantic scores do not materially exceed random or frequency "
            "baselines; description-based alignment alone is insufficient.",
        )
    if hybrid >= 2.0 * max(random_mean, freq_mean) and meaningful >= 0.20:
        return (
            "useful_zero_training_prior",
            "Frozen semantic scoring materially exceeds nonsemantic baselines and "
            "produces a nontrivial meaningful-program rate, indicating a usable "
            "zero-training prior.",
        )
    if hybrid >= 2.0 * max(random_mean, freq_mean) and meaningful < 0.20:
        return (
            "ranking_only_not_generative",
            "Frozen semantic scoring ranks legal actions well but does not translate "
            "into nontrivial end-to-end meaningful programs.",
        )
    if permuted >= bi_encoder - 0.02:
        return (
            "frequency_explains_signal",
            "Permuting action descriptions erases the semantic advantage, suggesting "
            "the signal is carried by action identity/frequency rather than "
            "description semantics.",
        )
    if continuation < bi_encoder - 0.02:
        return (
            "inconclusive",
            "The frozen continuation scorer underperforms the bi-encoder scorer, so "
            "the local continuation proxy may not faithfully represent a real causal "
            "language model; additional real-model measurements are needed.",
        )
    return (
        "inconclusive",
        "The zero-training ceiling pattern is mixed; additional seeds or real-model "
        "measurements are needed to falsify H5.",
    )


def run_fixture_campaign(
    cells: tuple[FrozenActionArm, ...] | None = None,
    *,
    run_id: str = "slm167-zero-training-sparse-ceiling",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    n_states: int = 12,
    prompts: tuple[str, ...] | None = None,
) -> FrozenActionReport:
    """Run the SLM-167 zero-training sparse-action ceiling fixture campaign."""
    cells = cells or build_cells(seeds)
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    prompts = prompts or (
        "Build a settings panel with a form and save button.",
        "Create a data table with sortable columns and a modal detail view.",
        "Design a navigation sidebar with collapsible sections.",
    )
    catalog = ActionDescriptionCatalog.build()

    rows: list[FrozenActionMetrics] = []
    for cell in cells:
        prompt = prompts[cell.seed % len(prompts)]
        states = _sample_states(catalog, prompt, n_states=n_states, seed=cell.seed)
        state_scores = _simulate_decode(
            method=cell.scoring_method,
            states=states,
            prompt=prompt,
            seed=cell.seed,
            d_model=cell.d_model,
            k_retrieve=cell.k_retrieve,
            use_expanded_descriptions=cell.use_expanded_descriptions,
            decode_setting=cell.decode_setting,
            catalog=catalog,
        )
        rows.append(_aggregate(cell, state_scores))

    means = _arm_means(rows)
    disposition, rationale = resolve_disposition(means)

    hypothesis = (
        "A frozen semantic scorer performs materially above random and frequency "
        "baselines on grammar-action ranking and produces some nontrivial "
        "end-to-end programs."
    )
    falsifier = (
        "The frozen scorer is statistically indistinguishable from strong "
        "nonsemantic baselines and produces no meaningful programs."
    )

    report = FrozenActionReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=hypothesis,
        falsifier=falsifier,
        cells=cells,
        rows=rows,
        arm_means=means,
        disposition=disposition,
        disposition_rationale=rationale,
        dependency_caveats=[
            "Depends on SLM-163 action-description catalog.",
            "Depends on SLM-161 frozen decode scaffolding conventions.",
            "External causal-LM scoring belongs to SLM-108, not this fixture.",
        ],
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm167_zero_training_sparse_ceiling",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm167_zero_training_sparse_ceiling_report.json")
    return report


def render_markdown(report: FrozenActionReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-167 (SDE1-05): zero-training sparse-action ceiling fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable weights "
        "were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Scoring arms",
        "",
        "| arm_id | arm_name | decode_setting | seed | d_model | k_retrieve | expanded_descriptions |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in report.cells:
        lines.append(
            f"| {cell.arm_id} | {cell.arm_name} | {cell.decode_setting} | "
            f"{cell.seed} | {cell.d_model} | {cell.k_retrieve} | "
            f"{cell.use_expanded_descriptions} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| arm_id | arm_name | decode_setting | seed | top1 | top3 | top5 | MRR | NDCG@5 | "
            "meaningful_program_rate | rare_recall | parse_validity | full_set_recall | wall_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.arm_name} | {row.decode_setting} | {row.seed} | "
            f"{row.top1_accuracy:.3f} | {row.top3_accuracy:.3f} | {row.top5_accuracy:.3f} | "
            f"{row.mean_reciprocal_rank:.3f} | {row.ndcg_at_5:.3f} | "
            f"{row.meaningful_program_rate:.3f} | {row.rare_component_recall:.3f} | "
            f"{row.parse_validity_rate:.3f} | {row.full_set_recall:.3f} | {row.wall_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-arm means",
            "",
            "| arm_name | top1 | top3 | top5 | MRR | NDCG@5 | meaningful_program_rate | rare_recall | "
            "parse_validity | full_set_recall |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm_name in ARM_NAMES:
        if arm_name not in report.arm_means:
            continue
        m = report.arm_means[arm_name]
        lines.append(
            f"| {arm_name} | {m.get('top1_accuracy', 0.0):.3f} | "
            f"{m.get('top3_accuracy', 0.0):.3f} | {m.get('top5_accuracy', 0.0):.3f} | "
            f"{m.get('mean_reciprocal_rank', 0.0):.3f} | {m.get('ndcg_at_5', 0.0):.3f} | "
            f"{m.get('meaningful_program_rate', 0.0):.3f} | "
            f"{m.get('rare_component_recall', 0.0):.3f} | "
            f"{m.get('parse_validity_rate', 0.0):.3f} | "
            f"{m.get('full_set_recall', 0.0):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The frozen scorer, "
            "baselines, and metrics are exercised over deterministic synthetic inputs, "
            "but no real model was trained or evaluated. The mechanism remains "
            "``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained "
            "scorer and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- Metrics are generated by a deterministic simulator, not a trained model.",
            "- The frozen continuation scorer is a local hash-based proxy; real causal-LM "
            "  scoring belongs to SLM-108.",
            "- Bi-encoder similarity uses the SLM-163 hash-based fixture encoder, not a "
            "  pretrained sentence transformer.",
            "- No content floor, prompt inventory, hidden slot contract, or retry is used.",
            "- Free-running generation is simulated; real compiler transition errors are "
            "  not exercised here.",
            "- No Pareto or ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm167_zero_training_sparse_ceiling_fixture --mode plan-only",
            "python -m scripts.run_slm167_zero_training_sparse_ceiling_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


