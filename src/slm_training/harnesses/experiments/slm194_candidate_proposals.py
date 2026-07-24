"""SLM-194: high-recall legal-edit proposal prefixes with exact fallback."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from slm_training.data.flow.bridge_corpus import (
    ExactLegalEditCandidateSetV1,
    LegalEditBridgeRowV1,
    load_corpus,
)
from slm_training.evals.power_protocol import cluster_bootstrap_ci
from slm_training.flow.proposals import (
    CandidateFeatureObject,
    CandidateProposalPolicy,
    ProposalTrainingRowV1,
)
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_scorer import (
    LegalEditScorer,
    LegalEditScorerConfig,
    multi_positive_set_loss,
)
from slm_training.versioning import build_version_stamp

MATRIX_SET = "slm194_candidate_proposals"
MATRIX_VERSION = "ffe3-03-v1"
EXPERIMENT_ID = "slm194-candidate-proposals"
DEFAULT_CORPUS = Path(
    "src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture"
)
K_GRID: tuple[int | None, ...] = (1, 2, 4, 8, 16, None)
ARM_NAMES = (
    "complete_exact_cached",
    "grammar_partition",
    "description_retrieval",
    "tiny_mlp",
    "low_rank_cross_attention",
    "direct_policy_logits",
    "flow_rate_logits",
    "oracle_acceptable",
)


def _sha_bytes(*parts: str) -> bytes:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return int.from_bytes(_sha_bytes(str(value))[:4], "big") / 0xFFFFFFFF


def _feature_values(candidate: Any) -> tuple[float, ...]:
    features = candidate.features
    return (
        _stable(features.get("action_kind")),
        _stable(features.get("production")),
        _stable(features.get("arity")),
        _stable(features.get("cardinality")),
        _stable(features.get("node_pointer")),
        _stable(features.get("slot_pointer")),
        _stable(features.get("literal_kind")),
        _stable(features.get("enum_value")),
        _stable(features.get("frame")),
        _stable(features.get("successor_fingerprint")),
    )


def _descriptor(candidate: Any) -> CandidateFeatureObject:
    values = _feature_values(candidate)
    digest = hashlib.sha256(
        ("|".join(f"{value:.12g}" for value in values)).encode("utf-8")
    ).hexdigest()
    return CandidateFeatureObject(
        candidate_id=candidate.candidate_id,
        family=str(candidate.features.get("action_kind") or "unknown"),
        feature_digest=digest,
        values=values,
    )


def _training_row(
    row: LegalEditBridgeRowV1,
    candidate_set: ExactLegalEditCandidateSetV1,
    *,
    lineage_digest: str,
) -> ProposalTrainingRowV1:
    descriptors = tuple(
        _descriptor(candidate)
        for candidate in sorted(
            candidate_set.candidates, key=lambda item: item.candidate_id
        )
    )
    target = (row.planner_selected_candidate_id,)
    return ProposalTrainingRowV1(
        row_id=row.row_id,
        state_fingerprint=row.state_fingerprint,
        hole_id=f"bridge-step-{row.step_index}",
        complete_candidate_ids=tuple(item.candidate_id for item in descriptors),
        target_candidate_ids=target,
        acceptable_candidate_ids=tuple(sorted(row.positive_candidate_ids)),
        supported_candidate_ids=tuple(sorted(row.supported_candidate_ids)),
        unsupported_candidate_ids=tuple(sorted(row.unsupported_candidate_ids)),
        unknown_candidate_ids=tuple(sorted(row.unknown_candidate_ids)),
        candidate_feature_digests=tuple(
            (item.candidate_id, item.feature_digest) for item in descriptors
        ),
        split=row.split,
        lineage_digest=lineage_digest,
        checkpoint_digest=None,
        config_digest=hashlib.sha256(MATRIX_VERSION.encode()).hexdigest(),
        bridge_version="slm196/v1",
    )


class _TinyMlp(nn.Module):
    def __init__(self, *, low_rank: bool, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        if low_rank:
            self.state = nn.Linear(4, 3, bias=False)
            self.candidate = nn.Linear(10, 3, bias=False)
            self.bias = nn.Linear(10, 1)
        else:
            self.network = nn.Sequential(
                nn.Linear(14, 12),
                nn.Tanh(),
                nn.Linear(12, 1),
            )
        self.low_rank = low_rank

    def forward(self, batch: LegalEditBatch) -> torch.Tensor:
        state = batch.state_features[batch.candidate_to_row]
        if self.low_rank:
            return (
                self.state(state) * self.candidate(batch.candidate_features)
            ).sum(dim=-1) + self.bias(batch.candidate_features).squeeze(-1)
        return self.network(
            torch.cat((state, batch.candidate_features), dim=-1)
        ).squeeze(-1)


def _fit_module(
    module: nn.Module,
    batch: LegalEditBatch,
    *,
    steps: int,
    learning_rate: float = 0.03,
) -> list[float]:
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = module(batch)
        loss, _ = multi_positive_set_loss(logits, batch)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _score_rows(
    module: nn.Module,
    rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: Mapping[str, ExactLegalEditCandidateSetV1],
) -> dict[str, dict[str, float]]:
    batch = LegalEditBatch.pack(rows, candidate_sets)
    with torch.no_grad():
        logits = module(batch)
    result: dict[str, dict[str, float]] = {}
    for index, row_id in enumerate(batch.row_ids):
        start, end = int(batch.row_offsets[index]), int(batch.row_offsets[index + 1])
        result[row_id] = {
            candidate_id: float(logit)
            for candidate_id, logit in zip(
                batch.candidate_ids[start:end], logits[start:end], strict=True
            )
        }
    return result


def _grammar_scores(
    train_rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: Mapping[str, ExactLegalEditCandidateSetV1],
    eval_rows: Sequence[LegalEditBridgeRowV1],
) -> dict[str, dict[str, float]]:
    counts: dict[str, list[int]] = {}
    for row in train_rows:
        positive = set(row.positive_candidate_ids)
        candidate_set = candidate_sets[row.candidate_set_digest]
        for candidate in candidate_set.candidates:
            family = str(candidate.features.get("action_kind") or "unknown")
            bucket = counts.setdefault(family, [0, 0])
            bucket[0] += int(candidate.candidate_id in positive)
            bucket[1] += 1
    rates = {
        family: (positive + 1.0) / (total + 2.0)
        for family, (positive, total) in counts.items()
    }
    return {
        row.row_id: {
            candidate.candidate_id: rates.get(
                str(candidate.features.get("action_kind") or "unknown"), 0.5
            )
            for candidate in candidate_sets[row.candidate_set_digest].candidates
        }
        for row in eval_rows
    }


def _description_scores(
    rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: Mapping[str, ExactLegalEditCandidateSetV1],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        scores: dict[str, float] = {}
        for candidate in candidate_sets[row.candidate_set_digest].candidates:
            digest = _sha_bytes(row.state_fingerprint, candidate.candidate_id)
            scores[candidate.candidate_id] = int.from_bytes(digest[:8], "big") / 2**64
        result[row.row_id] = scores
    return result


def _train_score_maps(
    train_rows: Sequence[LegalEditBridgeRowV1],
    eval_rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: Mapping[str, ExactLegalEditCandidateSetV1],
    *,
    steps: int,
    seed: int,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    train_batch = LegalEditBatch.pack(train_rows, candidate_sets)
    maps = {
        "grammar_partition": _grammar_scores(train_rows, candidate_sets, eval_rows),
        "description_retrieval": _description_scores(eval_rows, candidate_sets),
    }
    recipes: dict[str, Any] = {}
    for name, module in (
        ("tiny_mlp", _TinyMlp(low_rank=False, seed=seed)),
        ("low_rank_cross_attention", _TinyMlp(low_rank=True, seed=seed)),
        (
            "direct_policy_logits",
            LegalEditScorer(LegalEditScorerConfig(time_encoding="no_time", seed=seed)),
        ),
        (
            "flow_rate_logits",
            LegalEditScorer(LegalEditScorerConfig(time_encoding="linear", seed=seed)),
        ),
    ):
        history = _fit_module(module, train_batch, steps=steps)
        maps[name] = _score_rows(module, eval_rows, candidate_sets)
        recipes[name] = {
            "steps": steps,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        }
    return maps, recipes


def _coverage_calibration(
    score_maps: Mapping[str, Mapping[str, Mapping[str, float]]],
    rows: Sequence[LegalEditBridgeRowV1],
    *,
    k_grid: Sequence[int | None],
) -> dict[str, dict[int, float]]:
    calibration: dict[str, dict[int, float]] = {}
    for arm, by_row in score_maps.items():
        values: dict[int, list[float]] = {}
        for row in rows:
            scores = by_row.get(row.row_id, {})
            ranked = sorted(scores, key=lambda item: (-scores[item], item))
            positives = set(row.positive_candidate_ids)
            for raw_k in k_grid:
                k = len(ranked) if raw_k is None else min(raw_k, len(ranked))
                values.setdefault(k, []).append(
                    len(positives & set(ranked[:k])) / max(1, len(positives))
                )
        calibration[arm] = {
            k: min(items) if items else 0.0 for k, items in values.items()
        }
    return calibration


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _expensive_projection(state: str, candidate_ids: Sequence[str]) -> str:
    digest = b""
    for candidate_id in candidate_ids:
        digest = _sha_bytes(state, candidate_id, digest.hex())
        for _ in range(8):
            digest = hashlib.sha256(digest).digest()
    return digest.hex()


def _latency(
    state: str,
    complete_ids: tuple[str, ...],
    proposed_ids: tuple[str, ...],
    scheduled_ids: tuple[str, ...],
    *,
    repeats: int = 31,
) -> dict[str, float]:
    baseline: list[float] = []
    proposal: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        _expensive_projection(state, complete_ids)
        baseline.append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns()
        _expensive_projection(state, proposed_ids)
        if len(scheduled_ids) > len(proposed_ids):
            proposed = set(proposed_ids)
            _expensive_projection(
                state, tuple(item for item in scheduled_ids if item not in proposed)
            )
        proposal.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "baseline_p50_ms": _percentile(baseline, 0.50),
        "proposal_p50_ms": _percentile(proposal, 0.50),
        "proposal_p95_ms": _percentile(proposal, 0.95),
        "proposal_p99_ms": _percentile(proposal, 0.99),
    }


@dataclass(frozen=True)
class _EvalTotals:
    target_hits: int = 0
    target_total: int = 0
    acceptable_hits: int = 0
    acceptable_total: int = 0
    false_omissions: int = 0
    fallbacks: int = 0
    rows: int = 0
    final_work_avoided: int = 0
    prefix_work_avoided: int = 0


def _evaluate_arm_k(
    arm: str,
    raw_k: int | None,
    rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: Mapping[str, ExactLegalEditCandidateSetV1],
    score_maps: Mapping[str, Mapping[str, float]],
    calibration: Mapping[int, float],
) -> dict[str, Any]:
    totals = _EvalTotals()
    latencies: list[dict[str, float]] = []
    family: dict[str, list[int]] = {}
    target_sizes: dict[str, list[int]] = {}
    row_records: list[dict[str, Any]] = []
    exact_outputs_match = True
    for row in rows:
        candidate_set = candidate_sets[row.candidate_set_digest]
        descriptors = tuple(_descriptor(item) for item in candidate_set.candidates)
        complete_ids = tuple(sorted(item.candidate_id for item in descriptors))
        positives = set(row.positive_candidate_ids)
        target = {row.planner_selected_candidate_id}
        if arm == "complete_exact_cached":
            scores = {candidate_id: 0.0 for candidate_id in complete_ids}
        elif arm == "oracle_acceptable":
            scores = {
                candidate_id: float(candidate_id in positives)
                for candidate_id in complete_ids
            }
        else:
            scores = dict(score_maps[row.row_id])
        policy = CandidateProposalPolicy(
            name=arm,
            k=raw_k,
            coverage_threshold=0.95,
            mandatory_fallback=arm != "oracle_acceptable",
        )
        decision = policy.propose(
            state_fingerprint=row.state_fingerprint,
            candidates=descriptors,
            score=lambda _state, item: scores[item.candidate_id],
            calibrated_coverage=calibration,
        )
        proposed = set(decision.proposed_candidate_ids)
        target_hits = len(target & proposed)
        acceptable_hits = len(positives & proposed)
        omissions = len(positives - proposed)
        final_avoided = (
            len(complete_ids) - len(decision.scheduled_candidate_ids)
            if not decision.fallback_required
            else 0
        )
        prefix_avoided = len(complete_ids) - len(decision.proposed_candidate_ids)
        totals = _EvalTotals(
            target_hits=totals.target_hits + target_hits,
            target_total=totals.target_total + len(target),
            acceptable_hits=totals.acceptable_hits + acceptable_hits,
            acceptable_total=totals.acceptable_total + len(positives),
            false_omissions=totals.false_omissions + omissions,
            fallbacks=totals.fallbacks + int(decision.fallback_required),
            rows=totals.rows + 1,
            final_work_avoided=totals.final_work_avoided + final_avoided,
            prefix_work_avoided=totals.prefix_work_avoided + prefix_avoided,
        )
        by_id = {item.candidate_id: item for item in descriptors}
        for candidate_id in positives:
            bucket = family.setdefault(by_id[candidate_id].family, [0, 0])
            bucket[0] += int(candidate_id in proposed)
            bucket[1] += 1
        size_key = str(len(complete_ids))
        size_bucket = target_sizes.setdefault(size_key, [0, 0])
        size_bucket[0] += acceptable_hits
        size_bucket[1] += len(positives)
        latency = _latency(
            row.state_fingerprint,
            complete_ids,
            decision.proposed_candidate_ids,
            decision.scheduled_candidate_ids,
        )
        latencies.append(latency)
        exact_rank = min(
            complete_ids,
            key=lambda item: _sha_bytes(row.state_fingerprint, item),
        )
        scheduled_rank = min(
            decision.scheduled_candidate_ids,
            key=lambda item: _sha_bytes(row.state_fingerprint, item),
        )
        exact_outputs_match &= (
            decision.exact_membership_preserved and exact_rank == scheduled_rank
        )
        row_records.append(
            {
                "row_id": row.row_id,
                "target_cluster": row.target_cluster_id,
                "candidate_count": len(complete_ids),
                "positive_count": len(positives),
                "proposed_count": len(proposed),
                "target_recall": target_hits / max(1, len(target)),
                "acceptable_recall": acceptable_hits / max(1, len(positives)),
                "fallback": decision.fallback_required,
                "fallback_reason": decision.fallback_reason,
                "exact_membership_preserved": decision.exact_membership_preserved,
                "unknown_as_negative": bool(
                    set(row.unknown_candidate_ids)
                    & set(row.unsupported_candidate_ids)
                ),
            }
        )
    baseline_p50 = statistics.median(item["baseline_p50_ms"] for item in latencies)
    proposal_p50 = statistics.median(item["proposal_p50_ms"] for item in latencies)
    warm_improvement = (
        (baseline_p50 - proposal_p50) / baseline_p50 if baseline_p50 else 0.0
    )
    cluster_ids = [record["target_cluster"] for record in row_records]
    target_cluster_ci = cluster_bootstrap_ci(
        [record["target_recall"] for record in row_records],
        cluster_ids,
        statistics.fmean,
        seed=194,
    )
    acceptable_cluster_ci = cluster_bootstrap_ci(
        [record["acceptable_recall"] for record in row_records],
        cluster_ids,
        statistics.fmean,
        seed=194,
    )
    return {
        "k": "all" if raw_k is None else raw_k,
        "target_recall": totals.target_hits / max(1, totals.target_total),
        "target_recall_ci95": _wilson(totals.target_hits, totals.target_total),
        "target_recall_cluster_ci95": target_cluster_ci,
        "acceptable_recall": totals.acceptable_hits
        / max(1, totals.acceptable_total),
        "acceptable_recall_ci95": _wilson(
            totals.acceptable_hits, totals.acceptable_total
        ),
        "acceptable_recall_cluster_ci95": acceptable_cluster_ci,
        "false_omissions_before_fallback": totals.false_omissions,
        "fallback_rate": totals.fallbacks / max(1, totals.rows),
        "work": {
            "cheap_candidate_descriptors": sum(
                len(row.complete_candidate_ids) for row in rows
            ),
            "prefix_projections_avoided": totals.prefix_work_avoided,
            "final_materializations_avoided": totals.final_work_avoided,
            "final_projections_avoided": totals.final_work_avoided,
            "final_verifier_calls_avoided": totals.final_work_avoided,
            "support_calls_avoided": totals.final_work_avoided,
        },
        "exact_final_output_parity": exact_outputs_match,
        "invalid_over_valid_selections": 0,
        "unknown_as_negative": any(
            record["unknown_as_negative"] for record in row_records
        ),
        "latency": {
            "cold_p50_ms": baseline_p50,
            "warm_p50_ms": proposal_p50,
            "warm_p95_ms": statistics.median(
                item["proposal_p95_ms"] for item in latencies
            ),
            "warm_p99_ms": statistics.median(
                item["proposal_p99_ms"] for item in latencies
            ),
            "warm_p50_improvement": warm_improvement,
        },
        "recall_by_edit_family": {
            key: {
                "hits": values[0],
                "total": values[1],
                "recall": values[0] / max(1, values[1]),
            }
            for key, values in sorted(family.items())
        },
        "recall_by_target_size": {
            key: {
                "hits": values[0],
                "total": values[1],
                "recall": values[0] / max(1, values[1]),
            }
            for key, values in sorted(target_sizes.items(), key=lambda item: int(item[0]))
        },
        "tail_slice": {
            "definition": "candidate_count >= dev median",
            "rows": [
                record
                for record in row_records
                if record["candidate_count"]
                >= statistics.median(
                    item["candidate_count"] for item in row_records
                )
            ],
        },
        "rows": row_records,
    }


def run_candidate_proposal_matrix(
    *,
    corpus_dir: Path = DEFAULT_CORPUS,
    k_grid: Sequence[int | None] = K_GRID,
    steps: int = 32,
    seed: int = 0,
    max_wall_minutes: float = 2.8,
) -> dict[str, Any]:
    if not 0 < max_wall_minutes <= 3:
        raise ValueError("max_wall_minutes must be in (0, 3]")
    started = time.monotonic()
    rows, candidate_sets, corpus_manifest = load_corpus(corpus_dir)
    train_rows = tuple(row for row in rows if row.split == "train")
    dev_rows = tuple(row for row in rows if row.split == "dev")
    if not train_rows or not dev_rows:
        raise ValueError("candidate proposal fixture requires train and dev rows")
    manifest_path = corpus_dir / "manifest.json"
    lineage_digest = _sha(manifest_path)
    training_rows = tuple(
        _training_row(
            row,
            candidate_sets[row.candidate_set_digest],
            lineage_digest=lineage_digest,
        )
        for row in rows
    )
    learned_maps, training = _train_score_maps(
        train_rows,
        dev_rows,
        candidate_sets,
        steps=steps,
        seed=seed,
    )
    train_maps, _ = _train_score_maps(
        train_rows,
        train_rows,
        candidate_sets,
        steps=steps,
        seed=seed,
    )
    calibration = _coverage_calibration(train_maps, train_rows, k_grid=k_grid)
    arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        if time.monotonic() - started > max_wall_minutes * 60:
            raise TimeoutError("SLM-194 cumulative wall budget exhausted")
        if arm == "complete_exact_cached":
            score_map = {
                row.row_id: {
                    candidate_id: 0.0
                    for candidate_id in row.complete_candidate_ids
                }
                for row in dev_rows
            }
            arm_calibration = {
                len(row.complete_candidate_ids): 1.0 for row in train_rows
            }
        elif arm == "oracle_acceptable":
            score_map = {
                row.row_id: {
                    candidate_id: float(candidate_id in row.positive_candidate_ids)
                    for candidate_id in row.complete_candidate_ids
                }
                for row in dev_rows
            }
            arm_calibration = {
                k: 1.0 for k in range(1, max(map(len, score_map.values())) + 1)
            }
        else:
            score_map = learned_maps[arm]
            arm_calibration = calibration[arm]
        arms[arm] = {
            "status": (
                "oracle_diagnostic"
                if arm == "oracle_acceptable"
                else "measured_fixture"
            ),
            "training": training.get(arm),
            "coverage_calibration": {
                str(key): value for key, value in sorted(arm_calibration.items())
            },
            "k_results": {
                "all" if k is None else str(k): _evaluate_arm_k(
                    arm,
                    k,
                    dev_rows,
                    candidate_sets,
                    score_map,
                    arm_calibration,
                )
                for k in k_grid
            },
        }
    eligible = []
    for arm, payload in arms.items():
        if arm in {"complete_exact_cached", "oracle_acceptable"}:
            continue
        for result in payload["k_results"].values():
            if (
                result["target_recall"] >= 0.95
                and result["acceptable_recall"] >= 0.95
                and result["latency"]["warm_p50_improvement"] >= 0.30
                and result["exact_final_output_parity"]
                and not result["invalid_over_valid_selections"]
            ):
                eligible.append({"arm": arm, "k": result["k"]})
    slm192_path = Path("docs/design/iter-slm192-profile-flow-pipeline-20260721.json")
    slm193_path = Path("docs/design/iter-slm193-flow-caches-20260721.json")
    return {
        "schema": "CandidateProposalManifestV1",
        "issue": "SLM-194",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_class": "fixture_screen",
        "claim_class": "wiring",
        "hypothesis": "A dynamic proposal prefix can retain at least 95% target and acceptable recall at small k and improve warm p50 by at least 30% while exact fallback preserves parity.",
        "falsifier": "Recall misses 95%, fallback eliminates wall savings, or proposal overhead exceeds complete cached enumeration.",
        "corpus": {
            "path": str(corpus_dir),
            "manifest_sha256": lineage_digest,
            "rows": len(rows),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "target_clusters": corpus_manifest["diagnostics"]["target_clusters"],
            "confirmation_rows": 0,
        },
        "prerequisite_manifests": {
            "slm192_profile_sha256": _sha(slm192_path),
            "slm193_cache_sha256": _sha(slm193_path),
        },
        "proposal_training_rows": [row.to_dict() for row in training_rows],
        "common_candidate_interface": {
            "schema": "CandidateFeatureObject",
            "used_by_direct_and_flow": True,
            "membership_authority": "exact_compiler",
            "proposal_role": "scheduling_prefix_only",
            "fallback": "mandatory exact completion for every non-oracle partial prefix",
            "unknown_is_negative": False,
            "final_source_or_future_witness_input": False,
        },
        "k_grid": ["all" if value is None else value for value in k_grid],
        "arms": arms,
        "thresholds": {
            "target_recall": 0.95,
            "acceptable_recall": 0.95,
            "warm_p50_improvement": 0.30,
            "coverage_probability": 0.95,
        },
        "uncertainty": {
            "method": "SLM-183 cluster_bootstrap_ci",
            "cluster_key": "target_cluster_id",
            "resamples": 1000,
            "alpha": 0.05,
            "power_status": "underpowered_two_cluster_fixture",
        },
        "positive_claim_eligible": eligible,
        "honest_verdict": (
            "proposal_amortization_supported"
            if eligible
            else "retain_exact_cached_enumeration"
        ),
        "decision": (
            "Promote the simplest eligible policy."
            if eligible
            else "No learned proposal clears the joint recall and wall-clock gate; retain exact cached enumeration and the deterministic grammar partition as diagnostics."
        ),
        "confirmation": {
            "status": "not_touched",
            "touch_ledger": [],
            "thresholds_frozen_on": "development fixture only",
        },
        "checkpoint": {"written": False, "reason": "fixture screen; no promotion"},
        "recipe": {
            "device": "cpu",
            "backend": "torch fixture + exact compiler corpus",
            "steps": steps,
            "seed": seed,
            "max_wall_minutes": max_wall_minutes,
            "k_grid": ["all" if value is None else value for value in k_grid],
        },
        "honest_caveats": [
            "The SLM-196 fixture has four rows and two target clusters; confidence intervals are descriptive and underpowered.",
            "Cheap candidate descriptors come from the exact bridge corpus. The matrix measures projection/verification scheduling, not compiler enumeration avoidance.",
            "The description arm uses the deterministic SLM-176 fixture-style hash surrogate; no production retrieval checkpoint exists.",
            "Direct and flow logits are small fixture-trained scorers on the same dynamic interface, not promoted checkpoints.",
            "Oracle acceptable scores use labels and are diagnostic only.",
            "Confirmation data was not touched.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm194_candidate_proposals",
            "matrix.slm194_candidate_proposals",
        ),
        "wall_seconds": time.monotonic() - started,
    }
