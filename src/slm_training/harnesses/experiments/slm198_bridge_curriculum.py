"""SLM-198 bridge curriculum over the frozen SLM-197 direct policy."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from slm_training.data.flow.bridge_corpus import LegalEditBridgeRowV1, load_corpus
from slm_training.harnesses.experiments.slm197_direct_bridge_policy import (
    DEFAULT_CORPUS,
    DEFAULT_RECORDS,
    _evaluate,
    _free_running,
    _records,
    _schedule,
)
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_scorer import (
    DirectLegalEditPolicy,
    LegalEditScorer,
    LegalEditScorerConfig,
    multi_positive_set_loss,
)

ARMS = (
    "uniform_rows",
    "uniform_targets",
    "length_curriculum",
    "entropy_curriculum",
    "dependency_curriculum",
    "anti_curriculum",
    "oracle_difficulty",
)
SELECTABLE_ARMS = ARMS[:-1]
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class BridgeDifficultyV1:
    row_id: str
    bridge_length: int
    remaining_steps: int
    ast_nodes: int
    ast_depth: int
    ast_max_arity: int
    binders: int
    slots: int
    references: int
    state_variables: int
    candidate_count: int
    candidate_entropy: float
    dependency_width: int
    dependency_scc_count: int
    rare_edit_score: float
    source_target_distance: int
    planner_cost: float
    planner_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BridgeCurriculumStageV1:
    stage_id: int
    exposure_start: int
    exposure_end: int
    difficulty_quantile: float
    eligible_row_ids: tuple[str, ...]
    target_masses: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BridgeCurriculumManifestV1:
    schema: str
    arm: str
    seed: int
    total_exposures: int
    target_first: bool
    randomization: str
    stages: tuple[BridgeCurriculumStageV1, ...]
    final_support_row_ids: tuple[str, ...]
    final_support_digest: str
    final_target_ids: tuple[str, ...]
    final_target_support_digest: str
    source_content_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = [stage.to_dict() for stage in self.stages]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeCurriculumManifestV1":
        return cls(
            schema=str(data["schema"]),
            arm=str(data["arm"]),
            seed=int(data["seed"]),
            total_exposures=int(data["total_exposures"]),
            target_first=bool(data["target_first"]),
            randomization=str(data["randomization"]),
            stages=tuple(
                BridgeCurriculumStageV1(
                    stage_id=int(stage["stage_id"]),
                    exposure_start=int(stage["exposure_start"]),
                    exposure_end=int(stage["exposure_end"]),
                    difficulty_quantile=float(stage["difficulty_quantile"]),
                    eligible_row_ids=tuple(stage["eligible_row_ids"]),
                    target_masses={
                        str(key): float(value)
                        for key, value in stage["target_masses"].items()
                    },
                )
                for stage in data["stages"]
            ),
            final_support_row_ids=tuple(data["final_support_row_ids"]),
            final_support_digest=str(data["final_support_digest"]),
            final_target_ids=tuple(data["final_target_ids"]),
            final_target_support_digest=str(data["final_target_support_digest"]),
            source_content_fingerprint=str(data["source_content_fingerprint"]),
        )


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_difficulties(
    rows: Sequence[LegalEditBridgeRowV1], candidate_sets: dict[str, Any]
) -> dict[str, BridgeDifficultyV1]:
    """Compute deterministic, confirmation-independent scheduling features."""
    action_counts: Counter[str] = Counter()
    actions_by_row: dict[str, list[str]] = {}
    for row in rows:
        candidates = candidate_sets[row.candidate_set_digest].candidates
        actions = [str(candidate.edit.get("action", "unknown")) for candidate in candidates]
        actions_by_row[row.row_id] = actions
        action_counts.update(actions)
    result = {}
    for row in rows:
        summary = row.state_summary
        actions = actions_by_row[row.row_id]
        rare = sum(1.0 / action_counts[action] for action in actions)
        result[row.row_id] = BridgeDifficultyV1(
            row_id=row.row_id,
            bridge_length=row.bridge_length,
            remaining_steps=row.remaining_edit_distance,
            ast_nodes=int(summary.get("node_count", summary.get("statement_count", 0))),
            ast_depth=int(summary.get("max_depth", 0)),
            ast_max_arity=int(summary.get("max_arity", 0)),
            binders=int(summary.get("binder_count", 0)),
            slots=int(summary.get("slot_count", 0)),
            references=int(summary.get("reference_count", 0)),
            state_variables=int(summary.get("state_variable_count", 0)),
            candidate_count=len(row.complete_candidate_ids),
            candidate_entropy=math.log(max(1, len(row.complete_candidate_ids))),
            dependency_width=max((len(group) for group in row.independence_groups), default=0),
            dependency_scc_count=len(row.conflict_groups),
            rare_edit_score=rare,
            source_target_distance=row.bridge_length,
            planner_cost=float(row.cost_profile.get("planner_cost", row.bridge_length)),
            planner_available=row.planner_selected_candidate_id is not None,
        )
    return result


def _difficulty_key(
    arm: str, row: LegalEditBridgeRowV1, difficulty: BridgeDifficultyV1
) -> tuple[float, str]:
    if arm == "length_curriculum":
        score = difficulty.bridge_length + difficulty.remaining_steps / 10
    elif arm == "entropy_curriculum":
        score = difficulty.candidate_entropy + difficulty.rare_edit_score / 10
    elif arm == "dependency_curriculum":
        score = (
            difficulty.dependency_width
            + difficulty.dependency_scc_count
            + difficulty.ast_depth / 10
        )
    elif arm == "anti_curriculum":
        score = (
            difficulty.bridge_length
            + difficulty.candidate_entropy
            + difficulty.dependency_width
            + difficulty.dependency_scc_count
        )
    elif arm == "oracle_difficulty":
        score = float(row.remaining_edit_distance)
    else:
        score = 0.0
    return score, row.row_id


def build_manifest(
    rows: Sequence[LegalEditBridgeRowV1],
    *,
    arm: str,
    seed: int,
    epochs: int,
    source_content_fingerprint: str,
    difficulty: dict[str, BridgeDifficultyV1] | None = None,
) -> BridgeCurriculumManifestV1:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if epochs < 1 or not rows:
        raise ValueError("epochs and rows must be non-zero")
    row_ids = tuple(sorted(row.row_id for row in rows))
    targets = sorted({row.target_cluster_id for row in rows})
    masses = {target: 1 / len(targets) for target in targets}
    total = epochs * len(rows)
    stage_count = min(3, total)
    boundaries = tuple(index * total // stage_count for index in range(stage_count + 1))
    stages = []
    if difficulty is None:
        difficulty = {
            row.row_id: BridgeDifficultyV1(
                row_id=row.row_id,
                bridge_length=row.bridge_length,
                remaining_steps=row.remaining_edit_distance,
                ast_nodes=int(row.state_summary.get("statement_count", 0)),
                ast_depth=0,
                ast_max_arity=0,
                binders=0,
                slots=0,
                references=0,
                state_variables=0,
                candidate_count=len(row.complete_candidate_ids),
                candidate_entropy=math.log(max(1, len(row.complete_candidate_ids))),
                dependency_width=max(
                    (len(group) for group in row.independence_groups), default=0
                ),
                dependency_scc_count=len(row.conflict_groups),
                rare_edit_score=0.0,
                source_target_distance=row.bridge_length,
                planner_cost=float(row.bridge_length),
                planner_available=row.planner_selected_candidate_id is not None,
            )
            for row in rows
        }
    ranked = sorted(
        rows, key=lambda row: _difficulty_key(arm, row, difficulty[row.row_id])
    )
    if arm == "anti_curriculum":
        ranked.reverse()
    for index in range(stage_count):
        quantile = (index + 1) / stage_count
        if arm in {"uniform_rows", "uniform_targets"}:
            eligible = row_ids
        else:
            eligible_count = max(1, math.ceil(len(ranked) * quantile))
            eligible = tuple(sorted(row.row_id for row in ranked[:eligible_count]))
        stages.append(
            BridgeCurriculumStageV1(
                stage_id=index,
                exposure_start=boundaries[index],
                exposure_end=boundaries[index + 1],
                difficulty_quantile=quantile,
                eligible_row_ids=eligible,
                target_masses=masses,
            )
        )
    return BridgeCurriculumManifestV1(
        schema="BridgeCurriculumManifestV1",
        arm=arm,
        seed=seed,
        total_exposures=total,
        target_first=arm != "uniform_rows",
        randomization="seeded row shuffle" if arm == "uniform_rows" else "target->path->state",
        stages=tuple(stages),
        final_support_row_ids=row_ids,
        final_support_digest=_digest(row_ids),
        final_target_ids=tuple(targets),
        final_target_support_digest=_digest(targets),
        source_content_fingerprint=source_content_fingerprint,
    )


def validate_manifest(manifest: BridgeCurriculumManifestV1) -> None:
    if manifest.schema != "BridgeCurriculumManifestV1" or manifest.arm not in ARMS:
        raise ValueError("unsupported bridge curriculum manifest")
    if manifest.total_exposures < 1 or not manifest.stages:
        raise ValueError("manifest needs a positive exposure budget and stages")
    cursor = 0
    for index, stage in enumerate(manifest.stages):
        if stage.stage_id != index or stage.exposure_start != cursor:
            raise ValueError("curriculum stages are not contiguous")
        if stage.exposure_end <= stage.exposure_start:
            raise ValueError("curriculum stage is empty")
        if not stage.eligible_row_ids or not math.isfinite(stage.difficulty_quantile):
            raise ValueError("curriculum stage has invalid eligibility")
        if not math.isclose(sum(stage.target_masses.values()), 1.0):
            raise ValueError("target masses do not sum to one")
        cursor = stage.exposure_end
    if cursor != manifest.total_exposures:
        raise ValueError("curriculum stages do not cover the budget")
    if _digest(manifest.final_support_row_ids) != manifest.final_support_digest:
        raise ValueError("row support digest mismatch")
    if _digest(manifest.final_target_ids) != manifest.final_target_support_digest:
        raise ValueError("target support digest mismatch")
    if set(manifest.stages[-1].eligible_row_ids) != set(
        manifest.final_support_row_ids
    ):
        raise ValueError("final stage does not expose full row support")


class BridgeCurriculumSampler:
    """Deterministic resumable target-first exposure sampler."""

    def __init__(
        self,
        rows: Sequence[LegalEditBridgeRowV1],
        manifest: BridgeCurriculumManifestV1,
        difficulty: dict[str, BridgeDifficultyV1],
        *,
        cursor: int = 0,
    ) -> None:
        self.rows = {row.row_id: row for row in rows}
        self.manifest = manifest
        self.difficulty = difficulty
        self.cursor = cursor
        validate_manifest(manifest)
        self._sequence = self._build_sequence()
        if not 0 <= cursor <= len(self._sequence):
            raise ValueError("resume cursor outside exposure sequence")

    def _build_sequence(self) -> list[LegalEditBridgeRowV1]:
        rows = list(self.rows.values())
        if tuple(sorted(self.rows)) != self.manifest.final_support_row_ids:
            raise ValueError("manifest row support does not match sampler rows")
        targets = sorted({row.target_cluster_id for row in rows})
        if tuple(targets) != self.manifest.final_target_ids:
            raise ValueError("manifest target support does not match sampler rows")
        total = self.manifest.total_exposures
        if self.manifest.arm == "uniform_rows":
            sequence = [row for _ in range(total // len(rows)) for row in rows]
            random.Random(self.manifest.seed).shuffle(sequence)
            return sequence

        quotas = {
            target: total // len(targets) + int(index < total % len(targets))
            for index, target in enumerate(targets)
        }
        by_target: dict[str, list[LegalEditBridgeRowV1]] = defaultdict(list)
        for row in rows:
            by_target[row.target_cluster_id].append(row)
        target_sequences: dict[str, list[LegalEditBridgeRowV1]] = {}
        for target in targets:
            by_path: dict[str, list[LegalEditBridgeRowV1]] = defaultdict(list)
            for row in by_target[target]:
                by_path[row.bridge_id].append(row)
            paths = sorted(by_path)
            path_quotas = {
                path: quotas[target] // len(paths)
                + int(index < quotas[target] % len(paths))
                for index, path in enumerate(paths)
            }
            path_sequences: dict[str, list[LegalEditBridgeRowV1]] = {}
            for path in paths:
                group = by_path[path]
                if self.manifest.arm == "uniform_targets":
                    group.sort(key=lambda row: row.row_id)
                    random.Random(self.manifest.seed + len(target) + len(path)).shuffle(
                        group
                    )
                else:
                    group.sort(
                        key=lambda row: _difficulty_key(
                            self.manifest.arm, row, self.difficulty[row.row_id]
                        ),
                        reverse=self.manifest.arm == "anti_curriculum",
                    )
                path_sequences[path] = [
                    group[index % len(group)] for index in range(path_quotas[path])
                ]
            repeated = []
            while any(path_sequences.values()):
                for path in paths:
                    if path_sequences[path]:
                        repeated.append(path_sequences[path].pop(0))
            # A curriculum consumes all copies of easier states before opening
            # harder states; controls retain seeded row order.
            if self.manifest.arm not in {"uniform_targets"}:
                repeated.sort(
                    key=lambda row: _difficulty_key(
                        self.manifest.arm, row, self.difficulty[row.row_id]
                    ),
                    reverse=self.manifest.arm == "anti_curriculum",
                )
            target_sequences[target] = repeated
        sequence = []
        while any(target_sequences.values()):
            for target in targets:
                if target_sequences[target]:
                    sequence.append(target_sequences[target].pop(0))
        return sequence

    def __iter__(self) -> "BridgeCurriculumSampler":
        return self

    def __next__(self) -> LegalEditBridgeRowV1:
        if self.cursor >= len(self._sequence):
            raise StopIteration
        row = self._sequence[self.cursor]
        self.cursor += 1
        return row

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "BridgeCurriculumSamplerStateV1",
            "manifest_digest": _digest(self.manifest.to_dict()),
            "cursor": self.cursor,
        }

    @classmethod
    def resume(
        cls,
        rows: Sequence[LegalEditBridgeRowV1],
        manifest: BridgeCurriculumManifestV1,
        difficulty: dict[str, BridgeDifficultyV1],
        state: dict[str, Any],
    ) -> "BridgeCurriculumSampler":
        if state.get("manifest_digest") != _digest(manifest.to_dict()):
            raise ValueError("resume manifest digest mismatch")
        return cls(rows, manifest, difficulty, cursor=int(state["cursor"]))


def _train(
    rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: dict[str, Any],
    manifest: BridgeCurriculumManifestV1,
    difficulty: dict[str, BridgeDifficultyV1],
    *,
    learning_rate: float,
) -> tuple[DirectLegalEditPolicy, dict[str, Any]]:
    torch.manual_seed(manifest.seed)
    scorer = LegalEditScorer(LegalEditScorerConfig(time_encoding="no_time", seed=manifest.seed))
    policy = DirectLegalEditPolicy(scorer)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=learning_rate)
    sampler = BridgeCurriculumSampler(rows, manifest, difficulty)
    losses = []
    exposures: Counter[str] = Counter()
    target_exposures: Counter[str] = Counter()
    path_exposures: Counter[str] = Counter()
    stage_losses: dict[int, list[float]] = defaultdict(list)
    stage_targets: dict[int, Counter[str]] = defaultdict(Counter)
    candidate_tokens = 0
    training_started = time.monotonic()
    for exposure_index, row in enumerate(sampler):
        batch = LegalEditBatch.pack([row], candidate_sets)
        optimizer.zero_grad(set_to_none=True)
        logits = scorer(batch, schedule_progress=_schedule([row]))
        loss, _ = multi_positive_set_loss(logits, batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        stage = next(
            stage
            for stage in manifest.stages
            if stage.exposure_start <= exposure_index < stage.exposure_end
        )
        stage_losses[stage.stage_id].append(losses[-1])
        stage_targets[stage.stage_id][row.target_cluster_id] += 1
        exposures[row.row_id] += 1
        target_exposures[row.target_cluster_id] += 1
        path_exposures[row.bridge_id] += 1
        candidate_tokens += len(row.complete_candidate_ids)
    return policy, {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "exposures": len(losses),
        "candidate_tokens": candidate_tokens,
        "row_exposures": dict(sorted(exposures.items())),
        "target_exposures": dict(sorted(target_exposures.items())),
        "path_exposures": dict(sorted(path_exposures.items())),
        "stage_curves": [
            {
                "stage_id": stage.stage_id,
                "exposure_start": stage.exposure_start,
                "exposure_end": stage.exposure_end,
                "mean_set_mass_loss": sum(stage_losses[stage.stage_id])
                / len(stage_losses[stage.stage_id]),
                "final_set_mass_loss": stage_losses[stage.stage_id][-1],
                "target_exposures": dict(sorted(stage_targets[stage.stage_id].items())),
            }
            for stage in manifest.stages
        ],
        "wall_seconds": time.monotonic() - training_started,
        "cache_hits": sum(
            int(bool(row.cost_profile.get("cache_hit"))) * count
            for row_id, count in exposures.items()
            for row in (next(item for item in rows if item.row_id == row_id),)
        ),
        "sampler_state": sampler.state_dict(),
    }


def run_matrix(
    *,
    corpus_dir: Path = DEFAULT_CORPUS,
    records_path: Path = DEFAULT_RECORDS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = 8,
    learning_rate: float = 0.03,
    max_wall_minutes: float = 2.8,
) -> dict[str, Any]:
    if not 0 < max_wall_minutes <= 3:
        raise ValueError("max_wall_minutes must be in (0, 3]")
    started = time.monotonic()
    rows, candidate_sets, corpus_manifest = load_corpus(corpus_dir)
    train_rows = [row for row in rows if row.split == "train"]
    dev_rows = [row for row in rows if row.split == "dev"]
    train_targets = {row.target_cluster_id for row in train_rows}
    dev_targets = {row.target_cluster_id for row in dev_rows}
    train_groups = {row.split_group for row in train_rows}
    dev_groups = {row.split_group for row in dev_rows}
    if train_targets & dev_targets or train_groups & dev_groups:
        raise ValueError("bridge corpus leaks target or split group across train/dev")
    train_difficulty = compute_difficulties(train_rows, candidate_sets)
    dev_difficulty = compute_difficulties(dev_rows, candidate_sets)
    difficulty = {**train_difficulty, **dev_difficulty}
    records = _records(records_path)
    arms = {}
    manifest_templates: dict[str, Any] = {}
    identities = set()
    for arm in ARMS:
        runs = []
        for seed in seeds:
            if time.monotonic() - started > max_wall_minutes * 60:
                raise TimeoutError("SLM-198 cumulative wall budget exhausted")
            manifest = build_manifest(
                train_rows,
                arm=arm,
                seed=seed,
                epochs=epochs,
                source_content_fingerprint=corpus_manifest["content_fingerprint"],
                difficulty=train_difficulty,
            )
            policy, training = _train(
                train_rows,
                candidate_sets,
                manifest,
                train_difficulty,
                learning_rate=learning_rate,
            )
            identity = policy.scorer.artifact_identity()
            identities.add(identity["param_count"])
            if arm not in manifest_templates:
                manifest_templates[arm] = manifest.to_dict()
            free_running = _free_running(policy, records, seed=seed)
            traces = free_running.pop("traces")
            free_running["trace_digest"] = _digest(traces)
            runs.append(
                {
                    "seed": seed,
                    "manifest": {
                        "schema": manifest.schema,
                        "arm": manifest.arm,
                        "seed": manifest.seed,
                        "total_exposures": manifest.total_exposures,
                        "target_first": manifest.target_first,
                        "final_support_digest": manifest.final_support_digest,
                        "final_target_support_digest": (
                            manifest.final_target_support_digest
                        ),
                        "source_content_fingerprint": (
                            manifest.source_content_fingerprint
                        ),
                        "manifest_digest": _digest(manifest.to_dict()),
                    },
                    "training": training,
                    "evaluation": _evaluate(policy, dev_rows, candidate_sets, arm="D2"),
                    "free_running": free_running,
                    "artifact_identity": {
                        "schema": identity["schema"],
                        "param_count": identity["param_count"],
                        "scorer_id": identity["config"]["scorer_id"],
                    },
                }
            )
        arms[arm] = {
            "status": "development_diagnostic" if arm == "oracle_difficulty" else "measured_fixture",
            "selection_eligible": arm in SELECTABLE_ARMS,
            "runs": runs,
        }
    row_exposure_signatures = {
        _digest(run["training"]["row_exposures"])
        for arm in arms.values()
        for run in arm["runs"]
    }
    balanced_exposure_signatures = {
        _digest(run["training"]["target_exposures"])
        for name, arm in arms.items()
        if name != "uniform_rows"
        for run in arm["runs"]
    }
    token_counts = {
        run["training"]["candidate_tokens"]
        for arm in arms.values()
        for run in arm["runs"]
    }
    split_safe = not (
        {row.target_cluster_id for row in train_rows}
        & {row.target_cluster_id for row in dev_rows}
    )
    return {
        "schema": "BridgeCurriculumMatrixV1",
        "issue": "SLM-198",
        "run_class": "fixture_wiring",
        "claim_class": "wiring",
        "status": "upstream_blocked",
        "honest_verdict": "reject_curriculum_fixture_indistinguishable",
        "arms": arms,
        "manifest_templates": manifest_templates,
        "difficulty": {
            row_id: value.to_dict() for row_id, value in sorted(difficulty.items())
        },
        "matched_controls": {
            "final_support_equal": len(
                {
                    run["manifest"]["final_target_support_digest"]
                    for arm in arms.values()
                    for run in arm["runs"]
                }
            )
            == 1,
            "row_exposure_equal": len(row_exposure_signatures) == 1,
            "balanced_target_exposure_equal": len(balanced_exposure_signatures) == 1,
            "candidate_tokens_equal": len(token_counts) == 1,
            "parameter_count_equal": len(identities) == 1,
            "parameter_counts": sorted(identities),
            "split_safe": split_safe,
            "target_first_explicit": all(
                run["manifest"]["target_first"]
                for arm in SELECTABLE_ARMS[1:]
                for run in arms[arm]["runs"]
            ),
        },
        "recipe": {
            "device": "cpu",
            "backend": "SLM-197 direct legal-edit scorer",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seeds": list(seeds),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_targets": len({row.target_cluster_id for row in train_rows}),
            "dev_targets": len({row.target_cluster_id for row in dev_rows}),
            "max_wall_minutes": max_wall_minutes,
        },
        "inputs": {
            "corpus": str(corpus_dir),
            "content_fingerprint": corpus_manifest["content_fingerprint"],
            "publishable": corpus_manifest["publishable"],
            "records": str(records_path),
        },
        "confirmation": {
            "status": "blocked",
            "selected_arm": None,
            "reasons": [
                "the committed SLM-196 corpus is a non-publishable four-row fixture",
                "the train split contains only one target and two decision rows",
                "no powered long-bridge or deep-structure confirmation slice exists",
                "uniform-target balance and staged curricula are indistinguishable here",
            ],
        },
        "checkpoint": {
            "written": False,
            "reason": "no arm cleared confirmation; no reusable policy was selected",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
