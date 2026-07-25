"""SLM-292 (AP-010): default-off hard-valid semantic-contrast objective.

Consumes the SLM-290 ``openui_hard_valid_v1`` corpus
(``src/slm_training/resources/data/eval/openui_hard_valid_v1/pairs.jsonl`` --
see ``docs/design/semantic-contrast-corpus-v1.md``) to pair each admitted
positive OpenUI program with its hard-valid negative counterpart (same
prompt, parser/schema/reference-valid program, but semantically wrong -- one
declared plan factor mutated) and computes ONE preregistered pairwise-margin
contrastive loss over pooled sequence representations.

This module is intentionally framework-thin: it never imports
``slm_training.models.twotower`` and never itself calls a context encoder.
Callers (the TwoTower training loop, or a test) supply a ``rep_fn`` that maps
a list of strings to a ``(len(strings), d_model)`` tensor of pooled sequence
representations. That keeps the pure loss/sampling logic unit-testable
without constructing a full model, and keeps the objective's only production
call site (``TwoTowerModel.training_loss``) a single explicit, default-off
branch -- see ``semantic_contrast_loss_weight`` in ``TwoTowerConfig``.

Wiring-only note (SLM-292 acceptance bar): full promotion requires >=3-seed
replication with meaning-v2 +0.05 absolute (or binder/reference F1 +0.10)
and paired CI excluding zero, plus syntax/contract validity regression
<=0.01. Nothing in this module claims that bar has been met; see
``docs/design/iter-slm292-semantic-contrast-smoke-<date>.md`` for the
fixture-scale evidence actually collected.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "ContrastPairSample",
    "SemanticContrastStepResult",
    "load_contrast_pairs",
    "sample_contrast_pairs",
    "pairwise_margin_contrast_loss",
    "compute_semantic_contrast_step",
    "SUPPORTED_OBJECTIVES",
]

#: The one preregistered objective implemented for SLM-292. ``margin`` is a
#: pairwise-margin loss over cosine distances between pooled sequence
#: representations. InfoNCE / semantic-regret were considered (see the issue)
#: but not implemented in this pass -- keeping scope to the single lower-risk
#: objective the task instructions named as preferred.
SUPPORTED_OBJECTIVES = ("margin",)


@dataclass(frozen=True)
class ContrastPairSample:
    """One admitted hard-valid contrast pair, trimmed to training-relevant fields."""

    pair_id: str
    source_program_id: str
    family: str
    transform_id: str
    severity: str | None
    prompt: str
    positive_openui: str
    negative_openui: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "source_program_id": self.source_program_id,
            "family": self.family,
            "transform_id": self.transform_id,
            "severity": self.severity,
            "prompt": self.prompt,
        }


def load_contrast_pairs(
    path: str | Path,
    *,
    admitted_only: bool = True,
    split: str | None = None,
    families: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[ContrastPairSample]:
    """Stream-parse ``pairs.jsonl`` into trimmed :class:`ContrastPairSample` rows.

    Only ``admitted`` pairs (positive passes ``binding_aware_meaningful_v2``,
    matched negative fails it -- see ``docs/design/semantic-contrast-corpus-v1.md``)
    are usable hard-valid contrasts by construction; non-admitted rows (for
    example ``positive_control_identity`` rows where the "negative" side
    still passed) are excluded by default.
    """
    path = Path(path)
    families_set = set(families) if families is not None else None
    out: list[ContrastPairSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if admitted_only and not row.get("admitted"):
                continue
            family = row.get("family")
            if families_set is not None and family not in families_set:
                continue
            positive = row.get("positive") or {}
            negative = row.get("negative") or {}
            pos_record = positive.get("record") or {}
            neg_record = negative.get("record") or {}
            if split is not None and pos_record.get("split") != split:
                continue
            prompt = pos_record.get("prompt")
            positive_openui = pos_record.get("openui")
            negative_openui = neg_record.get("openui")
            if not prompt or not positive_openui or not negative_openui:
                continue
            out.append(
                ContrastPairSample(
                    pair_id=str(row.get("pair_id")),
                    source_program_id=str(row.get("source_program_id")),
                    family=str(family),
                    transform_id=str(row.get("transform_id")),
                    severity=positive.get("severity"),
                    prompt=str(prompt),
                    positive_openui=str(positive_openui),
                    negative_openui=str(negative_openui),
                )
            )
            if limit is not None and len(out) >= limit:
                break
    return out


def sample_contrast_pairs(
    pairs: Sequence[ContrastPairSample],
    n: int,
    rng: random.Random,
    *,
    family_weights: Sequence[tuple[str, float]] = (),
) -> list[ContrastPairSample]:
    """Deterministic (seeded) sample of ``n`` pairs from ``pairs``.

    With no ``family_weights``, samples uniformly at random (with
    replacement once the pool is smaller than ``n``, without replacement
    otherwise). With ``family_weights``, first draws a family per weight
    (families absent from the pool are ignored) then a uniform-random pair
    within that family.
    """
    if not pairs:
        raise ValueError("cannot sample contrast pairs from an empty corpus")
    if n <= 0:
        return []
    if not family_weights:
        if n <= len(pairs):
            return rng.sample(list(pairs), n)
        return [rng.choice(pairs) for _ in range(n)]

    by_family: dict[str, list[ContrastPairSample]] = {}
    for sample in pairs:
        by_family.setdefault(sample.family, []).append(sample)
    weighted = [(fam, w) for fam, w in family_weights if fam in by_family and w > 0.0]
    if not weighted:
        # Named families are absent from the pool -- fall back to uniform
        # rather than silently sampling nothing.
        return sample_contrast_pairs(pairs, n, rng)
    families = [fam for fam, _ in weighted]
    weights = [w for _, w in weighted]
    chosen: list[ContrastPairSample] = []
    for _ in range(n):
        family = rng.choices(families, weights=weights, k=1)[0]
        chosen.append(rng.choice(by_family[family]))
    return chosen


def pairwise_margin_contrast_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise margin loss over cosine distance to a shared (prompt) anchor.

    ``loss_i = relu(margin + dist(anchor_i, positive_i) - dist(anchor_i, negative_i))``

    Pulls the pooled representation of the gold (binder/reference-valid)
    program toward the prompt anchor and pushes the hard-valid-but-wrong
    mutation's representation away from it by at least ``margin``.

    Returns ``(loss, positive_distances, negative_distances)`` -- the two
    distance tensors are logged (never used for a second backward pass).
    """
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError(
            "anchor/positive/negative representations must share shape, got "
            f"{tuple(anchor.shape)}, {tuple(positive.shape)}, {tuple(negative.shape)}"
        )
    pos_dist = 1.0 - F.cosine_similarity(anchor, positive, dim=-1)
    neg_dist = 1.0 - F.cosine_similarity(anchor, negative, dim=-1)
    per_pair = F.relu(margin + pos_dist - neg_dist)
    return per_pair.mean(), pos_dist, neg_dist


@dataclass(frozen=True)
class SemanticContrastStepResult:
    """Everything the acceptance criteria require to be logged for one step."""

    loss: torch.Tensor
    loss_weight: float
    objective: str
    margin: float
    temperature: float | None
    num_pairs: int
    sampling_seed: int
    family_counts: dict[str, int]
    transform_counts: dict[str, int]
    pair_ids: tuple[str, ...]
    positive_distances: tuple[float, ...]
    negative_distances: tuple[float, ...]
    corpus_path: str
    split: str | None

    def metrics_dict(self) -> dict[str, Any]:
        """Flat scalar dict for ``TwoTowerModel.last_training_metrics``."""
        pos = self.positive_distances
        neg = self.negative_distances
        return {
            "semantic_contrast_loss": float(self.loss.detach().cpu()),
            "semantic_contrast_loss_weight": self.loss_weight,
            "semantic_contrast_objective": self.objective,
            "semantic_contrast_margin": self.margin,
            "semantic_contrast_temperature": self.temperature,
            "semantic_contrast_pairs": self.num_pairs,
            "semantic_contrast_sampling_seed": self.sampling_seed,
            "semantic_contrast_family_counts": dict(self.family_counts),
            "semantic_contrast_transform_counts": dict(self.transform_counts),
            "semantic_contrast_positive_distance_mean": (
                sum(pos) / len(pos) if pos else None
            ),
            "semantic_contrast_negative_distance_mean": (
                sum(neg) / len(neg) if neg else None
            ),
            "semantic_contrast_corpus_path": self.corpus_path,
            "semantic_contrast_split": self.split,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full structured record (for a JSON experiment report)."""
        d = self.metrics_dict()
        d["semantic_contrast_pair_ids"] = list(self.pair_ids)
        d["semantic_contrast_positive_distances"] = list(self.positive_distances)
        d["semantic_contrast_negative_distances"] = list(self.negative_distances)
        return d


def compute_semantic_contrast_step(
    pairs: Sequence[ContrastPairSample],
    rep_fn: Callable[[list[str]], torch.Tensor],
    *,
    objective: str = "margin",
    margin: float = 0.2,
    temperature: float | None = None,
    weight: float,
    batch_pairs: int,
    seed: int,
    step: int = 0,
    family_weights: Sequence[tuple[str, float]] = (),
    corpus_path: str = "",
    split: str | None = None,
) -> SemanticContrastStepResult:
    """Sample a mini-batch of hard-valid contrast pairs and score one objective.

    ``rep_fn`` is called exactly once on the concatenation of
    ``[prompts, positives, negatives]`` (each of length ``len(sample)``) so a
    caller backed by a shared encoder (e.g. ``TwoTowerModel._encode_context``)
    pays for a single extra forward pass, not three.
    """
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"unsupported semantic_contrast objective {objective!r}; "
            f"supported: {SUPPORTED_OBJECTIVES}"
        )
    if batch_pairs <= 0:
        raise ValueError("batch_pairs must be positive")

    # Deterministic combination of (seed, step) -- random.Random only accepts
    # None/int/float/str/bytes/bytearray as a seed, not an arbitrary tuple.
    rng = random.Random(f"{int(seed)}:{int(step)}")
    sample = sample_contrast_pairs(
        pairs, batch_pairs, rng, family_weights=family_weights
    )
    n = len(sample)
    prompts = [s.prompt for s in sample]
    positives = [s.positive_openui for s in sample]
    negatives = [s.negative_openui for s in sample]
    reps = rep_fn(prompts + positives + negatives)
    if reps.shape[0] != 3 * n:
        raise ValueError(
            f"rep_fn returned {reps.shape[0]} rows for {3 * n} inputs "
            f"(3 * {n} sampled pairs)"
        )
    anchor, positive, negative = reps[:n], reps[n : 2 * n], reps[2 * n : 3 * n]

    loss, pos_dist, neg_dist = pairwise_margin_contrast_loss(
        anchor, positive, negative, margin
    )

    family_counts = Counter(s.family for s in sample)
    transform_counts = Counter(s.transform_id for s in sample)
    return SemanticContrastStepResult(
        loss=loss,
        loss_weight=float(weight),
        objective=objective,
        margin=float(margin),
        temperature=temperature,
        num_pairs=n,
        sampling_seed=int(seed),
        family_counts=dict(family_counts),
        transform_counts=dict(transform_counts),
        pair_ids=tuple(s.pair_id for s in sample),
        positive_distances=tuple(float(v) for v in pos_dist.detach().cpu().tolist()),
        negative_distances=tuple(float(v) for v in neg_dist.detach().cpu().tolist()),
        corpus_path=str(corpus_path),
        split=split,
    )
