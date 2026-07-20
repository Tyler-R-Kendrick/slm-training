"""SLM-154 (SPV3-01): capacity-matched autoregressive compiler-legal action scorer.

This is a fixture/wiring baseline. It implements a small learned scorer that
receives prompt/context features, optional plan features, and the complete live
legal action set, then emits one score per legal action. The compiler remains
authoritative for membership, state transition, and verification.

Variants:
* ``global_head`` — fixed global vocabulary projection followed by an exact legal mask.
* ``mlp`` — independent MLP score per candidate from context/state/plan/candidate embeddings.
* ``cross_attention`` — candidate embeddings attend to context/state keys.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence
import zlib

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - torch may be absent in minimal environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from slm_training.data.semantic_plan.compiler import PlanActionFeatures
from slm_training.lineage.records import content_sha


__all__ = [
    "LEGAL_ACTION_SCORER_SCHEMA_VERSION",
    "LegalActionScorer",
    "LegalActionScorerConfig",
    "LegalActionScores",
    "ScorerDecision",
    "make_fixture_decisions",
    "train_fixture_scorer",
]


LEGAL_ACTION_SCORER_SCHEMA_VERSION = "legal_action_scorer/v1"
ScorerVariant = Literal["global_head", "mlp", "cross_attention"]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("legal_action_scorer requires torch")


def _stable_action_index(action: str, vocab_size: int) -> int:
    """Deterministic action -> output index mapping."""
    return (zlib.crc32(action.encode("utf-8")) & 0xFFFFFFFF) % vocab_size


@dataclass
class LegalActionScorerConfig:
    """Configuration for the fixture legal-action scorer."""

    variant: ScorerVariant = "mlp"
    d_model: int = 64
    hidden_dim: int = 64
    num_heads: int = 2
    dropout: float = 0.0
    max_vocabulary: int = 256
    action_embedding_dim: int = 32
    plan_feature_dim: int = 16
    seed: int = 0
    scorer_id: str = "legal-action-scorer-v1"
    supported_packs: tuple[str, ...] = ("openui",)


@dataclass(frozen=True)
class LegalActionScores:
    """Scorer output for one decision state."""

    scores: dict[str, float]
    logits: torch.Tensor
    legal_actions: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScorerDecision:
    """Decoded decision from a scorer."""

    action_identity: str | None
    decision_kind: Literal["scored", "forced", "abstain"]
    confidence: float = 0.0
    telemetry: dict[str, Any] = field(default_factory=dict)


class _ForcedMixin:
    """Shared forced-singleton handling."""

    @staticmethod
    def _check_forced(legal_actions: list[str]) -> ScorerDecision | None:
        if len(legal_actions) == 1:
            return ScorerDecision(
                action_identity=legal_actions[0],
                decision_kind="forced",
                confidence=1.0,
                telemetry={"skip_reason": "single_legal_action"},
            )
        return None


if nn is not None:

    class _ActionEmbedding(nn.Module):
        """Small learned embedding table keyed by stable action hash."""

        def __init__(self, max_actions: int, dim: int) -> None:
            super().__init__()
            self.max_actions = max_actions
            self.embed = nn.Embedding(max_actions, dim)

        def forward(self, actions: list[str]) -> torch.Tensor:
            indices = torch.tensor(
                [_stable_action_index(a, self.max_actions) for a in actions],
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            return self.embed(indices)

    class _PlanFeatureEncoder(nn.Module):
        """Convert PlanActionFeatures into a fixed-size numeric vector."""

        def __init__(self, out_dim: int) -> None:
            super().__init__()
            self.out_dim = out_dim
            self.project = nn.Linear(8, out_dim)

        def forward(self, features: list[PlanActionFeatures]) -> torch.Tensor:
            _require_torch()
            rows = []
            for pf in features:
                rows.append(
                    [
                        float(pf.matches_predicted_role),
                        float(pf.component_family_compatible),
                        float(pf.expected_coverage_contribution),
                        float(pf.topology_parent_order_compatible),
                        float(pf.cardinality_depth_delta),
                        float(pf.binding_pointer_compatible),
                        float(pf.plan_confidence),
                        float(pf.conflict_or_unknown),
                    ]
                )
            tensor = torch.tensor(rows, dtype=torch.float32, device=next(self.parameters()).device)
            return self.project(tensor)

    class LegalActionScorer(nn.Module, _ForcedMixin):
        """Student-side legal-action scorer baseline."""

        SCHEMA = LEGAL_ACTION_SCORER_SCHEMA_VERSION

        def __init__(
            self,
            config: LegalActionScorerConfig | None = None,
            device: str = "cpu",
        ) -> None:
            super().__init__()
            self.config = config if config is not None else LegalActionScorerConfig()
            torch.manual_seed(self.config.seed)
            self._device = torch.device(device)

            self.action_embed = _ActionEmbedding(
                max_actions=max(256, self.config.max_vocabulary),
                dim=self.config.action_embedding_dim,
            )
            self.plan_encoder = _PlanFeatureEncoder(self.config.plan_feature_dim)

            state_dim = self.config.d_model + self.config.d_model + self.config.plan_feature_dim
            candidate_dim = self.config.action_embedding_dim + self.config.plan_feature_dim

            if self.config.variant == "global_head":
                self.global_logits = nn.Linear(self.config.d_model, self.config.max_vocabulary)
            elif self.config.variant == "mlp":
                self.context_proj = nn.Linear(state_dim, self.config.hidden_dim)
                self.candidate_proj = nn.Linear(candidate_dim, self.config.hidden_dim)
                self.score_mlp = nn.Sequential(
                    nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.hidden_dim, 1),
                )
            elif self.config.variant == "cross_attention":
                self.query_proj = nn.Linear(candidate_dim, self.config.hidden_dim)
                self.key_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
                self.value_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    batch_first=True,
                )
                self.norm = nn.LayerNorm(self.config.hidden_dim)
                self.score_mlp = nn.Sequential(
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.hidden_dim, 1),
                )
            else:
                raise ValueError(f"unknown scorer variant: {self.config.variant}")

            self.to(self._device)

        @property
        def scorer_id(self) -> str:
            return self.config.scorer_id

        def _build_state_vector(
            self,
            context_features: dict[str, Any],
            state_features: dict[str, Any],
            plan_features: dict[str, Any] | None,
        ) -> torch.Tensor:
            """Concatenate deterministic numeric feature vectors."""
            vectors: list[torch.Tensor] = []
            for namespace in (context_features, state_features, plan_features or {}):
                values = []
                for key in sorted(namespace.keys()):
                    val = namespace[key]
                    if isinstance(val, bool):
                        values.append(float(val))
                    elif isinstance(val, (int, float)):
                        f = float(val)
                        if f == f:  # skip NaN
                            values.append(f)
                vectors.append(torch.tensor(values, dtype=torch.float32, device=self._device))

            # Pad/truncate each to expected dimension.
            def _pad(v: torch.Tensor, target: int) -> torch.Tensor:
                if v.numel() >= target:
                    return v[:target]
                return F.pad(v, (0, target - v.numel()))

            context_vec = _pad(vectors[0], self.config.d_model)
            state_vec = _pad(vectors[1], self.config.d_model)
            plan_vec = _pad(vectors[2], self.config.plan_feature_dim)
            return torch.cat([context_vec, state_vec, plan_vec], dim=-1)

        def score(
            self,
            context_features: dict[str, Any],
            state_features: dict[str, Any],
            legal_actions: list[str],
            *,
            plan_features: dict[str, Any] | None = None,
            plan_action_features: list[PlanActionFeatures] | None = None,
            pack_id: str = "openui",
        ) -> LegalActionScores:
            """Score every compiler-supplied legal action."""
            if pack_id not in self.config.supported_packs:
                logits = torch.full((1, len(legal_actions)), float("-inf"), device=self._device)
                return LegalActionScores(
                    scores={a: float("-inf") for a in legal_actions},
                    logits=logits,
                    legal_actions=tuple(legal_actions),
                    metadata={"abstained": True, "reason": "unsupported_pack"},
                )

            if not legal_actions:
                logits = torch.empty((1, 0), device=self._device)
                return LegalActionScores(
                    scores={},
                    logits=logits,
                    legal_actions=(),
                    metadata={"abstained": True, "reason": "empty_legal_set"},
                )

            forced = self._check_forced(legal_actions)
            if forced is not None:
                logits = torch.full((1, len(legal_actions)), float("-inf"), device=self._device)
                logits[0, 0] = 0.0
                return LegalActionScores(
                    scores={legal_actions[0]: 0.0},
                    logits=logits,
                    legal_actions=tuple(legal_actions),
                    metadata={"forced": True},
                )

            state_vec = self._build_state_vector(
                context_features, state_features, plan_features
            ).unsqueeze(0)

            plan_action_features = plan_action_features or [
                PlanActionFeatures(action_id=a) for a in legal_actions
            ]
            plan_action_tensor = self.plan_encoder(plan_action_features)
            action_tensor = self.action_embed(legal_actions)
            candidate_tensor = torch.cat([action_tensor, plan_action_tensor], dim=-1)

            if self.config.variant == "global_head":
                all_logits = self.global_logits(state_vec[:, : self.config.d_model])
                legal_indices = torch.tensor(
                    [_stable_action_index(a, self.config.max_vocabulary) for a in legal_actions],
                    dtype=torch.long,
                    device=self._device,
                )
                logits = all_logits[:, legal_indices]
            elif self.config.variant == "mlp":
                context_hidden = self.context_proj(state_vec)  # [1, H]
                candidate_hidden = self.candidate_proj(candidate_tensor)  # [B, H]
                combined = torch.cat(
                    [
                        context_hidden.expand(candidate_hidden.size(0), -1),
                        candidate_hidden,
                    ],
                    dim=-1,
                )
                logits = self.score_mlp(combined).T  # [1, B]
            elif self.config.variant == "cross_attention":
                queries = self.query_proj(candidate_tensor).unsqueeze(0)  # [1, B, H]
                # Treat the state vector as a short context sequence so attention
                # has more than one key/value to attend over.
                state_total = state_vec.size(-1)
                seq_len = max(1, (state_total + self.config.hidden_dim - 1) // self.config.hidden_dim)
                pad_len = seq_len * self.config.hidden_dim - state_total
                state_padded = F.pad(state_vec, (0, pad_len))
                state_seq = state_padded.view(seq_len, self.config.hidden_dim)
                keys = self.key_proj(state_seq).unsqueeze(0)  # [1, seq_len, H]
                values = self.value_proj(state_seq).unsqueeze(0)
                attn_out, _ = self.attention(queries, keys, values)
                attn_out = self.norm(attn_out + queries)
                logits = self.score_mlp(attn_out).squeeze(-1)  # [1, B]
            else:
                raise ValueError(f"unknown scorer variant: {self.config.variant}")

            scores = {
                action: float(logits[0, i].item())
                for i, action in enumerate(legal_actions)
            }
            return LegalActionScores(
                scores=scores,
                logits=logits,
                legal_actions=tuple(legal_actions),
                metadata={
                    "variant": self.config.variant,
                    "forced": False,
                    "n_legal": len(legal_actions),
                },
            )

        def decode(
            self,
            scores: LegalActionScores,
            legal_actions: list[str],
        ) -> ScorerDecision:
            """Choose the highest-scoring legal action."""
            forced = self._check_forced(legal_actions)
            if forced is not None:
                return forced

            logits = scores.logits[0]
            probs = F.softmax(logits, dim=-1)
            best = int(probs.argmax(dim=-1).item())
            return ScorerDecision(
                action_identity=legal_actions[best],
                decision_kind="scored",
                confidence=float(probs[best].item()),
                telemetry={"variant": self.config.variant, "n_legal": len(legal_actions)},
            )

        def loss(
            self,
            context_features: dict[str, Any],
            state_features: dict[str, Any],
            legal_actions: list[str],
            accepted_action_ids: list[str],
            *,
            plan_features: dict[str, Any] | None = None,
            plan_action_features: list[PlanActionFeatures] | None = None,
            pack_id: str = "openui",
        ) -> tuple[torch.Tensor, dict[str, float]]:
            """Legal-set cross-entropy with uniform target over accepted actions."""
            scores = self.score(
                context_features,
                state_features,
                legal_actions,
                plan_features=plan_features,
                plan_action_features=plan_action_features,
                pack_id=pack_id,
            )
            logits = scores.logits[0]
            target = torch.zeros(len(legal_actions), device=self._device)
            accepted_indices = [
                i for i, a in enumerate(legal_actions) if a in accepted_action_ids
            ]
            if accepted_indices:
                target[accepted_indices] = 1.0 / len(accepted_indices)
            else:
                target[:] = 1.0 / len(legal_actions)
            loss = -(target * F.log_softmax(logits, dim=-1)).sum()
            return loss, {"n_legal": len(legal_actions), "n_accepted": len(accepted_indices)}

        def artifact_identity(self) -> dict[str, Any]:
            return {
                "schema": self.SCHEMA,
                "scorer_id": self.config.scorer_id,
                "variant": self.config.variant,
                "config": self.config.__dict__,
                "param_count": sum(p.numel() for p in self.parameters()),
            }

        def compatibility_fingerprint(self) -> str:
            return content_sha(self.artifact_identity())

        def save(self, path: str) -> None:
            payload = {
                "schema": self.SCHEMA,
                "config": self.config.__dict__,
                "state_dict": self.state_dict(),
                "artifact_identity": self.artifact_identity(),
                "compatibility_fingerprint": self.compatibility_fingerprint(),
            }
            torch.save(payload, path)

        def load(self, path: str) -> None:
            payload = torch.load(path, map_location=self._device, weights_only=False)
            if payload.get("schema") != self.SCHEMA:
                raise ValueError(
                    f"checkpoint schema mismatch: expected {self.SCHEMA!r}, "
                    f"got {payload.get('schema')!r}"
                )
            loaded_config = LegalActionScorerConfig(**payload["config"])
            if loaded_config != self.config:
                raise ValueError("checkpoint config does not match current scorer config")
            self.load_state_dict(payload["state_dict"])

        @classmethod
        def from_checkpoint(
            cls, path: str, device: str = "cpu"
        ) -> "LegalActionScorer":
            payload = torch.load(path, map_location=device, weights_only=False)
            if payload.get("schema") != cls.SCHEMA:
                raise ValueError(
                    f"checkpoint schema mismatch: expected {cls.SCHEMA!r}, "
                    f"got {payload.get('schema')!r}"
                )
            config = LegalActionScorerConfig(**payload["config"])
            instance = cls(config, device=device)
            instance.load_state_dict(payload["state_dict"])
            return instance

else:  # pragma: no cover

    class LegalActionScorer:  # type: ignore[no-redef]
        """Torch-free stub."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "LegalActionScorer requires torch; install torch to use the scorer"
            )


@dataclass(frozen=True)
class FixtureDecision:
    """One synthetic decision for the fixture trainer."""

    decision_id: str
    context_features: dict[str, Any]
    state_features: dict[str, Any]
    legal_actions: list[str]
    accepted_action_ids: list[str]
    plan_features: dict[str, Any] | None = None
    plan_action_features: list[PlanActionFeatures] | None = None
    pack_id: str = "openui"


def make_fixture_decisions(
    n: int = 64,
    *,
    seed: int = 0,
    min_actions: int = 2,
    max_actions: int = 8,
) -> list[FixtureDecision]:
    """Generate deterministic synthetic compiler-legal decision states."""
    rng = random.Random(seed)
    families = ("component", "bind", "literal", "ref")
    templates = ("card", "text", "button", "stack")
    decisions: list[FixtureDecision] = []

    for i in range(n):
        n_actions = rng.randint(min_actions, max_actions)
        actions = []
        for j in range(n_actions):
            family = families[j % len(families)]
            template = templates[(j + i) % len(templates)]
            slot = f"slot{j % 3}"
            actions.append(f"{family}:{slot}:{template}")

        accepted = [actions[rng.randrange(n_actions)]]
        if rng.random() < 0.15 and n_actions > 2:
            second = actions[rng.randrange(n_actions)]
            if second not in accepted:
                accepted.append(second)

        plan_action_features = [
            PlanActionFeatures(
                action_id=a,
                matches_predicted_role=a == accepted[0],
                component_family_compatible=a.startswith("component"),
                expected_coverage_contribution=1.0 / n_actions if a in accepted else 0.0,
                topology_parent_order_compatible=rng.random() > 0.3,
                binding_pointer_compatible=a.startswith("bind"),
                plan_confidence=0.8 if a in accepted else 0.2,
                conflict_or_unknown=a not in accepted,
            )
            for a in actions
        ]

        decisions.append(
            FixtureDecision(
                decision_id=f"decision-{i}",
                context_features={
                    "pack_id": "openui",
                    "n_mentioned_components": rng.randint(1, 5),
                    "position": i,
                },
                state_features={
                    "state_family_id": "fixture",
                    "depth": rng.randint(0, 3),
                    "branch_count": n_actions,
                },
                legal_actions=actions,
                accepted_action_ids=accepted,
                plan_features={
                    "plan_steps": rng.randint(1, 4),
                    "coverage_ratio": rng.random(),
                },
                plan_action_features=plan_action_features,
                pack_id="openui",
            )
        )
    return decisions


def train_fixture_scorer(
    decisions: list[FixtureDecision],
    config: LegalActionScorerConfig | None = None,
    *,
    steps: int = 40,
    lr: float = 0.05,
    device: str = "cpu",
) -> dict[str, Any]:
    """Tiny fixture trainer over synthetic legal-action decisions."""
    _require_torch()
    scorer = LegalActionScorer(config=config, device=device)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)
    history: list[dict[str, float]] = []

    for step in range(steps):
        total_loss = torch.tensor(0.0, device=device)
        metrics_sum: dict[str, float] = {"n_legal": 0.0, "n_accepted": 0.0}
        for decision in decisions:
            loss, metrics = scorer.loss(
                decision.context_features,
                decision.state_features,
                decision.legal_actions,
                decision.accepted_action_ids,
                plan_features=decision.plan_features,
                plan_action_features=decision.plan_action_features,
                pack_id=decision.pack_id,
            )
            total_loss = total_loss + loss
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history.append(
            {
                "step": step + 1,
                "loss": float(total_loss.detach()),
                "n_decisions": len(decisions),
                **{k: v / max(1, len(decisions)) for k, v in metrics_sum.items()},
            }
        )

    return {
        "scorer": scorer,
        "steps": steps,
        "lr": lr,
        "n_decisions": len(decisions),
        "history": history,
        "final_loss": history[-1]["loss"] if history else float("nan"),
    }


def evaluate_fixture_scorer(
    scorer: "LegalActionScorer",
    decisions: Sequence[FixtureDecision],
) -> dict[str, Any]:
    """Teacher-forced top-1 accuracy and forced-decision accounting."""
    correct = 0
    total_scored = 0
    forced = 0
    abstained = 0
    for decision in decisions:
        scores = scorer.score(
            decision.context_features,
            decision.state_features,
            decision.legal_actions,
            plan_features=decision.plan_features,
            plan_action_features=decision.plan_action_features,
            pack_id=decision.pack_id,
        )
        if scores.metadata.get("forced"):
            forced += 1
            continue
        if scores.metadata.get("abstained"):
            abstained += 1
            continue
        total_scored += 1
        chosen = scorer.decode(scores, decision.legal_actions).action_identity
        if chosen in decision.accepted_action_ids:
            correct += 1

    accuracy = correct / total_scored if total_scored else float("nan")
    return {
        "n": len(decisions),
        "correct": correct,
        "total_scored": total_scored,
        "forced": forced,
        "abstained": abstained,
        "accuracy": accuracy,
    }
