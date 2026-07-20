"""SLM-150 (SPV2-02): global semantic energy/value critic baseline.

Wiring/fixture harness only. The critic is a tiny shared MLP that emits a scalar
energy (lower is better), a value (= -energy), per-factor energy heads, and an
abstention confidence. It consumes only inference-available features and is
never allowed to override final verifier membership or suppress ``UNKNOWN``.

Real generalization requires the SPV2-01 hard-valid contrast corpus; until that
corpus exists, ``make_fixture_examples`` supplies deterministic synthetic
contrast groups.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - torch may be absent in minimal environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from slm_training.lineage.records import content_sha

__all__ = [
    "CRITIC_SCHEMA_VERSION",
    "CriticExample",
    "GlobalSemanticCritic",
    "GlobalSemanticCriticConfig",
    "SemanticEnergyOutput",
    "coverage_contract_heuristic",
    "global_critic_factor_loss",
    "global_critic_listwise_loss",
    "global_critic_pairwise_loss",
    "make_fixture_examples",
    "rerank_candidates",
]

CRITIC_SCHEMA_VERSION = "global_semantic_critic/v1"

# Keys and prefixes that are never inference-available for the global critic.
# They look like gold/evaluator leakage and must be ignored by ``featurize``.
_FORBIDDEN_KEYS = frozenset(
    {"accepted_output", "verdict", "semantic_score", "label", "factor_targets"}
)
_FORBIDDEN_PREFIXES = ("gold_", "target_", "judge_")


def _is_safe_key(key: str) -> bool:
    """Return True when ``key`` may enter the inference feature vector."""
    if key in _FORBIDDEN_KEYS:
        return False
    return not any(key.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES)


def _extract_safe_values(obj: Any, path: tuple[str, ...] = ()) -> list[float]:
    """Recursively extract finite numeric scalars from inference-safe keys.

    Keys are processed in sorted order so the resulting vector is deterministic.
    Strings, ``None``, and values under forbidden keys are ignored.
    """
    values: list[float] = []
    if isinstance(obj, Mapping):
        for key in sorted(obj.keys()):
            if not _is_safe_key(str(key)):
                continue
            values.extend(_extract_safe_values(obj[key], (*path, str(key))))
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            values.extend(_extract_safe_values(item, (*path, str(index))))
    elif isinstance(obj, bool):
        values.append(1.0 if obj else 0.0)
    elif isinstance(obj, (int, float)):
        f = float(obj)
        if math.isfinite(f):
            values.append(f)
    return values


@dataclass(frozen=True)
class SemanticEnergyOutput:
    """Energy/value decision for one candidate or group."""

    energy: float
    value: float
    factor_energies: dict[str, float]
    confidence: float
    abstained: bool
    reason_code: str
    input_fingerprint: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalSemanticCriticConfig:
    """Small defaults for the fixture wiring critic."""

    d_model: int = 64
    hidden_dim: int = 64
    num_factors: int = 5
    dropout: float = 0.0
    seed: int = 0
    scorer_id: str = "global-semantic-critic-v1"
    confidence_threshold: float = 0.5
    supported_packs: tuple[str, ...] = ("openui",)


@dataclass
class CriticExample:
    """One contrast example for the global semantic critic."""

    prompt_context: dict
    semantic_plan: dict
    canonical_program_ast: dict
    contract_features: dict
    label: int  # 1 positive, 0 negative, -1 UNKNOWN
    factor_targets: dict[str, float]
    group_id: str
    source: str
    family: str
    severity: str


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

if nn is not None:

    class GlobalSemanticCritic(nn.Module):
        """Tiny MLP critic: lower energy is better; value = -energy."""

        FACTOR_NAMES = ("coverage", "roles", "topology", "bindings", "contract")

        def __init__(
            self,
            config: GlobalSemanticCriticConfig | None = None,
            device: str = "cpu",
        ) -> None:
            super().__init__()
            self.config = config or GlobalSemanticCriticConfig()
            torch.manual_seed(self.config.seed)
            self._device = torch.device(device)

            self.encoder = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
            self.energy_head = nn.Linear(self.config.hidden_dim, 1)
            self.confidence_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.factor_heads = nn.ModuleDict(
                {name: nn.Linear(self.config.hidden_dim, 1) for name in self.FACTOR_NAMES}
            )
            self.to(self._device)

        @property
        def scorer_id(self) -> str:
            return self.config.scorer_id

        def featurize(
            self,
            prompt_context: Mapping[str, Any],
            semantic_plan: Mapping[str, Any],
            canonical_program_ast: Mapping[str, Any],
            contract_features: Mapping[str, Any],
        ) -> torch.Tensor:
            """Build a deterministic numeric feature vector from safe fields only.

            Allowed values are finite numeric scalars reachable through keys that
            are not in the forbidden set and do not start with a forbidden prefix:

            - ``prompt_context``: e.g. ``n_mentioned_components``,
              ``component_type_bits``
            - ``semantic_plan``: e.g. ``plan_steps``, ``coverage_ratio``
            - ``canonical_program_ast``: e.g. ``component_count``, ``depth``,
              ``binding_count``, ``role_count``
            - ``contract_features``: e.g. ``required_component_count``,
              ``required_roles``

            Ignored keys include: ``gold_*``, ``target_*``, ``judge_*``,
            ``accepted_output``, ``verdict``, ``semantic_score``, ``label``,
            ``factor_targets``, plus any non-numeric or non-finite values.
            """
            values: list[float] = []
            for namespace, obj in (
                ("prompt_context", prompt_context),
                ("semantic_plan", semantic_plan),
                ("canonical_program_ast", canonical_program_ast),
                ("contract_features", contract_features),
            ):
                values.extend(_extract_safe_values(obj, (namespace,)))

            target = self.config.d_model
            if len(values) >= target:
                values = values[:target]
            else:
                values = values + [0.0] * (target - len(values))

            return torch.tensor(values, dtype=torch.float32)

        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            """Batch forward pass returning energy, value, factors, confidence."""
            x = features.to(self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            h = self.encoder(x)
            energy = self.energy_head(h).squeeze(-1)
            confidence = self.confidence_head(h).squeeze(-1)
            factor_energies = {
                name: head(h).squeeze(-1) for name, head in self.factor_heads.items()
            }
            return {
                "energy": energy,
                "value": -energy,
                "factor_energies": factor_energies,
                "confidence": confidence,
            }

        def score(
            self,
            prompt_context: Mapping[str, Any],
            semantic_plan: Mapping[str, Any],
            canonical_program_ast: Mapping[str, Any],
            contract_features: Mapping[str, Any],
        ) -> SemanticEnergyOutput:
            """Single-example path; abstains on unsupported packs or low confidence."""
            pack_id = (
                prompt_context.get("pack_id", "openui")
                if isinstance(prompt_context, Mapping)
                else "openui"
            )
            fingerprint = content_sha(
                {
                    "prompt_context": dict(prompt_context),
                    "semantic_plan": dict(semantic_plan),
                    "canonical_program_ast": dict(canonical_program_ast),
                    "contract_features": dict(contract_features),
                }
            )
            provenance: dict[str, Any] = {"pack_id": pack_id}

            if not self.supported_pack(pack_id):
                return SemanticEnergyOutput(
                    energy=0.0,
                    value=0.0,
                    factor_energies={name: 0.0 for name in self.FACTOR_NAMES},
                    confidence=0.0,
                    abstained=True,
                    reason_code="unsupported_pack",
                    input_fingerprint=fingerprint,
                    provenance=provenance,
                )

            features = self.featurize(
                prompt_context,
                semantic_plan,
                canonical_program_ast,
                contract_features,
            ).to(self._device)

            self.eval()
            with torch.no_grad():
                outputs = self.forward(features)

            energy = float(outputs["energy"].item())
            value = float(outputs["value"].item())
            confidence = float(outputs["confidence"].item())
            factor_energies = {
                name: float(tensor.item())
                for name, tensor in outputs["factor_energies"].items()
            }

            if confidence < self.config.confidence_threshold:
                return SemanticEnergyOutput(
                    energy=energy,
                    value=value,
                    factor_energies=factor_energies,
                    confidence=confidence,
                    abstained=True,
                    reason_code="low_confidence",
                    input_fingerprint=fingerprint,
                    provenance=provenance,
                )

            return SemanticEnergyOutput(
                energy=energy,
                value=value,
                factor_energies=factor_energies,
                confidence=confidence,
                abstained=False,
                reason_code="scored",
                input_fingerprint=fingerprint,
                provenance=provenance,
            )

        def supported_pack(self, pack_id: str) -> bool:
            return pack_id in self.config.supported_packs

        def artifact_identity(self) -> dict[str, Any]:
            return {
                "schema": CRITIC_SCHEMA_VERSION,
                "scorer_id": self.config.scorer_id,
                "config": self.config.__dict__,
                "factor_names": list(self.FACTOR_NAMES),
                "param_count": sum(p.numel() for p in self.parameters()),
            }

        def compatibility_fingerprint(self) -> str:
            return content_sha(self.artifact_identity())

        def save(self, path: str) -> None:
            payload = {
                "schema": CRITIC_SCHEMA_VERSION,
                "config": self.config.__dict__,
                "state_dict": self.state_dict(),
                "artifact_identity": self.artifact_identity(),
                "compatibility_fingerprint": self.compatibility_fingerprint(),
            }
            torch.save(payload, path)

        def load(self, path: str) -> None:
            payload = torch.load(path, map_location=self._device, weights_only=False)
            if payload.get("schema") != CRITIC_SCHEMA_VERSION:
                raise ValueError(
                    f"checkpoint schema mismatch: expected {CRITIC_SCHEMA_VERSION!r}, "
                    f"got {payload.get('schema')!r}"
                )
            loaded_config = GlobalSemanticCriticConfig(**payload["config"])
            if loaded_config != self.config:
                raise ValueError(
                    "checkpoint config does not match current critic config"
                )
            self.load_state_dict(payload["state_dict"])

        @classmethod
        def from_checkpoint(cls, path: str, device: str = "cpu") -> "GlobalSemanticCritic":
            payload = torch.load(path, map_location=device, weights_only=False)
            if payload.get("schema") != CRITIC_SCHEMA_VERSION:
                raise ValueError(
                    f"checkpoint schema mismatch: expected {CRITIC_SCHEMA_VERSION!r}, "
                    f"got {payload.get('schema')!r}"
                )
            config = GlobalSemanticCriticConfig(**payload["config"])
            instance = cls(config, device=device)
            instance.load_state_dict(payload["state_dict"])
            return instance

        @classmethod
        def from_records(
            cls,
            records: Sequence[Any],
            config: GlobalSemanticCriticConfig | None = None,
            device: str = "cpu",
        ) -> "GlobalSemanticCritic":
            """Fixture initializer; ``records`` are ignored for the wiring baseline."""
            return cls(config=config, device=device)

else:  # pragma: no cover

    class GlobalSemanticCritic:  # type: ignore[no-redef]
        """Torch-free stub."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "GlobalSemanticCritic requires torch; install torch to use the critic"
            )


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

if torch is not None:

    def global_critic_pairwise_loss(
        energies: torch.Tensor,
        labels: torch.Tensor,
        group_ids: list[str],
        margin: float = 0.1,
    ) -> torch.Tensor:
        """Within each group, positives should have lower energy than negatives.

        ``UNKNOWN`` (-1) labels and energy ties are skipped.
        """
        known = [i for i, label in enumerate(labels) if int(label) in (0, 1)]
        groups: dict[str, list[int]] = {}
        for i in known:
            groups.setdefault(group_ids[i], []).append(i)

        terms: list[torch.Tensor] = []
        eps = 1e-8
        for members in groups.values():
            pos = [i for i in members if int(labels[i]) == 1]
            neg = [i for i in members if int(labels[i]) == 0]
            for p in pos:
                for n in neg:
                    if abs(float(energies[p]) - float(energies[n])) < eps:
                        continue
                    terms.append(F.relu(energies[p] - energies[n] + margin))

        if not terms:
            return energies.new_zeros(())
        return torch.stack(terms).mean()

    def global_critic_listwise_loss(
        energies: torch.Tensor,
        labels: torch.Tensor,
        group_ids: list[str],
    ) -> torch.Tensor:
        """Per-group softmax over ``-energies`` with uniform target over positives.

        Groups with no positive label are skipped. ``UNKNOWN`` (-1) rows are
        included in the normalization denominator but receive zero target mass.
        """
        known = [i for i, label in enumerate(labels) if int(label) in (0, 1)]
        groups: dict[str, list[int]] = {}
        for i in known:
            groups.setdefault(group_ids[i], []).append(i)

        terms: list[torch.Tensor] = []
        for members in groups.values():
            pos = [i for i in members if int(labels[i]) == 1]
            if not pos:
                continue
            e = energies[members]
            log_probs = F.log_softmax(-e, dim=0)
            target = torch.zeros_like(e)
            pos_indices = torch.tensor(
                [members.index(p) for p in pos], dtype=torch.long
            )
            target[pos_indices] = 1.0 / len(pos)
            terms.append(-(target * log_probs).sum())

        if not terms:
            return energies.new_zeros(())
        return torch.stack(terms).mean()

    def global_critic_factor_loss(
        pred_factors: Mapping[str, torch.Tensor],
        target_factors: Mapping[str, float],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked MSE over factor head predictions."""
        mask = mask.bool()
        if int(mask.sum()) == 0:
            return next(iter(pred_factors.values())).new_zeros(())

        terms: list[torch.Tensor] = []
        for name, pred in pred_factors.items():
            target = float(target_factors.get(name, 0.0))
            p = pred.squeeze()[mask]
            t = torch.full_like(p, target)
            terms.append(F.mse_loss(p, t))

        if not terms:
            return torch.tensor(0.0)
        return torch.stack(terms).mean()

else:  # pragma: no cover

    def global_critic_pairwise_loss(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("global_critic_pairwise_loss requires torch")

    def global_critic_listwise_loss(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("global_critic_listwise_loss requires torch")

    def global_critic_factor_loss(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("global_critic_factor_loss requires torch")


# --------------------------------------------------------------------------- #
# Deterministic heuristic
# --------------------------------------------------------------------------- #


def coverage_contract_heuristic(
    program_features: Mapping[str, Any],
    contract_features: Mapping[str, Any],
) -> float:
    """Estimate component coverage gap; lower is better.

    Returns the normalized gap between required contract components and the
    components present in the program. Over-coverage is not penalized.
    """
    present = float(program_features.get("component_count", 0))
    required = float(contract_features.get("required_component_count", 0))
    if required <= 0:
        return 0.0
    gap = max(0.0, required - present)
    return gap / required


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #


def rerank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    critic: GlobalSemanticCritic,
    prompt_context: Mapping[str, Any],
    semantic_plan: Mapping[str, Any],
    contract_features: Mapping[str, Any],
    *,
    local_scores: Mapping[str, float] | None = None,
    lambda_global: float = 1.0,
) -> tuple[list[str], SemanticEnergyOutput | None, dict[str, Any]]:
    """Return a permutation of candidate ids ordered by ``local - lambda * energy``.

    ``candidates`` is a sequence of mappings, each containing at least a
    ``candidate_id`` key plus inference-safe program AST features.

    If every candidate abstains, the returned list is empty, the output carries
    ``abstained=True`` and ``reason_code="all_abstained"``, and the trace
    records per-candidate critic outputs.
    """
    local_scores = local_scores or {}
    outputs: dict[str, SemanticEnergyOutput] = {}
    scored: list[tuple[float, str, SemanticEnergyOutput]] = []

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        local = float(local_scores.get(candidate_id, 0.0))
        output = critic.score(
            prompt_context,
            semantic_plan,
            candidate,
            contract_features,
        )
        outputs[candidate_id] = output
        if not output.abstained:
            combined = local - lambda_global * output.energy
            scored.append((combined, candidate_id, output))

    trace: dict[str, Any] = {
        "outputs": outputs,
        "scored_count": len(scored),
        "lambda_global": lambda_global,
    }

    if not scored:
        dummy = SemanticEnergyOutput(
            energy=0.0,
            value=0.0,
            factor_energies={name: 0.0 for name in critic.FACTOR_NAMES},
            confidence=0.0,
            abstained=True,
            reason_code="all_abstained",
            input_fingerprint="",
            provenance={},
        )
        return [], dummy, trace

    scored.sort(key=lambda item: (-item[0], item[1]))
    order = [candidate_id for _, candidate_id, _ in scored]
    best_output = scored[0][2]
    trace["combined_scores"] = {
        candidate_id: combined for combined, candidate_id, _ in scored
    }
    return order, best_output, trace


# --------------------------------------------------------------------------- #
# Fixture data builder
# --------------------------------------------------------------------------- #


def make_fixture_examples(
    n_groups: int = 16,
    candidates_per_group: int = 4,
    seed: int = 0,
) -> list[CriticExample]:
    """Deterministic synthetic contrast examples using only inference-safe fields."""
    rng = random.Random(seed)
    families = ("hero", "cta", "navbar", "footer", "card")
    severities = ("low", "medium", "high")
    factor_names = ("coverage", "roles", "topology", "bindings", "contract")

    examples: list[CriticExample] = []
    for group_index in range(n_groups):
        for candidate_index in range(candidates_per_group):
            if group_index % 2 == 0 and candidate_index < 2:
                label = 1
            elif candidate_index == candidates_per_group - 1:
                label = -1
            else:
                label = 0

            program_features = {
                "candidate_id": f"g{group_index}_c{candidate_index}",
                "component_count": rng.randint(1, 6),
                "depth": rng.randint(1, 4),
                "binding_count": rng.randint(0, 5),
                "role_count": rng.randint(1, 5),
            }

            examples.append(
                CriticExample(
                    prompt_context={
                        "pack_id": "openui",
                        "n_mentioned_components": rng.randint(1, 5),
                    },
                    semantic_plan={
                        "plan_steps": rng.randint(1, 4),
                        "coverage_ratio": rng.random(),
                    },
                    canonical_program_ast=program_features,
                    contract_features={
                        "required_component_count": rng.randint(2, 6),
                    },
                    label=label,
                    factor_targets={name: rng.random() for name in factor_names},
                    group_id=f"fixture_group_{group_index}",
                    source="fixture",
                    family=families[group_index % len(families)],
                    severity=severities[group_index % len(severities)],
                )
            )
    return examples
