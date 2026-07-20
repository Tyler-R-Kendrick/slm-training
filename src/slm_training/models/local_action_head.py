"""State-local action heads for CAP2-03.

Implements a common interface for scoring only the legal actions supplied for an
exact compiler state.  Head families include a global masked control, a local
flat head, a ternary digit head, a ternary distance-2 ECOC head, and a
grammar-factorized head.

Forced states with a single legal action bypass the learned head and return the
compiler action directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
import zlib

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.models.action_code_registry import ActionCodeEntry, ActionSchema
from slm_training.models.quantization.residual_planes import ResidualTritStack
from slm_training.models.semantic_cost import (
    CostMatrix,
    build_ternary_ecoc_entry,
    trit_distance,
)


@dataclass
class StateContext:
    """Context supplied by the compiler for a state-local decision."""

    state_family_id: str
    state_signature: tuple[Any, ...] = ()
    branch_count: int = 0
    forced: bool = False
    sensitivity: Mapping[str, float] | None = None
    completion_support_size: int | None = None


@dataclass
class LocalActionOutput:
    """Output of a LocalActionHead.score call.

    Attributes:
        logits: per-legal-action logits [batch, num_legal].  Present for flat
            and global-masked heads.
        trits: predicted trit logits or hard trits [batch, m, 3] or [batch, m].
            Present for ternary digit/ECOC heads.
        factor_logits: dict of per-factor logits for grammar-factorized heads.
        head_family: identifier for telemetry.
        metadata: extra head-specific data.
    """

    logits: torch.Tensor | None = None
    trits: torch.Tensor | None = None
    factor_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    head_family: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecision:
    """Decoded action decision.

    Attributes:
        action_identity: chosen action identity, or None for forced/abstain.
        decision_kind: "scored", "forced", "abstain", "refine", "detected_error".
        confidence: max softmax probability for scored decisions.
        codeword: predicted codeword for ternary heads, if applicable.
        telemetry: head-specific telemetry.
    """

    action_identity: str | None
    decision_kind: str
    confidence: float = 0.0
    codeword: tuple[int, ...] | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)


class LocalActionHead(nn.Module):
    """Shared interface for state-local action heads."""

    head_family: str = ""

    def score(
        self,
        hidden: torch.Tensor,
        state_context: StateContext,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        raise NotImplementedError

    def decode(
        self,
        output: LocalActionOutput,
        legal_actions: list[str],
    ) -> ActionDecision:
        raise NotImplementedError


def _check_forced(legal_actions: list[str]) -> ActionDecision | None:
    """If only one legal action exists, return a forced decision."""
    if len(legal_actions) == 1:
        return ActionDecision(
            action_identity=legal_actions[0],
            decision_kind="forced",
            confidence=1.0,
            telemetry={"skip_reason": "single_legal_action"},
        )
    return None


def _stable_action_index(action: str, vocab_size: int) -> int:
    """Deterministic action -> output index mapping (stable across processes)."""
    return (zlib.crc32(action.encode("utf-8")) & 0xFFFFFFFF) % vocab_size


class GlobalMaskedHead(LocalActionHead):
    """Global output-class logits followed by an exact legal mask."""

    head_family = "global_masked"

    def __init__(self, hidden_dim: int, max_vocabulary: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vocabulary = max_vocabulary
        self.global_logits = nn.Linear(hidden_dim, max_vocabulary)

    def score(
        self,
        hidden: torch.Tensor,
        state_context: StateContext,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        # Map legal action identities to indices in [0, max_vocabulary).
        legal_indices = torch.tensor(
            [_stable_action_index(a, self.max_vocabulary) for a in legal_actions],
            dtype=torch.long,
            device=hidden.device,
        )
        all_logits = self.global_logits(hidden)
        # Gather and mask: scores for illegal actions are -inf.
        masked = torch.full_like(all_logits, float("-inf"))
        masked[:, legal_indices] = all_logits[:, legal_indices]
        return LocalActionOutput(
            logits=masked,
            head_family=self.head_family,
            metadata={"legal_indices": legal_indices},
        )

    def decode(
        self,
        output: LocalActionOutput,
        legal_actions: list[str],
    ) -> ActionDecision:
        forced = _check_forced(legal_actions)
        if forced is not None:
            return forced
        assert output.logits is not None
        legal_indices = torch.tensor(
            [_stable_action_index(a, self.max_vocabulary) for a in legal_actions],
            dtype=torch.long,
            device=output.logits.device,
        )
        legal_scores = output.logits[:, legal_indices]
        probs = F.softmax(legal_scores, dim=-1)
        best = int(probs.argmax(dim=-1).item())
        return ActionDecision(
            action_identity=legal_actions[best],
            decision_kind="scored",
            confidence=float(probs[0, best].item()),
            telemetry={"head_family": self.head_family},
        )


class LocalFlatHead(LocalActionHead):
    """Directly score the b(q) legal semantic actions."""

    head_family = "local_flat"

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scorer = nn.Linear(hidden_dim, 1)

    def score(
        self,
        hidden: torch.Tensor,
        state_context: StateContext,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        # hidden [batch, hidden_dim]; produce one score per legal action.
        # We tile hidden and score each action with a learned embedding.
        if not hasattr(self, "action_embeddings"):
            # Lazy per-action embeddings keyed by hash.  In a real model these
            # would be a shared learned table indexed by action identity.
            self.action_embeddings: dict[str, nn.Parameter] = {}
        embeddings: list[torch.Tensor] = []
        for action in legal_actions:
            if action not in self.action_embeddings:
                self.action_embeddings[action] = nn.Parameter(
                    torch.randn(self.hidden_dim) * 0.02
                )
            embeddings.append(self.action_embeddings[action])
        stacked = torch.stack(embeddings, dim=0)  # [b, hidden_dim]
        # [batch, b] = hidden @ stacked.T + scorer(hidden) broadcast
        scores = hidden @ stacked.T  # [batch, b]
        return LocalActionOutput(
            logits=scores,
            head_family=self.head_family,
            metadata={"action_count": len(legal_actions)},
        )

    def decode(
        self,
        output: LocalActionOutput,
        legal_actions: list[str],
    ) -> ActionDecision:
        forced = _check_forced(legal_actions)
        if forced is not None:
            return forced
        assert output.logits is not None
        probs = F.softmax(output.logits, dim=-1)
        best = int(probs.argmax(dim=-1).item())
        return ActionDecision(
            action_identity=legal_actions[best],
            decision_kind="scored",
            confidence=float(probs[0, best].item()),
            telemetry={"head_family": self.head_family},
        )


class TernaryDigitHead(LocalActionHead):
    """Encode local action IDs with m=ceil(log_3 b) trits."""

    head_family = "ternary_digit"

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_trits = 6
        self.trit_logits = nn.Linear(hidden_dim, self.max_trits * 3)

    def _m(self, b: int) -> int:
        import math

        if b <= 1:
            return 0
        return math.ceil(math.log(b, 3))

    def score(
        self,
        hidden: torch.Tensor,
        state_context: StateContext,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        b = len(legal_actions)
        m = self._m(b)
        flat = self.trit_logits(hidden)
        batch = hidden.shape[0]
        if m == 0:
            trits = torch.zeros(batch, 0, 3, device=hidden.device)
        else:
            trits = flat[:, : m * 3].view(batch, m, 3)
        return LocalActionOutput(
            trits=trits,
            head_family=self.head_family,
            metadata={"action_count": b, "m": m},
        )

    def decode(
        self,
        output: LocalActionOutput,
        legal_actions: list[str],
    ) -> ActionDecision:
        forced = _check_forced(legal_actions)
        if forced is not None:
            return forced
        assert output.trits is not None
        hard_trits = output.trits.argmax(dim=-1)  # [batch, m]
        m = hard_trits.shape[-1]
        index = torch.zeros(hard_trits.shape[0], dtype=torch.long, device=hard_trits.device)
        for i in range(m):
            index = index * 3 + hard_trits[:, i]
        idx = int(index.item())
        if idx < len(legal_actions):
            return ActionDecision(
                action_identity=legal_actions[idx],
                decision_kind="scored",
                telemetry={"head_family": self.head_family, "codeword": tuple(hard_trits[0].tolist())},
                codeword=tuple(hard_trits[0].tolist()),
            )
        return ActionDecision(
            action_identity=None,
            decision_kind="abstain",
            telemetry={"head_family": self.head_family, "reason": "invalid_trit_index"},
        )


class TernaryECOCHead(LocalActionHead):
    """Ternary distance-2 ECOC head with semantic-cost-aware codewords."""

    head_family = "ternary_ecoc"

    def __init__(
        self,
        hidden_dim: int,
        registry: Any | None = None,
        costs: CostMatrix | None = None,
        use_detection: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.registry = registry
        self.costs = costs
        self.use_detection = use_detection
        self.max_trits = 8
        self.trit_logits = nn.Linear(hidden_dim, self.max_trits * 3)

    def _get_entry(self, legal_actions: list[str]) -> ActionCodeEntry:
        if self.registry is None:
            raise ValueError("TernaryECOCHead requires a registry")
        schema = ActionSchema(
            state_family_id=f"ecoc_{len(legal_actions)}",
            action_identities=tuple(sorted(legal_actions)),
        )
        entry = self.registry.get(schema)
        if entry is None:
            entry = build_ternary_ecoc_entry(
                schema,
                self.costs,
                detect_single_trit_error=self.use_detection,
                cost_matrix_source="provided" if self.costs is not None else "uniform",
            )
            self.registry.register(entry)
        return entry

    def score(
        self,
        hidden: torch.Tensor,
        state_context: StateContext,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        entry = self._get_entry(legal_actions)
        m = len(entry.alphabet_radices)
        flat = self.trit_logits(hidden)
        batch = hidden.shape[0]
        if m == 0:
            trits = torch.zeros(batch, 0, 3, device=hidden.device)
        else:
            trits = flat[:, : m * 3].view(batch, m, 3)
        return LocalActionOutput(
            trits=trits,
            head_family=self.head_family,
            metadata={
                "action_count": len(legal_actions),
                "m": m,
                "entry_hash": entry.entry_hash,
                "min_distance": entry.minimum_hamming_distance,
            },
        )

    def decode(
        self,
        output: LocalActionOutput,
        legal_actions: list[str],
    ) -> ActionDecision:
        forced = _check_forced(legal_actions)
        if forced is not None:
            return forced
        assert output.trits is not None
        entry = self._get_entry(legal_actions)
        hard_trits = output.trits.argmax(dim=-1)  # [batch, m]
        codeword = tuple(int(t) for t in hard_trits[0].tolist())
        action = entry.action_for_codeword(codeword)
        if action is not None:
            return ActionDecision(
                action_identity=action,
                decision_kind="scored",
                codeword=codeword,
                telemetry={
                    "head_family": self.head_family,
                    "entry_hash": entry.entry_hash,
                },
            )
        # Nearest-codeword fallback under Hamming distance.
        best_action: str | None = None
        best_dist = None
        for assignment in entry.assignments:
            d = trit_distance(codeword, assignment.codeword)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_action = assignment.action_identity
        if self.use_detection and best_dist == 1:
            return ActionDecision(
                action_identity=None,
                decision_kind="detected_error",
                codeword=codeword,
                telemetry={
                    "head_family": self.head_family,
                    "nearest_distance": best_dist,
                    "entry_hash": entry.entry_hash,
                },
            )
        policy = entry.invalid_code_policy
        if policy == "abstain":
            return ActionDecision(
                action_identity=None,
                decision_kind="abstain",
                codeword=codeword,
                telemetry={
                    "head_family": self.head_family,
                    "nearest_distance": best_dist,
                    "entry_hash": entry.entry_hash,
                },
            )
        if policy == "refine":
            return ActionDecision(
                action_identity=None,
                decision_kind="refine",
                codeword=codeword,
                telemetry={
                    "head_family": self.head_family,
                    "nearest_distance": best_dist,
                },
            )
        # Fallback: nearest action, but report the policy used.
        return ActionDecision(
            action_identity=best_action,
            decision_kind="scored",
            codeword=codeword,
            telemetry={
                "head_family": self.head_family,
                "fallback": "nearest_codeword",
                "nearest_distance": best_dist,
            },
        )


class GrammarFactorizedHead(LocalActionHead):
    """Grammar-factorized head predicting production family, slot/type, ref class, template.

    This is a simplified factorization sufficient to demonstrate the family.  A
    full OpenUI implementation would mirror the production-codec sigil alphabet.
    """

    head_family = "grammar_factorized"

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.family_logits = nn.Linear(hidden_dim, 4)
        self.slot_type_logits = nn.Linear(hidden_dim, 4)
        self.ref_class_logits = nn.Linear(hidden_dim, 3)
        self.template_logits = nn.Linear(hidden_dim, 4)

    def _parse_action(self, action: str) -> dict[str, int]:
        """Map a simple action identity string to factor indices."""
        parts = action.split(":")
        family = parts[0] if parts else "other"
        slot = parts[1] if len(parts) > 1 else "none"
        ref = parts[2] if len(parts) > 2 else "none"
        template = parts[3] if len(parts) > 3 else "none"
        families = {"component": 0, "bind": 1, "literal": 2, "ref": 3}
        slots = {"root": 0, "arg0": 1, "arg1": 2, "none": 3}
        refs = {"local": 0, "global": 1, "none": 2}
        templates = {"card": 0, "text": 1, "button": 2, "none": 3}
        return {
            "family": families.get(family, 3),
            "slot_type": slots.get(slot, 3),
            "ref_class": refs.get(ref, 2),
            "template": templates.get(template, 3),
        }

    def _reconstruct_action(self, factors: dict[str, int]) -> str:
        families = {0: "component", 1: "bind", 2: "literal", 3: "other"}
        slots = {0: "root", 1: "arg0", 2: "arg1", 3: "none"}
        refs = {0: "local", 1: "global", 2: "none"}
        templates = {0: "card", 1: "text", 2: "button", 3: "none"}
        return ":".join(
            [
                families.get(factors["family"], "other"),
                slots.get(factors["slot_type"], "none"),
                refs.get(factors["ref_class"], "none"),
                templates.get(factors["template"], "none"),
            ]
        )

    def score(
        self,
        hidden: torch.Tensor,
        state_context: StateContext,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        return LocalActionOutput(
            factor_logits={
                "family": self.family_logits(hidden),
                "slot_type": self.slot_type_logits(hidden),
                "ref_class": self.ref_class_logits(hidden),
                "template": self.template_logits(hidden),
            },
            head_family=self.head_family,
            metadata={"action_count": len(legal_actions)},
        )

    def decode(
        self,
        output: LocalActionOutput,
        legal_actions: list[str],
    ) -> ActionDecision:
        forced = _check_forced(legal_actions)
        if forced is not None:
            return forced
        factors = {
            name: int(logits.argmax(dim=-1).item())
            for name, logits in output.factor_logits.items()
        }
        reconstructed = self._reconstruct_action(factors)
        if reconstructed in legal_actions:
            return ActionDecision(
                action_identity=reconstructed,
                decision_kind="scored",
                telemetry={"head_family": self.head_family, "factors": factors},
            )
        # Find the nearest legal action by factor match count.
        best_action = legal_actions[0]
        best_score = -1
        target = self._parse_action(reconstructed)
        for action in legal_actions:
            parsed = self._parse_action(action)
            score = sum(1 for k in target if target[k] == parsed[k])
            if score > best_score:
                best_score = score
                best_action = action
        return ActionDecision(
            action_identity=best_action,
            decision_kind="scored",
            telemetry={
                "head_family": self.head_family,
                "factors": factors,
                "fallback": "nearest_factor_match",
            },
        )


class ResidualTritPlaneHead(LocalActionHead):
    """Local scorer with a base embedding table + ternary residual plane refinement.

    The base score is ``hidden @ E[a]^T`` for each legal action ``a``.  A
    :class:`ResidualTritStack` refines the score vector in one shot.  This is a
    wiring fixture: action hashing can collide, so production use needs a real
    action index registry.
    """

    head_family = "residual_trit_plane"

    def __init__(
        self,
        hidden_dim: int,
        max_actions: int = 512,
        R: int = 2,
        scale_mode: Literal[
            "geometric_balanced",
            "learned_independent",
            "learned_monotone",
        ] = "geometric_balanced",
        residual_normalization: Literal["none", "rms", "variance_preserving"] = "none",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_actions = max_actions
        self.R = R
        self.base_embeddings = nn.Embedding(max_actions, hidden_dim)
        self.residual_stack = ResidualTritStack(
            in_features=hidden_dim,
            out_features=max_actions,
            R=R,
            scale_mode=scale_mode,
            residual_normalization=residual_normalization,
            bias=True,
        )

    def score(
        self,
        hidden: torch.Tensor,
        state_context: StateContext,
        legal_actions: list[str],
        *,
        max_planes: int | None = None,
        return_diagnostics: bool = False,
    ) -> LocalActionOutput:
        indices = torch.tensor(
            [_stable_action_index(a, self.max_actions) for a in legal_actions],
            dtype=torch.long,
            device=hidden.device,
        )
        base_emb = self.base_embeddings(indices)  # [b, hidden_dim]
        base_scores = hidden @ base_emb.T  # [batch, b]
        if return_diagnostics:
            diag = self.residual_stack(
                hidden,
                max_planes=max_planes,
                return_diagnostics=True,
            )
            all_scores = diag.final_output
        else:
            diag = None
            all_scores = self.residual_stack(hidden, max_planes=max_planes)
        residual_scores = all_scores[:, indices]  # [batch, b]
        metadata: dict[str, Any] = {
            "action_count": len(legal_actions),
            "R": self.R,
            "scale_mode": self.residual_stack.scale_mode,
            "residual_normalization": self.residual_stack.residual_normalization,
        }
        if diag is not None:
            metadata["plane_diagnostics"] = diag
        return LocalActionOutput(
            logits=base_scores + residual_scores,
            head_family=self.head_family,
            metadata=metadata,
        )

    def decode(
        self,
        output: LocalActionOutput,
        legal_actions: list[str],
    ) -> ActionDecision:
        forced = _check_forced(legal_actions)
        if forced is not None:
            return forced
        assert output.logits is not None
        probs = F.softmax(output.logits, dim=-1)
        best = int(probs.argmax(dim=-1).item())
        return ActionDecision(
            action_identity=legal_actions[best],
            decision_kind="scored",
            confidence=float(probs[0, best].item()),
            telemetry={"head_family": self.head_family},
        )
