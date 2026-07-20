"""Dynamic contract-index pointer scorer for grammar-action binding.

SLM-168 (SDE2-01): wiring-only default-off module.  The scorer is intentionally
not wired into live ``_choice_ltr_decode_batch``; callers that set
``pointer_mode="legacy_tokens"`` (the default) get no-op behavior.  This module
provides the data contract, scorer variants, and deterministic fixture helpers
needed to evaluate explicit pointer supervision before promotion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PointerCandidate",
    "PointerCandidateSet",
    "DynamicPointerScorerConfig",
    "DynamicPointerScorer",
    "PointerDecision",
    "count_pointer_scorer_parameters",
    "estimate_pointer_scorer_flops",
]

DYNAMIC_POINTER_SCORER_SCHEMA_VERSION = "dynamic_pointer_scorer/v1"
PointerMode = Literal["legacy_tokens", "dynamic_head"]
PointerCandidateSource = Literal[
    "structured_contract", "authored_only", "inventory_in_prompt"
]
PointerKind = Literal["slot", "runtime_symbol", "binder", "state", "schema_entity"]
PointerProvenance = Literal[
    "request_contract", "runtime", "authored_prompt", "compiler_scope"
]


@dataclass(frozen=True)
class PointerCandidate:
    """One request-visible candidate for a contract-index pointer decision."""

    stable_id: str
    display_text: str
    kind: PointerKind
    type_name: str | None
    provenance: PointerProvenance

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointerCandidate":
        return cls(
            stable_id=str(data["stable_id"]),
            display_text=str(data["display_text"]),
            kind=str(data["kind"]),  # type: ignore[arg-type]
            type_name=data.get("type_name"),
            provenance=str(data["provenance"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PointerCandidateSet:
    """Versioned set of pointer candidates built from inference-available fields."""

    candidates: tuple[PointerCandidate, ...]
    permitted_sources: tuple[str, ...]
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "permitted_sources": list(self.permitted_sources),
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointerCandidateSet":
        return cls(
            candidates=tuple(
                PointerCandidate.from_dict(c) for c in data.get("candidates", [])
            ),
            permitted_sources=tuple(data.get("permitted_sources", [])),
            manifest_hash=str(data["manifest_hash"]),
        )

    def __len__(self) -> int:
        return len(self.candidates)

    def index(self, stable_id: str) -> int | None:
        """Return the candidate index for ``stable_id`` or None if absent."""
        for i, cand in enumerate(self.candidates):
            if cand.stable_id == stable_id:
                return i
        return None


@dataclass
class DynamicPointerScorerConfig:
    """Pinned configuration for a dynamic pointer scorer."""

    pointer_mode: PointerMode = "legacy_tokens"
    pointer_candidate_source: PointerCandidateSource = "structured_contract"
    d_model: int = 128
    pointer_hidden_dim: int = 256
    pointer_heads: int = 4
    pointer_temperature: float = 1.0
    dropout: float = 0.0
    schema: str = DYNAMIC_POINTER_SCORER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamicPointerScorerConfig":
        return cls(
            pointer_mode=str(data.get("pointer_mode", "legacy_tokens")),  # type: ignore[arg-type]
            pointer_candidate_source=str(
                data.get("pointer_candidate_source", "structured_contract")
            ),  # type: ignore[arg-type]
            d_model=int(data.get("d_model", 128)),
            pointer_hidden_dim=int(data.get("pointer_hidden_dim", 256)),
            pointer_heads=int(data.get("pointer_heads", 4)),
            pointer_temperature=float(data.get("pointer_temperature", 1.0)),
            dropout=float(data.get("dropout", 0.0)),
            schema=str(
                data.get("schema", DYNAMIC_POINTER_SCORER_SCHEMA_VERSION)
            ),
        )


@dataclass(frozen=True)
class PointerDecision:
    """One pointer decision with provenance and diagnostics."""

    state_signature: str
    candidate_set_hash: str
    gold_index: int | None
    selected_index: int
    scores: tuple[float, ...]
    mask: tuple[bool, ...]
    pointer_mode: str
    candidate_source: str
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointerDecision":
        return cls(
            state_signature=str(data["state_signature"]),
            candidate_set_hash=str(data["candidate_set_hash"]),
            gold_index=data.get("gold_index"),
            selected_index=int(data["selected_index"]),
            scores=tuple(float(s) for s in data["scores"]),
            mask=tuple(bool(m) for m in data["mask"]),
            pointer_mode=str(data["pointer_mode"]),
            candidate_source=str(data["candidate_source"]),
            latency_seconds=float(data["latency_seconds"]),
        )


def _hash_manifest(candidates: tuple[PointerCandidate, ...]) -> str:
    payload = json.dumps(
        [c.to_dict() for c in candidates], sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class DynamicPointerScorer(nn.Module):
    """Default-off dynamic pointer scorer over a live candidate set.

    When ``pointer_mode="legacy_tokens"`` the module is None internally and all
    score methods return uniform zero scores, preserving checkpoint/decode
    behavior.  When ``pointer_mode="dynamic_head"`` a small query/key head
    scores each candidate relative to the current decoder state.
    """

    SCHEMA = DYNAMIC_POINTER_SCORER_SCHEMA_VERSION

    def __init__(
        self,
        config: DynamicPointerScorerConfig | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or DynamicPointerScorerConfig()
        self.device = str(device)
        self._build()

    def _build(self) -> None:
        if self.config.pointer_mode == "legacy_tokens":
            self.scorer: nn.Module | None = None
            return
        if self.config.pointer_mode != "dynamic_head":
            raise ValueError(
                f"unknown pointer_mode: {self.config.pointer_mode!r}"
            )
        d_model = int(self.config.d_model)
        hidden = int(self.config.pointer_hidden_dim)
        self.kind_embeddings = nn.Embedding(len(_KIND_INDEX), d_model)
        self.query_proj = nn.Linear(d_model, hidden)
        self.key_proj = nn.Linear(d_model, hidden)
        self.type_compatibility = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(float(self.config.dropout))
        self.scorer = nn.ModuleDict(
            {
                "kind_embeddings": self.kind_embeddings,
                "query_proj": self.query_proj,
                "key_proj": self.key_proj,
                "type_compatibility": self.type_compatibility,
                "dropout": self.dropout,
            }
        )

    def forward(
        self,
        state_vector: torch.Tensor,
        candidates: PointerCandidateSet,
    ) -> torch.Tensor:
        """Return logits ``(n_candidates,)`` for the live candidate set.

        ``state_vector`` has shape ``(d_model,)``.  Candidate representations are
        built from kind embeddings and a deterministic hash of the display text
        so the fixture is CPU-safe and replayable.
        """
        if self.scorer is None or self.config.pointer_mode == "legacy_tokens":
            return torch.zeros(len(candidates), device=state_vector.device)

        if len(candidates) == 0:
            return torch.zeros(0, device=state_vector.device)

        d_model = int(self.config.d_model)
        # Deterministic candidate vectors from display text.
        cand_vectors = torch.stack(
            [_candidate_vector(c, d_model, state_vector.device) for c in candidates.candidates]
        )
        kind_ids = torch.tensor(
            [_KIND_INDEX.get(c.kind, 0) for c in candidates.candidates],
            dtype=torch.long,
            device=state_vector.device,
        )
        kind_vecs = self.kind_embeddings(kind_ids)
        cand_repr = cand_vectors + kind_vecs

        q = self.query_proj(self.dropout(state_vector.unsqueeze(0))).squeeze(0)
        k = self.key_proj(self.dropout(cand_repr))
        logits = torch.matmul(k, q) / max(1e-6, float(self.config.pointer_temperature))

        # Small type-compatibility bias.
        type_bias = self.type_compatibility(cand_repr).squeeze(-1)
        logits = logits + type_bias
        return logits

    def score(
        self,
        state_vector: torch.Tensor,
        candidates: PointerCandidateSet,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return masked log-probabilities over the candidate set."""
        logits = self.forward(state_vector, candidates)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(dtype=torch.bool, device=logits.device), float("-inf"))
        return F.log_softmax(logits, dim=-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "state_dict": {k: v.cpu().tolist() for k, v in self.state_dict().items()},
            "schema": self.SCHEMA,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "DynamicPointerScorer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        config = DynamicPointerScorerConfig.from_dict(data.get("config", {}))
        scorer = cls(config, device=device)
        if scorer.scorer is not None and "state_dict" in data:
            state = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in data["state_dict"].items()}
            scorer.load_state_dict(state, strict=False)
        return scorer


_KIND_INDEX: dict[PointerKind, int] = {
    "slot": 0,
    "runtime_symbol": 1,
    "binder": 2,
    "state": 3,
    "schema_entity": 4,
}


def _candidate_vector(
    candidate: PointerCandidate,
    d_model: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Deterministic hash-based vector for a candidate (CPU-safe, no downloads)."""
    text = f"{candidate.stable_id}:{candidate.display_text}:{candidate.kind}:{candidate.type_name or ''}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = __import__("random").Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(d_model)]
    vec = torch.tensor(values, dtype=torch.float32, device=device)
    norm = vec.norm(dim=-1, keepdim=True)
    if norm.item() > 0:
        vec = vec / norm
    return vec


def count_pointer_scorer_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in a pointer scorer."""
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def estimate_pointer_scorer_flops(
    module: DynamicPointerScorer,
    n_candidates: int,
) -> int:
    """Return a rough MAC/FLOP estimate for one pointer scoring pass."""
    if module.scorer is None:
        return 0
    d_model = int(module.config.d_model)
    hidden = int(module.config.pointer_hidden_dim)
    # query projection + key projection per candidate + dot products.
    flops = d_model * hidden
    flops += n_candidates * d_model * hidden
    flops += n_candidates * hidden
    return flops
