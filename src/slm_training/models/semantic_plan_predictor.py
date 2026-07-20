"""SLM-144 SPV1-01: tiny semantic-plan predictor heads and fixture trainer.

This module provides standalone, CPU-trainable heads for archetype classification,
role-set prediction with a learned-slot bipartite loss, and serialized role
inventory decoding. It is intentionally separate from production TwoTower wiring
and is meant only as a fixture/evidence harness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import SemanticPlanV1

__all__ = [
    "SemanticPlanPredictor",
    "ArchetypeClassifierHead",
    "RoleSetPredictorHead",
    "SerializedInventoryHead",
    "PlanTrainingExample",
    "PlanBatchCollator",
    "train_fixture_predictor",
    "featurize_program_spec",
    "build_role_set_target",
    "predict_role_set_from_logits",
    "predict_serialized_inventory",
]


class SemanticPlanPredictor(Protocol):
    """Protocol for a predictor that maps a feature vector to plan factors."""

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]: ...


class ArchetypeClassifierHead(nn.Module):
    """Small MLP archetype classifier."""

    def __init__(
        self,
        input_dim: int,
        num_archetypes: int,
        *,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_archetypes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class RoleSetPredictorHead(nn.Module):
    """Learned-slot role-set predictor with bipartite matching loss.

    A fixed set of ``num_slots`` slot queries is combined with the input
    embedding. Each slot scores every role (plus a blank role) via a dot
    product with a learned role embedding table. The training loss greedily
    matches predicted slots to gold roles and pushes unmatched slots toward the
    blank class.
    """

    def __init__(
        self,
        input_dim: int,
        num_roles: int,
        num_slots: int,
        *,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_roles = num_roles
        self.num_slots = num_slots
        self.blank_role = num_roles
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.slot_queries = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.role_embed = nn.Embedding(num_roles, hidden_dim)
        self.scale = hidden_dim**-0.5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return slot logits of shape (batch, num_slots, num_roles + 1)."""
        batch_size = inputs.shape[0]
        hidden = self.input_proj(inputs)  # (B, H)
        slot_emb = hidden.unsqueeze(1) + self.slot_queries.unsqueeze(0)  # (B, S, H)
        role_emb = self.role_embed.weight * self.scale  # (R, H)
        logits = torch.matmul(slot_emb, role_emb.t())  # (B, S, R)
        blank_bias = torch.zeros(
            batch_size, self.num_slots, 1, device=inputs.device, dtype=inputs.dtype
        )
        return torch.cat([logits, blank_bias], dim=-1)


def _greedy_match_slots(
    slot_logits: torch.Tensor, role_set_mask: torch.Tensor
) -> list[tuple[int, int]]:
    """Greedily match predicted slots to gold roles using negative logit cost."""
    gold_roles = role_set_mask.nonzero(as_tuple=True)[0].tolist()
    available = list(range(slot_logits.shape[0]))
    matched: list[tuple[int, int]] = []
    for role_idx in gold_roles:
        if not available:
            break
        costs = -slot_logits[available, role_idx]
        best_local = int(costs.argmin())
        slot_idx = available.pop(best_local)
        matched.append((slot_idx, role_idx))
    return matched


def role_set_bipartite_loss(
    slot_logits: torch.Tensor,
    role_set_masks: torch.Tensor,
    blank_role: int,
) -> torch.Tensor:
    """Bipartite matching loss for learned role-set slots.

    Each gold role is assigned to the slot with the highest score for that role.
    Unmatched slots are supervised toward the blank class.
    """
    batch_size, num_slots, _ = slot_logits.shape
    total = torch.tensor(0.0, device=slot_logits.device)
    for b in range(batch_size):
        matched = _greedy_match_slots(slot_logits[b], role_set_masks[b])
        if matched:
            slots, roles = zip(*matched)
            slots_t = torch.tensor(slots, device=slot_logits.device, dtype=torch.long)
            roles_t = torch.tensor(roles, device=slot_logits.device, dtype=torch.long)
            total += F.cross_entropy(
                slot_logits[b][slots_t], roles_t, reduction="sum"
            )
        unmatched = [s for s in range(num_slots) if s not in {m[0] for m in matched}]
        if unmatched:
            unmatched_t = torch.tensor(
                unmatched, device=slot_logits.device, dtype=torch.long
            )
            blank_t = torch.full(
                (len(unmatched),),
                blank_role,
                device=slot_logits.device,
                dtype=torch.long,
            )
            total += F.cross_entropy(
                slot_logits[b][unmatched_t], blank_t, reduction="sum"
            )
    return total / batch_size


class SerializedInventoryHead(nn.Module):
    """Autoregressive serialized-inventory head over a fixed role vocabulary."""

    def __init__(
        self,
        input_dim: int,
        num_roles: int,
        max_len: int,
        *,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_roles = num_roles
        self.max_len = max_len
        self.start_token = num_roles
        self.embed = nn.Embedding(num_roles + 1, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_roles)
        self.hidden_init = nn.Linear(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, max_len, num_roles)."""
        batch_size = inputs.shape[0]
        device = inputs.device
        hidden = self.hidden_init(inputs)
        token = torch.full((batch_size,), self.start_token, device=device)
        logits_list: list[torch.Tensor] = []
        for _ in range(self.max_len):
            embedded = self.embed(token)
            hidden = self.gru(embedded, hidden)
            logits_list.append(self.out(hidden))
            token = logits_list[-1].argmax(dim=-1)
        return torch.stack(logits_list, dim=1)

    def teacher_forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced forward for training.

        ``targets`` has shape (batch, max_len) with role indices and ``-1`` pads;
        the start token is prepended internally and pads are remapped to the
        start token for embedding only.
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        hidden = self.hidden_init(inputs)
        start = torch.full(
            (batch_size, 1), self.start_token, device=device, dtype=torch.long
        )
        shifted = torch.cat([start, targets[:, :-1]], dim=1)
        shifted = shifted.clamp(min=0)
        embedded = self.embed(shifted)  # (B, L, H)
        outputs: list[torch.Tensor] = []
        for t in range(self.max_len):
            hidden = self.gru(embedded[:, t], hidden)
            outputs.append(self.out(hidden))
        return torch.stack(outputs, dim=1)


@dataclass(frozen=True)
class PlanTrainingExample:
    """One training example for the fixture semantic-plan predictor."""

    example_id: str
    input_features: torch.Tensor
    archetype_label: int
    role_set_mask: torch.Tensor
    serialized_roles: torch.Tensor
    source_plan: SemanticPlanV1 | None = None
    program_spec: ProgramSpec | None = None


@dataclass(frozen=True)
class PlanBatchCollator:
    """Collate ``PlanTrainingExample`` instances into a batched tensor dict."""

    pad_role_index: int = -1

    def __call__(
        self, examples: Sequence[PlanTrainingExample]
    ) -> dict[str, torch.Tensor | list[str]]:
        return {
            "example_ids": [e.example_id for e in examples],
            "input_features": torch.stack([e.input_features for e in examples]),
            "archetype_labels": torch.tensor(
                [e.archetype_label for e in examples], dtype=torch.long
            ),
            "role_set_masks": torch.stack([e.role_set_mask for e in examples]),
            "serialized_roles": torch.stack([e.serialized_roles for e in examples]),
        }


def featurize_program_spec(
    spec: ProgramSpec,
    family_vocab: dict[str, int],
) -> torch.Tensor:
    """Return a count vector of component families present in ``spec.ast``."""
    vec = torch.zeros(len(family_vocab), dtype=torch.float32)
    ast = spec.ast
    if not isinstance(ast, dict):
        return vec

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            family = node.get("typeName")
            if isinstance(family, str) and family in family_vocab:
                vec[family_vocab[family]] += 1.0
            for child in node.get("props", {}).get("children", []):
                _walk(child)

    _walk(ast.get("root") if isinstance(ast.get("root"), dict) else ast)
    return vec


def build_role_set_target(
    role_ids: Sequence[str],
    role_vocab: dict[str, int],
    num_roles: int,
) -> torch.Tensor:
    """Build a binary mask target for the role-set predictor."""
    mask = torch.zeros(num_roles, dtype=torch.float32)
    for role_id in role_ids:
        idx = role_vocab.get(role_id)
        if idx is not None:
            mask[idx] = 1.0
    return mask


def _serialize_role_set(
    role_ids: Sequence[str],
    role_vocab: dict[str, int],
    max_len: int,
    pad_index: int = -1,
) -> torch.Tensor:
    """Canonicalize a role set as a fixed-length sorted integer sequence."""
    indices = sorted(
        {role_vocab[r] for r in role_ids if r in role_vocab},
        key=lambda i: i,
    )
    padded = (indices + [pad_index] * max_len)[:max_len]
    return torch.tensor(padded, dtype=torch.long)


def predict_role_set_from_logits(
    slot_logits: torch.Tensor,
    blank_role: int,
) -> list[int]:
    """Return the deduplicated set of predicted role indices from slot logits."""
    preds = slot_logits.argmax(dim=-1).tolist()
    return sorted({r for r in preds if r != blank_role})


def predict_serialized_inventory(
    inventory_logits: torch.Tensor,
    pad_index: int = -1,
) -> list[int]:
    """Return the deduplicated predicted role sequence, stopping at pad."""
    preds = inventory_logits.argmax(dim=-1).tolist()
    result: list[int] = []
    seen: set[int] = set()
    for r in preds:
        if r == pad_index:
            break
        if r not in seen:
            result.append(r)
            seen.add(r)
    return result


def _set_metrics(
    pred_sets: Sequence[set[int]],
    gold_sets: Sequence[set[int]],
) -> dict[str, float]:
    if not pred_sets:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(len(p & g) for p, g in zip(pred_sets, gold_sets))
    pred_total = sum(len(p) for p in pred_sets)
    gold_total = sum(len(g) for g in gold_sets)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def train_fixture_predictor(
    train_examples: Sequence[PlanTrainingExample],
    val_examples: Sequence[PlanTrainingExample] | None = None,
    *,
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train the three fixture heads on CPU.

    Returns the trained modules, final metrics, and training history. The
    network is tiny on purpose: this is wiring evidence, not a production
    checkpoint.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if not train_examples:
        raise ValueError("train_examples must not be empty")

    input_dim = int(train_examples[0].input_features.shape[0])
    num_roles = int(train_examples[0].role_set_mask.shape[0])
    num_archetypes = int(
        max(e.archetype_label for e in train_examples) + 1
    )
    max_len = int(train_examples[0].serialized_roles.shape[0])
    num_slots = max_len

    archetype_head = ArchetypeClassifierHead(input_dim, num_archetypes).to(device)
    role_head = RoleSetPredictorHead(input_dim, num_roles, num_slots).to(device)
    inventory_head = SerializedInventoryHead(input_dim, num_roles, max_len).to(device)

    optimizer = torch.optim.Adam(
        list(archetype_head.parameters())
        + list(role_head.parameters())
        + list(inventory_head.parameters()),
        lr=lr,
    )
    collator = PlanBatchCollator()
    history: list[dict[str, float]] = []

    def _eval(
        examples: Sequence[PlanTrainingExample],
    ) -> dict[str, float]:
        archetype_head.eval()
        role_head.eval()
        inventory_head.eval()
        batch = collator(examples)
        inputs = batch["input_features"].to(device)
        labels = batch["archetype_labels"].to(device)
        role_masks = batch["role_set_masks"].to(device)
        serialized = batch["serialized_roles"].to(device)

        with torch.no_grad():
            arch_logits = archetype_head(inputs)
            role_logits = role_head(inputs)
            inv_logits = inventory_head.teacher_forward(inputs, serialized)

            arch_acc = (
                (arch_logits.argmax(dim=-1) == labels).float().mean().item()
            )
            blank = role_head.blank_role
            pred_sets = [
                set(predict_role_set_from_logits(role_logits[i], blank))
                for i in range(role_logits.shape[0])
            ]
            gold_sets = [
                set(role_masks[i].nonzero(as_tuple=True)[0].tolist())
                for i in range(role_masks.shape[0])
            ]
            role_metrics = _set_metrics(pred_sets, gold_sets)

            inv_acc = (
                (inv_logits.argmax(dim=-1) == serialized)
                .float()
                .mean()
                .item()
            )
        return {
            "archetype_accuracy": arch_acc,
            "role_precision": role_metrics["precision"],
            "role_recall": role_metrics["recall"],
            "role_f1": role_metrics["f1"],
            "inventory_token_accuracy": inv_acc,
        }

    for epoch in range(epochs):
        archetype_head.train()
        role_head.train()
        inventory_head.train()
        indices = list(range(len(train_examples)))
        random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_examples), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = collator([train_examples[i] for i in batch_idx])
            inputs = batch["input_features"].to(device)
            labels = batch["archetype_labels"].to(device)
            role_masks = batch["role_set_masks"].to(device)
            serialized = batch["serialized_roles"].to(device)

            arch_logits = archetype_head(inputs)
            role_logits = role_head(inputs)
            inv_logits = inventory_head.teacher_forward(inputs, serialized)

            loss = (
                F.cross_entropy(arch_logits, labels)
                + role_set_bipartite_loss(
                    role_logits, role_masks, role_head.blank_role
                )
                + F.cross_entropy(
                    inv_logits.transpose(1, 2),
                    serialized,
                    ignore_index=collator.pad_role_index,
                )
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        entry: dict[str, float] = {
            "epoch": float(epoch),
            "loss": epoch_loss / max(n_batches, 1),
        }
        if val_examples:
            entry.update(_eval(val_examples))
        history.append(entry)

    final_train = _eval(train_examples)
    final_val = _eval(val_examples) if val_examples else {}

    return {
        "archetype_head": archetype_head,
        "role_set_head": role_head,
        "serialized_inventory_head": inventory_head,
        "input_dim": input_dim,
        "num_roles": num_roles,
        "num_archetypes": num_archetypes,
        "max_len": max_len,
        "num_slots": num_slots,
        "final_train_metrics": final_train,
        "final_val_metrics": final_val,
        "history": history,
    }
