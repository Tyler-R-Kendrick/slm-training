"""Verifier-guided minimal semantic repair targets (SPV2-05).

This is a fixture-only wiring baseline: it turns the existing hard-valid
corruption taxonomy into replayable repair records, scores compiler-legal edits
with a tiny learned model, and compares deterministic baseline policies. It does
not run a full TwoTower train, does not download an external teacher, and makes
no ship readiness claim. Real verifier-backed counterfactual action values are
deferred to SLM-131 / VSS finite replay.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
except Exception:  # pragma: no cover - torch may be absent in minimal environments
    torch = None  # type: ignore[assignment]

from slm_training.data.corrupt import (
    CorruptionCase,
    CorruptionOperator,
    OperatorFamily,
    generate_corruptions,
)
from slm_training.dsl.lang_core import validate
from slm_training.versioning import build_version_stamp


__all__ = [
    "ConflictSlice",
    "LegalEdit",
    "RepairEvidence",
    "RepairFeatureExtractor",
    "RepairPolicyName",
    "SemanticRepairRecordV1",
    "SemanticRepairScorer",
    "apply_repair_policy",
    "build_repair_records_from_corruption",
    "train_repair_policy_fixture",
]


RepairPolicyName = Literal["random", "edit_distance", "oracle", "learned"]
_COMPLETENESS_CLASSES = ("EXACT", "SOUND_OVERAPPROX", "HEURISTIC")

_TOKEN_RE = re.compile(r'\n|"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]')


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("semantic_repair scorer requires torch")


def _canonical(source: str) -> str:
    program = validate(source)
    return (program.serialized or source).strip()


def _fingerprint(source: str) -> str:
    return hashlib.sha256(_canonical(source).encode("utf-8")).hexdigest()[:32]


def _tokenize(source: str) -> list[str]:
    return _TOKEN_RE.findall(source)


def _token_edit_distance(left: str, right: str) -> int:
    """Token-level Levenshtein distance used by the repair cost model."""
    a, b = _tokenize(left), _tokenize(right)
    previous = list(range(len(b) + 1))
    for i, lhs in enumerate(a, start=1):
        current = [i]
        for j, rhs in enumerate(b, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lhs != rhs))
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class RepairEvidence:
    """One reason-coded failure observation from the verifier/contract cascade."""

    reason_code: str
    analyzer: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "analyzer": self.analyzer,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepairEvidence":
        return cls(
            reason_code=data.get("reason_code", ""),
            analyzer=data.get("analyzer", ""),
            detail=data.get("detail", ""),
        )


@dataclass(frozen=True)
class ConflictSlice:
    """Localized dependency/conflict slice authorizing a bounded repair."""

    stage: str
    failing_node_ids: tuple[str | int, ...]
    dependency_frontier: tuple[str | int, ...]
    protected_node_ids: tuple[str | int, ...]
    completeness_class: Literal["EXACT", "SOUND_OVERAPPROX", "HEURISTIC"]
    source_provenance: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "failing_node_ids": list(self.failing_node_ids),
            "dependency_frontier": list(self.dependency_frontier),
            "protected_node_ids": list(self.protected_node_ids),
            "completeness_class": self.completeness_class,
            "source_provenance": self.source_provenance,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictSlice":
        completeness = data.get("completeness_class", "HEURISTIC")
        if completeness not in _COMPLETENESS_CLASSES:
            completeness = "HEURISTIC"
        return cls(
            stage=data.get("stage", ""),
            failing_node_ids=tuple(data.get("failing_node_ids", [])),
            dependency_frontier=tuple(data.get("dependency_frontier", [])),
            protected_node_ids=tuple(data.get("protected_node_ids", [])),
            completeness_class=completeness,  # type: ignore[arg-type]
            source_provenance=data.get("source_provenance", ""),
            notes=data.get("notes", ""),
        )

    def can_authorize_repair(self) -> bool:
        """Only EXACT or SOUND_OVERAPPROX slices drive primary localized repair."""
        return self.completeness_class in ("EXACT", "SOUND_OVERAPPROX")


@dataclass(frozen=True)
class LegalEdit:
    """One compiler-legal edit action available in a corrupted program state."""

    edit_id: str
    kind: str
    before: str
    after: str
    cost: int
    source: str = "oracle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
            "cost": self.cost,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegalEdit":
        return cls(
            edit_id=data.get("edit_id", ""),
            kind=data.get("kind", ""),
            before=data.get("before", ""),
            after=data.get("after", ""),
            cost=data.get("cost", 0),
            source=data.get("source", "oracle"),
        )


@dataclass(frozen=True)
class SemanticRepairRecordV1:
    """Replayable minimal-semantic-repair supervision record."""

    record_id: str
    source_fingerprint: str
    broken_openui: str
    failure_evidence: tuple[RepairEvidence, ...]
    conflict_slice: ConflictSlice
    legal_edits: tuple[LegalEdit, ...]
    accepted_edit_ids: tuple[str, ...]
    oracle_edit_id: str | None
    lineage: dict[str, Any]
    metadata: dict[str, Any]
    schema_version: str = "semantic_repair/v1"
    version_stamp: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "source_fingerprint": self.source_fingerprint,
            "broken_openui": self.broken_openui,
            "failure_evidence": [e.to_dict() for e in self.failure_evidence],
            "conflict_slice": self.conflict_slice.to_dict(),
            "legal_edits": [e.to_dict() for e in self.legal_edits],
            "accepted_edit_ids": list(self.accepted_edit_ids),
            "oracle_edit_id": self.oracle_edit_id,
            "lineage": self.lineage,
            "metadata": self.metadata,
            "version_stamp": self.version_stamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticRepairRecordV1":
        return cls(
            record_id=data.get("record_id", ""),
            source_fingerprint=data.get("source_fingerprint", ""),
            broken_openui=data.get("broken_openui", ""),
            failure_evidence=tuple(
                RepairEvidence.from_dict(e) for e in data.get("failure_evidence", [])
            ),
            conflict_slice=ConflictSlice.from_dict(data.get("conflict_slice", {})),
            legal_edits=tuple(
                LegalEdit.from_dict(e) for e in data.get("legal_edits", [])
            ),
            accepted_edit_ids=tuple(data.get("accepted_edit_ids", [])),
            oracle_edit_id=data.get("oracle_edit_id"),
            lineage=dict(data.get("lineage", {})),
            metadata=dict(data.get("metadata", {})),
            schema_version=data.get("schema_version", "semantic_repair/v1"),
            version_stamp=data.get("version_stamp"),
        )


def _parse_diagnostic(diag: str) -> RepairEvidence:
    """Split ``analyzer:detail`` diagnostics into structured evidence."""
    if ":" in diag:
        reason, detail = diag.split(":", 1)
    else:
        reason, detail = diag, ""
    return RepairEvidence(
        reason_code=reason.strip().lower(),
        analyzer=reason.strip().lower(),
        detail=detail.strip(),
    )


def _case_to_record(
    case: CorruptionCase,
    source_fingerprint: str,
    record_id: str,
    *,
    completeness_class: Literal["EXACT", "SOUND_OVERAPPROX", "HEURISTIC"] | None = None,
) -> SemanticRepairRecordV1:
    """Project one verified corruption case to a repair record."""
    evidence = tuple(_parse_diagnostic(d) for d in case.diagnostics)
    slice_completeness: Literal["EXACT", "SOUND_OVERAPPROX", "HEURISTIC"] = (
        "EXACT" if case.exact_repair else "SOUND_OVERAPPROX"
    )
    if completeness_class is not None:
        slice_completeness = completeness_class

    conflict_slice = ConflictSlice(
        stage=case.family.value,
        failing_node_ids=case.failure_cone or (case.location,),
        dependency_frontier=case.ast_path,
        protected_node_ids=case.preserved_nodes,
        completeness_class=slice_completeness,
        source_provenance=f"oracle:{case.operator.value}",
        notes=f"location={case.location}; edit_distance={case.edit_distance}",
    )

    legal_edits: list[LegalEdit] = []
    for idx, repair in enumerate(case.acceptable_repairs):
        edit_id = f"{record_id}-e{idx}"
        legal_edits.append(
            LegalEdit(
                edit_id=edit_id,
                kind="replace_program",
                before=case.broken_openui,
                after=repair,
                cost=_token_edit_distance(case.broken_openui, repair),
                source="oracle_acceptable",
            )
        )

    accepted_ids = tuple(e.edit_id for e in legal_edits)
    oracle_id = accepted_ids[0] if accepted_ids else None

    lineage = {
        "source_fingerprint": source_fingerprint,
        "operator": case.operator.value,
        "operator_family": case.family.value,
        "location": case.location,
        "clean_ast": case.clean_ast,
        "ast_path": list(case.ast_path),
        "source_span": list(case.source_span),
        "edit_distance": case.edit_distance,
    }
    metadata = {
        "family": case.family.value,
        "operator": case.operator.value,
        "location": case.location,
        "exact_repair": case.exact_repair,
        "edit_distance": case.edit_distance,
        "n_legal_edits": len(legal_edits),
    }

    return SemanticRepairRecordV1(
        record_id=record_id,
        source_fingerprint=source_fingerprint,
        broken_openui=case.broken_openui,
        failure_evidence=evidence,
        conflict_slice=conflict_slice,
        legal_edits=tuple(legal_edits),
        accepted_edit_ids=accepted_ids,
        oracle_edit_id=oracle_id,
        lineage=lineage,
        metadata=metadata,
        version_stamp=build_version_stamp(
            "harness.distill", "data.semantic_contrast"
        ),
    )


def build_repair_records_from_corruption(
    clean_openui: str,
    *,
    record_id_prefix: str = "repair",
    completeness_class: Literal["EXACT", "SOUND_OVERAPPROX", "HEURISTIC"] | None = None,
) -> tuple[SemanticRepairRecordV1, ...]:
    """Generate a repair record for every applicable catalog corruption.

    Uses the existing hard-valid corruption taxonomy
    (``slm_training.data.corrupt.oracle``) so every broken program is rejected by
    the authoritative verifier and every legal edit is a known acceptable repair.
    """
    canonical = _canonical(clean_openui)
    source_fp = _fingerprint(canonical)
    cases = generate_corruptions(canonical)
    records: list[SemanticRepairRecordV1] = []
    for idx, case in enumerate(cases):
        record_id = f"{record_id_prefix}-{case.operator.value}-{idx}"
        records.append(
            _case_to_record(
                case,
                source_fingerprint=source_fp,
                record_id=record_id,
                completeness_class=completeness_class,
            )
        )
    return tuple(records)


class RepairFeatureExtractor:
    """Torch-free feature extractor for repair scoring."""

    def __init__(self) -> None:
        self.family_to_idx = {family.value: i for i, family in enumerate(OperatorFamily)}
        self.operator_to_idx = {
            operator.value: i for i, operator in enumerate(CorruptionOperator)
        }
        self.float_keys = [
            "normalized_distance",
            "cost",
            "n_preserved_nodes",
            "n_diagnostics",
            "exact_repair",
            "legal_set_size",
        ]

    def extract(
        self,
        record: SemanticRepairRecordV1,
        edit: LegalEdit,
    ) -> dict[str, float | int]:
        """Return a feature dictionary with a trailing ``accepted`` label."""
        broken_tokens = _tokenize(record.broken_openui)
        after_tokens = _tokenize(edit.after)
        max_len = max(len(broken_tokens), len(after_tokens), 1)
        distance = _token_edit_distance(record.broken_openui, edit.after)

        family = str(record.metadata.get("family", ""))
        operator = str(record.metadata.get("operator", ""))

        features: dict[str, float | int] = {
            "normalized_distance": 1.0 - distance / max_len,
            "cost": float(edit.cost),
            "n_preserved_nodes": float(len(record.conflict_slice.protected_node_ids)),
            "n_diagnostics": float(len(record.failure_evidence)),
            "exact_repair": 1.0 if record.metadata.get("exact_repair") else 0.0,
            "legal_set_size": float(len(record.legal_edits)),
            "family_idx": self.family_to_idx.get(family, 0),
            "operator_idx": self.operator_to_idx.get(operator, 0),
            "accepted": 1.0 if edit.edit_id in record.accepted_edit_ids else 0.0,
        }
        return features

    def to_scorer_inputs(
        self, features: dict[str, float | int]
    ) -> dict[str, torch.Tensor]:
        """Convert a feature dictionary to the tensor dictionary the scorer expects."""
        _require_torch()
        return {
            "floats": torch.tensor(
                [float(features[k]) for k in self.float_keys],
                dtype=torch.float32,
            ),
            "family_idx": torch.tensor(
                int(features["family_idx"]), dtype=torch.long
            ),
            "operator_idx": torch.tensor(
                int(features["operator_idx"]), dtype=torch.long
            ),
        }


class SemanticRepairScorer(torch.nn.Module):
    """Tiny learned scorer for compiler-legal repair edits."""

    def __init__(
        self,
        *,
        float_dim: int = 6,
        family_vocab: int = len(OperatorFamily),
        operator_vocab: int = len(CorruptionOperator),
        family_dim: int = 4,
        operator_dim: int = 4,
        hidden_dim: int = 16,
    ) -> None:
        _require_torch()
        super().__init__()
        self.float_dim = float_dim
        self.family_embed = torch.nn.Embedding(family_vocab, family_dim)
        self.operator_embed = torch.nn.Embedding(operator_vocab, operator_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(float_dim + family_dim + operator_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        floats = inputs["floats"]
        if floats.ndim == 1:
            floats = floats.unsqueeze(0)
        family = inputs["family_idx"]
        if family.ndim == 0:
            family = family.unsqueeze(0)
        operator = inputs["operator_idx"]
        if operator.ndim == 0:
            operator = operator.unsqueeze(0)
        embedded = torch.cat(
            [
                floats,
                self.family_embed(family),
                self.operator_embed(operator),
            ],
            dim=-1,
        )
        return self.mlp(embedded).squeeze(-1)

    def score(
        self,
        record: SemanticRepairRecordV1,
        edit: LegalEdit,
        extractor: RepairFeatureExtractor,
    ) -> float:
        """Return a scalar score for ``edit`` in the context of ``record``."""
        features = extractor.extract(record, edit)
        features.pop("accepted", None)
        inputs = extractor.to_scorer_inputs(features)
        with torch.no_grad():
            return float(self(inputs).item())


def apply_repair_policy(
    record: SemanticRepairRecordV1,
    policy: RepairPolicyName,
    *,
    scorer: SemanticRepairScorer | None = None,
    extractor: RepairFeatureExtractor | None = None,
    rng: random.Random | None = None,
) -> tuple[LegalEdit, dict[str, Any]]:
    """Select one legal edit according to ``policy``.

    ``oracle`` returns the lowest-cost accepted edit. ``edit_distance`` returns
    the lowest-cost legal edit regardless of acceptance. ``random`` samples
    uniformly from the legal set. ``learned`` returns the highest scorer output.
    """
    if not record.legal_edits:
        raise ValueError("record has no legal edits")

    if rng is None:
        rng = random.Random(0)

    if policy == "random":
        chosen = rng.choice(record.legal_edits)
        return chosen, {"policy": policy, "score": None, "rank": None}

    if policy == "edit_distance":
        chosen = min(record.legal_edits, key=lambda e: e.cost)
        return chosen, {"policy": policy, "score": None, "rank": None}

    if policy == "oracle":
        accepted = [e for e in record.legal_edits if e.edit_id in record.accepted_edit_ids]
        if not accepted:
            chosen = rng.choice(record.legal_edits)
            return chosen, {"policy": policy, "score": None, "rank": None, "unknown": True}
        chosen = min(accepted, key=lambda e: e.cost)
        return chosen, {"policy": policy, "score": None, "rank": None}

    if policy == "learned":
        if scorer is None or extractor is None:
            raise ValueError("learned policy requires scorer and extractor")
        scored = [
            (edit, scorer.score(record, edit, extractor))
            for edit in record.legal_edits
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        chosen = scored[0][0]
        rank = {edit.edit_id: i for i, (edit, _) in enumerate(scored)}
        return chosen, {
            "policy": policy,
            "score": scored[0][1],
            "rank": rank,
        }

    raise ValueError(f"unknown repair policy: {policy}")


def train_repair_policy_fixture(
    records: list[SemanticRepairRecordV1],
    scorer: SemanticRepairScorer,
    extractor: RepairFeatureExtractor,
    *,
    steps: int = 30,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Tiny fixture trainer that ranks accepted edits above non-accepted edits.

    Uses a binary logit loss over the complete legal edit set for each record.
    The teacher signal is ``1`` for every accepted edit and ``0`` for every other
    legal edit. Returns loss history and a lightweight metric summary.
    """
    _require_torch()
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)

    decisions: list[tuple[dict[str, torch.Tensor], float]] = []
    for record in records:
        for edit in record.legal_edits:
            features = extractor.extract(record, edit)
            label = float(features.pop("accepted"))
            inputs = extractor.to_scorer_inputs(features)
            decisions.append((inputs, label))

    if not decisions:
        return {"steps": steps, "lr": lr, "n_decisions": 0, "history": []}

    criterion = torch.nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []

    for step in range(steps):
        total_loss = scorer.forward(decisions[0][0]).new_zeros(())
        for inputs, label in decisions:
            score = scorer(inputs)
            target = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            total_loss = total_loss + criterion(score, target)
        loss = total_loss / len(decisions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append({"step": step + 1, "loss": float(loss.detach())})

    final_loss = history[-1]["loss"] if history else float("nan")
    return {
        "steps": steps,
        "lr": lr,
        "n_decisions": len(decisions),
        "history": history,
        "final_loss": final_loss,
    }
