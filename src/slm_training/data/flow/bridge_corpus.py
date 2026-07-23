"""Canonical full-bridge rows and exact dynamic legal-edit candidate sets.

The module is deliberately objective-neutral.  It materializes inference-time
candidate membership, replay evidence, and labels, but contains no scorer,
rate, loss, or model-selection policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.parser import validate
from slm_training.harnesses.experiments.slm188_edit_algebra import (
    CanonicalEdit,
    TransitionCertificateV1,
    apply_canonical_edit,
    plan_edit_sequence,
)
from slm_training.models.tree_edit_diffusion import parse_statements

ROW_SCHEMA = "LegalEditBridgeRowV1"
CANDIDATE_SCHEMA = "LegalEditCandidateV1"
CANDIDATE_SET_SCHEMA = "ExactLegalEditCandidateSetV1"
MANIFEST_SCHEMA = "LegalEditBridgeCorpusManifestV1"
PLANNER_MANIFEST_SCHEMA = "LegalEditPlannerManifestV1"

_FORBIDDEN_INPUT_TERMS = frozenset(
    {
        "confirmation",
        "future_witness",
        "gold",
        "hidden_gold",
        "target_ast",
        "target_program",
        "planner_selected",
        "positive_candidate_ids",
    }
)
_MODEL_FEATURE_KEYS = frozenset(
    {
        "action_kind",
        "arity",
        "cardinality",
        "enum_value",
        "frame",
        "literal_kind",
        "node_pointer",
        "production",
        "slot_pointer",
        "successor_fingerprint",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _semantic_edit(edit: CanonicalEdit | Mapping[str, Any]) -> dict[str, Any]:
    raw = edit.to_dict() if isinstance(edit, CanonicalEdit) else dict(edit)
    return {
        key: raw.get(key)
        for key in (
            "action",
            "target_name",
            "production",
            "child_name",
            "slot",
            "direction",
            "index",
            "previous_index",
        )
    }


def stable_candidate_id(edit: CanonicalEdit | Mapping[str, Any]) -> str:
    """Return a permanent semantic ID, independent of row order and timing."""
    return f"edit_{content_digest(_semantic_edit(edit))[:24]}"


def _certificate(
    source: str,
    target: str,
    edit: CanonicalEdit,
    version_pins: Mapping[str, Any],
) -> TransitionCertificateV1:
    return TransitionCertificateV1(
        source_fingerprint=canonical_fingerprint(source),
        target_fingerprint=canonical_fingerprint(target),
        edit=edit.to_dict(),
        source_program=source,
        target_program=target,
        verifier_accepted=True,
        verifier_detail="canonical_parse_replay_ok",
        version_pins=dict(version_pins),
    )


def verify_certificate(certificate: Mapping[str, Any]) -> None:
    cert = TransitionCertificateV1.from_dict(dict(certificate))
    expected = content_digest(cert.to_dict(exclude_digest=True))
    if cert.schema != "TransitionCertificateV1":
        raise ValueError(f"unsupported transition certificate schema: {cert.schema}")
    if cert.certificate_digest != expected:
        raise ValueError("transition certificate digest mismatch")
    if not cert.verifier_accepted or cert.source_program is None or cert.target_program is None:
        raise ValueError("transition certificate is not independently replayable")
    if canonical_fingerprint(cert.source_program) != cert.source_fingerprint:
        raise ValueError("transition certificate source fingerprint mismatch")
    replayed = apply_canonical_edit(
        cert.source_program, CanonicalEdit.from_dict(cert.edit)
    )
    if replayed is None or canonical_fingerprint(replayed) != cert.target_fingerprint:
        raise ValueError("transition certificate replay mismatch")
    if canonical_fingerprint(cert.target_program) != cert.target_fingerprint:
        raise ValueError("transition certificate target fingerprint mismatch")


@dataclass(frozen=True)
class RequestEditContractV1:
    """Inference-visible finite edit domain for one request."""

    productions: tuple[tuple[str, str], ...]
    slots: tuple[str, ...]
    statement_names: tuple[str, ...]
    enum_values: tuple[str, ...] = ("column", "row")
    source_policy: str = "declared_request_contract"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequestEditContractV1":
        productions = tuple(
            sorted(
                (str(item["name"]), str(item["kind"]))
                for item in data.get("productions", ())
            )
        )
        if not productions or any(kind not in {"container", "leaf"} for _, kind in productions):
            raise ValueError("request contract needs typed leaf/container productions")
        return cls(
            productions=productions,
            slots=tuple(sorted({str(value) for value in data.get("slots", ())})),
            statement_names=tuple(
                sorted({str(value) for value in data.get("statement_names", ())})
            ),
            enum_values=tuple(
                sorted({str(value) for value in data.get("enum_values", ("column", "row"))})
            ),
            source_policy=str(data.get("source_policy", "declared_request_contract")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "RequestEditContractV1",
            "productions": [
                {"name": name, "kind": kind} for name, kind in self.productions
            ],
            "slots": list(self.slots),
            "statement_names": list(self.statement_names),
            "enum_values": list(self.enum_values),
            "source_policy": self.source_policy,
        }


@dataclass(frozen=True)
class LegalEditCandidateV1:
    candidate_id: str
    edit: dict[str, Any]
    successor_fingerprint: str
    transition_certificate: dict[str, Any]
    features: dict[str, Any]
    schema: str = CANDIDATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LegalEditCandidateV1":
        allowed = {
            "schema",
            "candidate_id",
            "edit",
            "successor_fingerprint",
            "transition_certificate",
            "features",
        }
        extra = set(data) - allowed
        if extra:
            raise ValueError(f"unknown candidate fields: {sorted(extra)}")
        candidate = cls(
            schema=str(data.get("schema", CANDIDATE_SCHEMA)),
            candidate_id=str(data["candidate_id"]),
            edit=dict(data["edit"]),
            successor_fingerprint=str(data["successor_fingerprint"]),
            transition_certificate=dict(data["transition_certificate"]),
            features=dict(data["features"]),
        )
        if candidate.schema != CANDIDATE_SCHEMA:
            raise ValueError(f"unsupported candidate schema: {candidate.schema}")
        if candidate.candidate_id != stable_candidate_id(candidate.edit):
            raise ValueError("candidate ID is not its canonical semantic ID")
        if set(candidate.features) - _MODEL_FEATURE_KEYS:
            raise ValueError("candidate contains a non-allowlisted model feature")
        verify_certificate(candidate.transition_certificate)
        return candidate


@dataclass(frozen=True)
class ExactLegalEditCandidateSetV1:
    state_fingerprint: str
    candidates: tuple[LegalEditCandidateV1, ...]
    candidate_set_digest: str = ""
    schema: str = CANDIDATE_SET_SCHEMA

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.candidates, key=lambda item: item.candidate_id))
        if ordered != self.candidates:
            object.__setattr__(self, "candidates", ordered)
        ids = [item.candidate_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate ID")
        expected = content_digest(
            {
                "schema": self.schema,
                "state_fingerprint": self.state_fingerprint,
                "candidates": [item.to_dict() for item in ordered],
            }
        )
        if self.candidate_set_digest and self.candidate_set_digest != expected:
            raise ValueError("candidate-set digest mismatch")
        object.__setattr__(self, "candidate_set_digest", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state_fingerprint": self.state_fingerprint,
            "candidates": [item.to_dict() for item in self.candidates],
            "candidate_set_digest": self.candidate_set_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExactLegalEditCandidateSetV1":
        if set(data) - {
            "schema",
            "state_fingerprint",
            "candidates",
            "candidate_set_digest",
        }:
            raise ValueError("unknown exact candidate-set fields")
        return cls(
            schema=str(data.get("schema", CANDIDATE_SET_SCHEMA)),
            state_fingerprint=str(data["state_fingerprint"]),
            candidates=tuple(
                LegalEditCandidateV1.from_dict(item)
                for item in data.get("candidates", ())
            ),
            candidate_set_digest=str(data.get("candidate_set_digest", "")),
        )


def _candidate_features(
    edit: CanonicalEdit, successor_fingerprint: str, pointer_ordinals: Mapping[str, int]
) -> dict[str, Any]:
    node_pointer = pointer_ordinals.get(edit.target_name)
    slot_pointer = pointer_ordinals.get(edit.slot or "")
    return {
        "action_kind": edit.action,
        "production": edit.production,
        "arity": 1 if edit.child_name else 0,
        "cardinality": 1,
        "node_pointer": node_pointer,
        "slot_pointer": slot_pointer,
        "literal_kind": "slot" if edit.slot is not None else None,
        "enum_value": edit.direction,
        "frame": None,
        "successor_fingerprint": successor_fingerprint,
    }


def enumerate_live_candidates(
    source: str,
    contract: RequestEditContractV1,
    *,
    version_pins: Mapping[str, Any] | None = None,
) -> ExactLegalEditCandidateSetV1:
    """Enumerate every valid edit in the finite request-declared domain."""
    validate(source)
    source = source.strip()
    statements = parse_statements(source)
    if statements is None:
        raise ValueError("source is outside the statement edit domain")
    by_name = {item.name: item for item in statements}
    pointer_values = sorted(
        {*contract.statement_names, *contract.slots, *by_name}
    )
    pointer_ordinals = {value: index for index, value in enumerate(pointer_values)}
    edits: list[CanonicalEdit] = []

    def add(**kwargs: Any) -> None:
        prototype = CanonicalEdit(edit_id="", **kwargs)
        edit_id = stable_candidate_id(prototype)
        edits.append(CanonicalEdit(edit_id=edit_id, **kwargs))

    for stmt in sorted(statements, key=lambda item: item.name):
        kind = "container" if stmt.has_list else "leaf"
        for production, candidate_kind in contract.productions:
            if candidate_kind == kind and production != stmt.comp:
                add(
                    action="ReplaceProduction",
                    target_name=stmt.name,
                    production=production,
                    inverse_action="ReplaceProduction",
                )
        if stmt.has_list:
            for enum_value in contract.enum_values:
                if enum_value not in stmt.rest:
                    add(
                        action="SetEnum",
                        target_name=stmt.name,
                        direction=enum_value,
                        inverse_action="SetEnum",
                    )
            for child in sorted(by_name):
                if child != stmt.name and child not in stmt.children:
                    add(
                        action="InsertChild",
                        target_name=stmt.name,
                        child_name=child,
                        inverse_action="DeleteChild",
                    )
            for child in sorted(stmt.children):
                add(
                    action="DeleteChild",
                    target_name=stmt.name,
                    child_name=child,
                    inverse_action="InsertChild",
                )
        else:
            for slot in contract.slots:
                if slot not in stmt.rest:
                    add(
                        action="BindSlotPointer",
                        target_name=stmt.name,
                        slot=slot,
                        inverse_action="BindSlotPointer",
                    )
        if stmt.name != "root":
            add(
                action="DeleteStatement",
                target_name=stmt.name,
                inverse_action="InsertStatement",
            )

    missing_names = sorted(set(contract.statement_names) - set(by_name))
    for name in missing_names:
        for production, kind in contract.productions:
            if kind == "leaf":
                for slot in contract.slots:
                    add(
                        action="InsertStatement",
                        target_name=name,
                        production=production,
                        slot=slot,
                        inverse_action="DeleteStatement",
                    )
            else:
                for child in sorted(by_name):
                    if child != "root":
                        add(
                            action="InsertStatement",
                            target_name=name,
                            production=production,
                            child_name=child,
                            direction=contract.enum_values[0],
                            inverse_action="DeleteStatement",
                        )

    candidates: list[LegalEditCandidateV1] = []
    seen_successors: set[tuple[str, str]] = set()
    for edit in edits:
        try:
            successor = apply_canonical_edit(source, edit)
            successor_fp = (
                canonical_fingerprint(successor) if successor is not None else ""
            )
        except Exception:  # noqa: BLE001
            successor = None
            successor_fp = ""
        if successor is None:
            continue
        identity = (edit.edit_id, successor_fp)
        if successor_fp == canonical_fingerprint(source) or identity in seen_successors:
            continue
        seen_successors.add(identity)
        cert = _certificate(source, successor, edit, version_pins or {})
        candidates.append(
            LegalEditCandidateV1(
                candidate_id=edit.edit_id,
                edit=edit.to_dict(),
                successor_fingerprint=successor_fp,
                transition_certificate=cert.to_dict(),
                features=_candidate_features(edit, successor_fp, pointer_ordinals),
            )
        )
    return ExactLegalEditCandidateSetV1(
        state_fingerprint=canonical_fingerprint(source),
        candidates=tuple(candidates),
    )


@dataclass(frozen=True)
class LegalEditBridgeRowV1:
    row_id: str
    bridge_id: str
    target_cluster_id: str
    program_family: str
    lineage: dict[str, Any]
    split_group: str
    split: str
    prompt_ref: str
    prompt_hash: str
    context_ref: str
    context_hash: str
    source_policy: str
    source_state_fingerprint: str
    target_state_fingerprint: str
    target_canonical_ast_digest: str
    state_fingerprint: str
    state_summary: dict[str, Any]
    exact_state_ref: str
    step_index: int
    bridge_length: int
    normalized_progress: float
    sampled_time: float
    focus: dict[str, Any]
    dependency_capsule: dict[str, Any]
    candidate_set_ref: str
    candidate_set_digest: str
    complete_candidate_ids: tuple[str, ...]
    candidate_successors: dict[str, str]
    positive_candidate_ids: tuple[str, ...]
    supported_candidate_ids: tuple[str, ...]
    unsupported_candidate_ids: tuple[str, ...]
    unknown_candidate_ids: tuple[str, ...]
    planner_selected_candidate_id: str | None
    remaining_edit_distance: int
    independence_groups: tuple[tuple[str, ...], ...]
    conflict_groups: tuple[tuple[str, ...], ...]
    transition_certificate_ids: tuple[str, ...]
    versions: dict[str, Any]
    cost_profile: dict[str, Any]
    schema: str = ROW_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LegalEditBridgeRowV1":
        allowed = set(cls.__dataclass_fields__)
        extra = set(data) - allowed
        if extra:
            raise ValueError(f"unknown bridge row fields: {sorted(extra)}")
        converted = dict(data)
        for name in (
            "complete_candidate_ids",
            "positive_candidate_ids",
            "supported_candidate_ids",
            "unsupported_candidate_ids",
            "unknown_candidate_ids",
            "transition_certificate_ids",
        ):
            converted[name] = tuple(converted[name])
        for name in ("independence_groups", "conflict_groups"):
            converted[name] = tuple(tuple(group) for group in converted[name])
        row = cls(**converted)
        row.validate()
        return row

    def validate(self) -> None:
        if self.schema != ROW_SCHEMA or self.split not in {"train", "dev"}:
            raise ValueError("unsupported bridge row schema or split")
        ids = set(self.complete_candidate_ids)
        if len(ids) != len(self.complete_candidate_ids):
            raise ValueError("duplicate complete candidate ID")
        partitions = [
            set(self.supported_candidate_ids),
            set(self.unsupported_candidate_ids),
            set(self.unknown_candidate_ids),
        ]
        if any(left & right for i, left in enumerate(partitions) for right in partitions[i + 1 :]):
            raise ValueError("candidate support partitions overlap")
        if set().union(*partitions) != ids:
            raise ValueError("candidate support partitions are not exhaustive")
        if not set(self.positive_candidate_ids) <= partitions[0]:
            raise ValueError("positive candidates must be certified supported")
        if (
            self.planner_selected_candidate_id is not None
            and self.planner_selected_candidate_id not in ids
        ):
            raise ValueError("planner-selected diagnostic is not live")
        if set(self.candidate_successors) != ids:
            raise ValueError("candidate successor mapping is incomplete")
        if not 0.0 <= self.normalized_progress <= 1.0:
            raise ValueError("normalized progress is outside [0, 1]")
        if not all(
            (
                self.prompt_ref,
                self.prompt_hash,
                self.context_ref,
                self.context_hash,
            )
        ):
            raise ValueError("prompt/context references and hashes must be non-empty")

    def model_input(self, candidates: ExactLegalEditCandidateSetV1) -> dict[str, Any]:
        if candidates.candidate_set_digest != self.candidate_set_digest:
            raise ValueError("row/candidate-set digest mismatch")
        payload = {
            "state_fingerprint": self.state_fingerprint,
            "state_summary": self.state_summary,
            "step_index": self.step_index,
            "normalized_progress": self.normalized_progress,
            "sampled_time": self.sampled_time,
            "focus": self.focus,
            "dependency_capsule": self.dependency_capsule,
            "candidate_set_digest": self.candidate_set_digest,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "features": item.features,
                }
                for item in candidates.candidates
            ],
        }
        _reject_forbidden(payload)
        return payload


def _reject_forbidden(value: Any, path: str = "model_input") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_INPUT_TERMS or "confirmation" in normalized:
                raise ValueError(f"forbidden model-input field at {path}.{key}")
            _reject_forbidden(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_forbidden(nested, f"{path}[{index}]")


def _target_consistent(successor: str, target: str, max_edits: int) -> bool:
    try:
        if canonical_fingerprint(successor) == canonical_fingerprint(target):
            return True
        edits, reason = plan_edit_sequence(successor, target)
        if reason != "planned" or len(edits) > max_edits:
            return False
        current = successor
        for edit in edits:
            current = apply_canonical_edit(current, edit) or ""
            if not current:
                return False
        return canonical_fingerprint(current) == canonical_fingerprint(target)
    except Exception:  # noqa: BLE001
        return False


def _split_for_cluster(cluster_id: str) -> str:
    return "dev" if int(content_digest(cluster_id)[:8], 16) % 5 == 0 else "train"


def build_bridge_rows(
    record: Mapping[str, Any],
    *,
    version_pins: Mapping[str, Any],
    max_edits: int = 12,
) -> tuple[list[LegalEditBridgeRowV1], dict[str, ExactLegalEditCandidateSetV1]]:
    """Build all non-terminal rows for one replayable canonical bridge."""
    _reject_forbidden(record.get("model_features", {}), "record.model_features")
    source = str(record["source_program"]).strip()
    target = str(record["target_program"]).strip()
    validate(source)
    validate(target)
    contract = RequestEditContractV1.from_dict(record["request_contract"])
    planned, reason = plan_edit_sequence(source, target)
    if reason != "planned" or len(planned) > max_edits:
        raise ValueError(f"bridge planner failed: {reason}")
    current = source
    bridge_identity = {
        "record_id": str(record["id"]),
        "source": canonical_fingerprint(source),
        "target": canonical_fingerprint(target),
        "edits": [_semantic_edit(edit) for edit in planned],
    }
    bridge_id = f"bridge_{content_digest(bridge_identity)[:24]}"
    cluster_id = str(record.get("target_cluster_id") or f"target_{content_digest(target)[:16]}")
    split = str(record.get("split") or _split_for_cluster(cluster_id))
    rows: list[LegalEditBridgeRowV1] = []
    candidate_sets: dict[str, ExactLegalEditCandidateSetV1] = {}
    for step_index, selected_edit in enumerate(planned):
        candidate_set = enumerate_live_candidates(
            current, contract, version_pins=version_pins
        )
        if not candidate_set.candidates:
            raise ValueError("non-terminal bridge state has no live candidate")
        candidate_sets[candidate_set.candidate_set_digest] = candidate_set
        selected_id = stable_candidate_id(selected_edit)
        if selected_id not in {item.candidate_id for item in candidate_set.candidates}:
            raise ValueError(
                f"planner-selected edit is absent from exact live set: {selected_edit.action}"
            )
        positive_ids: list[str] = []
        successors: dict[str, str] = {}
        for candidate in candidate_set.candidates:
            cert = candidate.transition_certificate
            successor = str(cert["target_program"])
            successors[candidate.candidate_id] = candidate.successor_fingerprint
            if _target_consistent(successor, target, max_edits - step_index):
                positive_ids.append(candidate.candidate_id)
        all_ids = tuple(item.candidate_id for item in candidate_set.candidates)
        positives = tuple(sorted(positive_ids))
        unknown = tuple(sorted(set(all_ids) - set(positives)))
        state_ref = f"states/{content_digest(current)}.json"
        target_fp = canonical_fingerprint(target)
        identity = {
            "bridge_id": bridge_id,
            "state": canonical_fingerprint(current),
            "candidate_set": candidate_set.candidate_set_digest,
            "step": step_index,
        }
        row = LegalEditBridgeRowV1(
            row_id=f"row_{content_digest(identity)[:24]}",
            bridge_id=bridge_id,
            target_cluster_id=cluster_id,
            program_family=str(record.get("program_family", "openui")),
            lineage=dict(record.get("lineage", {})),
            split_group=str(record.get("split_group", cluster_id)),
            split=split,
            prompt_ref=str(record["prompt_ref"]),
            prompt_hash=str(record["prompt_hash"]),
            context_ref=str(record["context_ref"]),
            context_hash=str(record["context_hash"]),
            source_policy=contract.source_policy,
            source_state_fingerprint=canonical_fingerprint(source),
            target_state_fingerprint=target_fp,
            target_canonical_ast_digest=target_fp,
            state_fingerprint=canonical_fingerprint(current),
            state_summary={
                "statement_count": len(parse_statements(current) or ()),
                "request_contract_digest": content_digest(contract.to_dict()),
            },
            exact_state_ref=state_ref,
            step_index=step_index,
            bridge_length=len(planned),
            normalized_progress=step_index / max(1, len(planned)),
            sampled_time=step_index / max(1, len(planned)),
            focus={"node_id": None, "source": "whole_state"},
            dependency_capsule={"groups": [], "source": "request_contract"},
            candidate_set_ref=f"candidate_sets/{candidate_set.candidate_set_digest}.json",
            candidate_set_digest=candidate_set.candidate_set_digest,
            complete_candidate_ids=all_ids,
            candidate_successors=successors,
            positive_candidate_ids=positives,
            supported_candidate_ids=positives,
            unsupported_candidate_ids=(),
            unknown_candidate_ids=unknown,
            planner_selected_candidate_id=selected_id,
            remaining_edit_distance=len(planned) - step_index,
            independence_groups=tuple((item,) for item in positives),
            conflict_groups=(),
            transition_certificate_ids=tuple(
                str(item.transition_certificate["certificate_digest"])
                for item in candidate_set.candidates
            ),
            versions=dict(version_pins),
            cost_profile={
                "candidate_count": len(all_ids),
                "cache_hit": False,
                "planner_selected_action": selected_edit.action,
            },
        )
        row.validate()
        rows.append(row)
        current = apply_canonical_edit(current, selected_edit) or ""
        if not current:
            raise ValueError("selected bridge transition did not replay")
    if canonical_fingerprint(current) != canonical_fingerprint(target):
        raise ValueError("bridge did not terminate at its target")
    return rows, candidate_sets


def validate_rows(
    rows: Iterable[LegalEditBridgeRowV1],
    candidate_sets: Mapping[str, ExactLegalEditCandidateSetV1],
) -> dict[str, Any]:
    row_list = list(rows)
    split_groups: dict[str, str] = {}
    target_clusters: dict[str, str] = {}
    for row in row_list:
        row.validate()
        candidate_set = candidate_sets.get(row.candidate_set_digest)
        if candidate_set is None:
            raise ValueError(f"missing exact candidate set: {row.candidate_set_digest}")
        if tuple(item.candidate_id for item in candidate_set.candidates) != row.complete_candidate_ids:
            raise ValueError("exact candidate-set reconstruction mismatch")
        if candidate_set.state_fingerprint != row.state_fingerprint:
            raise ValueError("candidate set belongs to a different state")
        row.model_input(candidate_set)
        for candidate in candidate_set.candidates:
            verify_certificate(candidate.transition_certificate)
            if candidate.successor_fingerprint != row.candidate_successors[candidate.candidate_id]:
                raise ValueError("candidate successor identity mismatch")
        for key, mapping in (
            (row.split_group, split_groups),
            (row.target_cluster_id, target_clusters),
        ):
            prior = mapping.setdefault(key, row.split)
            if prior != row.split:
                raise ValueError("split-group or target-cluster leakage")
    return {
        "rows": len(row_list),
        "replay_rate": 1.0 if row_list else 0.0,
        "candidate_reconstruction_rate": 1.0 if row_list else 0.0,
        "split_safe": True,
        "forbidden_model_input_fields": 0,
    }


def write_corpus(
    output: Path,
    rows: Iterable[LegalEditBridgeRowV1],
    candidate_sets: Mapping[str, ExactLegalEditCandidateSetV1],
    manifest: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = output / "candidate_sets"
    candidate_dir.mkdir(exist_ok=True)
    state_dir = output / "states"
    state_dir.mkdir(exist_ok=True)
    for digest, candidate_set in sorted(candidate_sets.items()):
        (candidate_dir / f"{digest}.json").write_text(
            json.dumps(candidate_set.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if candidate_set.candidates:
            state = str(
                candidate_set.candidates[0].transition_certificate["source_program"]
            )
            state_digest = content_digest(state)
            (state_dir / f"{state_digest}.json").write_text(
                json.dumps(
                    {
                        "schema": "ExactLegalEditStateV1",
                        "state_fingerprint": candidate_set.state_fingerprint,
                        "program": state,
                        "content_digest": state_digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    row_list = sorted(rows, key=lambda row: row.row_id)
    (output / "bridge_rows.jsonl").write_text(
        "".join(canonical_json(row.to_dict()) + "\n" for row in row_list),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_corpus(
    output: Path,
) -> tuple[list[LegalEditBridgeRowV1], dict[str, ExactLegalEditCandidateSetV1], dict[str, Any]]:
    rows = [
        LegalEditBridgeRowV1.from_dict(json.loads(line))
        for line in (output / "bridge_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_sets: dict[str, ExactLegalEditCandidateSetV1] = {}
    for path in sorted((output / "candidate_sets").glob("*.json")):
        candidate_set = ExactLegalEditCandidateSetV1.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if path.stem != candidate_set.candidate_set_digest:
            raise ValueError("content-addressed candidate-set filename mismatch")
        candidate_sets[path.stem] = candidate_set
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for row in rows:
        state_payload = json.loads(
            (output / row.exact_state_ref).read_text(encoding="utf-8")
        )
        if content_digest(state_payload["program"]) != state_payload["content_digest"]:
            raise ValueError("content-addressed exact-state digest mismatch")
        if canonical_fingerprint(state_payload["program"]) != row.state_fingerprint:
            raise ValueError("exact-state reconstruction mismatch")
    validate_rows(rows, candidate_sets)
    return rows, candidate_sets, manifest
