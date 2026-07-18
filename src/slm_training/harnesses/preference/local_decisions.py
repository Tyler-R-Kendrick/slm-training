"""Strict exact-state decision events mined from decode traces."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


Split = Literal["train", "held_out"]
EvidenceKind = Literal["constraint_shadow", "counterfactual"]


def _ids(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def split_for_group(group_id: str) -> Split:
    """Stable 80/20 split; every event from one prompt family stays together."""
    return "held_out" if int(_sha(group_id)[:8], 16) % 5 == 0 else "train"


@dataclass(frozen=True)
class DecisionEventV1:
    event_id: str
    group_id: str
    context_text: str
    canvas_ids: tuple[int, ...]
    position: int
    good_token_ids: tuple[int, ...]
    bad_token_ids: tuple[int, ...]
    legal_token_ids: tuple[int, ...]
    evidence_kind: EvidenceKind
    evidence_confidence: float
    decision_kind: str
    split: Split
    policy_checkpoint_sha: str
    tokenizer_sha: str
    decode_config_hash: str
    seed: int
    trajectory_id: str
    source_suite: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "canvas_ids", tuple(map(int, self.canvas_ids)))
        for field in ("good_token_ids", "bad_token_ids", "legal_token_ids"):
            object.__setattr__(self, field, _ids(getattr(self, field)))
        if self.version != 1:
            raise ValueError("unsupported decision event version")
        if not self.event_id or not self.group_id or not self.context_text:
            raise ValueError("event_id, group_id, and context_text are required")
        if not 0 <= self.position < len(self.canvas_ids):
            raise ValueError("position must address canvas_ids")
        if not self.good_token_ids or not self.bad_token_ids:
            raise ValueError("good_token_ids and bad_token_ids must be non-empty")
        if set(self.good_token_ids) & set(self.bad_token_ids):
            raise ValueError("good and bad token sets must be disjoint")
        if not set(self.good_token_ids).issubset(self.legal_token_ids):
            raise ValueError("good token ids must be verifier-legal")
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1]")
        if self.split != split_for_group(self.group_id):
            raise ValueError("split must be derived from group_id")
        for value in (
            self.policy_checkpoint_sha,
            self.tokenizer_sha,
            self.decode_config_hash,
            self.trajectory_id,
        ):
            if not value:
                raise ValueError("event identity fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionEventV1":
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown decision event fields: {sorted(unknown)}")
        return cls(**value)


def decision_signature_metadata(event: DecisionEventV1) -> dict[str, object]:
    return {
        "decision_kind": event.decision_kind,
        "legal_token_ids": list(event.legal_token_ids),
        "good_token_ids": list(event.good_token_ids),
        "bad_token_ids": list(event.bad_token_ids),
    }


def decision_signature(event: DecisionEventV1) -> str:
    return f"{event.decision_kind}@{_sha(decision_signature_metadata(event))[:12]}"


def decision_support_signature_metadata(
    event: DecisionEventV1,
) -> dict[str, object]:
    """Return grammar state and judged positive, excluding sampled negatives."""
    return {
        "decision_kind": event.decision_kind,
        "legal_token_ids": list(event.legal_token_ids),
        "good_token_ids": list(event.good_token_ids),
    }


def decision_support_signature(event: DecisionEventV1) -> str:
    metadata = decision_support_signature_metadata(event)
    return f"{event.decision_kind}@{_sha(metadata)[:12]}"


def decision_signature_support(
    events: Iterable[DecisionEventV1], *, min_train_support: int = 1
) -> dict[str, Any]:
    if min_train_support < 1:
        raise ValueError("minimum train signature support must be positive")
    rows = list(events)
    counts = {
        split: Counter(
            decision_support_signature(event)
            for event in rows
            if event.split == split
        )
        for split in ("train", "held_out")
    }
    metadata = {
        decision_support_signature(event): decision_support_signature_metadata(event)
        for event in rows
    }
    held = set(counts["held_out"])
    covered = sorted(
        signature
        for signature in held
        if counts["train"][signature] >= min_train_support
    )
    uncovered = sorted(held - set(covered))
    return {
        "minimum_train_support": min_train_support,
        "counts": {
            split: dict(sorted(split_counts.items()))
            for split, split_counts in counts.items()
        },
        "metadata": dict(sorted(metadata.items())),
        "held_out_coverage": {
            "covered": covered,
            "uncovered": uncovered,
            "passed": not uncovered,
        },
    }


def objective_signature_support(
    events: Iterable[DecisionEventV1], *, min_train_support: int = 1
) -> dict[str, Any]:
    """Held-out coverage of the OBJECTIVE signature (which includes sampled negatives).

    Unlike ``decision_signature_support`` (state support, good-only), this keys on
    ``decision_signature`` (good + bad). A corpus can therefore pass state support
    yet fail here when the sampled bad-action set at a held-out state has no train
    support — the E284 blocker (stable state support does not imply objective
    support).
    """
    if min_train_support < 1:
        raise ValueError("minimum train signature support must be positive")
    rows = list(events)
    counts = {
        split: Counter(
            decision_signature(event) for event in rows if event.split == split
        )
        for split in ("train", "held_out")
    }
    held = set(counts["held_out"])
    covered = sorted(
        signature for signature in held if counts["train"][signature] >= min_train_support
    )
    uncovered = sorted(held - set(covered))
    return {
        "minimum_train_support": min_train_support,
        "counts": {
            split: dict(sorted(split_counts.items()))
            for split, split_counts in counts.items()
        },
        "held_out_coverage": {
            "covered": covered,
            "uncovered": uncovered,
            "passed": not uncovered,
        },
    }


def _event(
    trace: dict[str, Any],
    payload: dict[str, Any],
    *,
    evidence_kind: EvidenceKind,
    ordinal: str,
) -> DecisionEventV1:
    meta = trace.get("meta") or {}
    group_id = str(meta.get("record_id") or _sha(str(meta.get("prompt") or "")))
    canvas = tuple(map(int, payload["pre_canvas"]))
    position = int(payload["position"])
    good = _ids(payload["good_token_ids"])
    bad = _ids(payload["bad_token_ids"])
    legal = _ids(payload["legal_token_ids"])
    if evidence_kind == "counterfactual" and not set(bad).issubset(legal):
        raise ValueError("counterfactual token ids must all be verifier-legal")
    identity = {
        "trajectory_id": trace.get("trajectory_id"),
        "ordinal": ordinal,
        "position": position,
        "canvas": canvas,
        "good": good,
        "bad": bad,
        "evidence": evidence_kind,
    }
    return DecisionEventV1(
        event_id=_sha(identity),
        group_id=group_id,
        context_text=str(meta.get("context_text") or ""),
        canvas_ids=canvas,
        position=position,
        good_token_ids=good,
        bad_token_ids=bad,
        legal_token_ids=legal,
        evidence_kind=evidence_kind,
        evidence_confidence=float(payload.get("evidence_confidence", 1.0)),
        decision_kind=str(payload.get("decision_kind") or evidence_kind),
        split=split_for_group(group_id),
        policy_checkpoint_sha=str(meta.get("policy_checkpoint_sha") or ""),
        tokenizer_sha=str(meta.get("tokenizer_sha") or ""),
        decode_config_hash=str(meta.get("decode_config_hash") or ""),
        seed=int(meta.get("seed", 0)),
        trajectory_id=str(trace.get("trajectory_id") or ""),
        source_suite=meta.get("source_suite"),
    )


def events_from_trace(trace: dict[str, Any]) -> list[DecisionEventV1]:
    """Mine exact constraint shadows and explicitly verified counterfactuals."""
    out: list[DecisionEventV1] = []
    raw_events = trace.get("events") or []
    probes = {
        str(item.get("state_hash")): item
        for item in raw_events
        if item.get("kind") == "counterfactual_probe" and item.get("state_hash")
    }
    for step_index, step in enumerate(trace.get("steps") or []):
        for commit_index, commit in enumerate(step.get("commits") or []):
            allowed = _ids(commit.get("allowed_id_set") or ())
            raw = commit.get("raw_id")
            chosen = commit.get("id")
            canvas = commit.get("pre_canvas")
            if not allowed or raw is None or chosen is None or canvas is None:
                continue
            if int(raw) in allowed or int(chosen) not in allowed:
                continue
            out.append(
                _event(
                    trace,
                    {
                        "pre_canvas": canvas,
                        "position": int(commit["t"]),
                        "good_token_ids": [chosen],
                        "bad_token_ids": [raw],
                        "legal_token_ids": allowed,
                        "decision_kind": "constraint_shadow",
                    },
                    evidence_kind="constraint_shadow",
                    ordinal=f"{step_index}:{commit_index}",
                )
            )
    for index, item in enumerate(raw_events):
        if item.get("kind") != "counterfactual_decision":
            continue
        if item.get("same_state_verified") is not True:
            raise ValueError("counterfactual decision lacks same-state verification")
        state_hash = str(item.get("state_hash") or "")
        probe = probes.get(state_hash)
        if probe is None or probe.get("qualified") is not True:
            raise ValueError("counterfactual decision lacks a qualified judge probe")
        from slm_training.harnesses.preference.counterfactuals import (
            SEMANTIC_VERIFIER_V1,
            label_pareto_candidates,
        )

        if (probe.get("verifier") or {}).get("name") != SEMANTIC_VERIFIER_V1:
            raise ValueError("counterfactual decision uses an unknown verifier")
        good, bad = label_pareto_candidates(list(probe.get("candidates") or []))
        shared = ("pre_canvas", "position", "legal_token_ids", "state_source")
        if (
            good != sorted({int(value) for value in item.get("good_token_ids") or []})
            or bad
            != sorted({int(value) for value in item.get("bad_token_ids") or []})
            or any(item.get(field) != probe.get(field) for field in shared)
        ):
            raise ValueError("counterfactual decision does not match judge probe")
        out.append(
            _event(
                trace,
                item,
                evidence_kind="counterfactual",
                ordinal=f"counterfactual:{index}",
            )
        )
    return out


def load_trace_rows(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        paths = sorted(path.rglob("traces.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no sharded trace stores under {path}")
    else:
        paths = [path]
    rows: list[dict[str, Any]] = []
    for trace_path in paths:
        with trace_path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def load_decision_events(path: Path | str) -> list[DecisionEventV1]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [DecisionEventV1.from_dict(json.loads(line)) for line in handle if line.strip()]


def counterfactual_evidence_from_traces(
    traces: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain the qualified judge probes backing mined counterfactual events."""
    rows: list[dict[str, Any]] = []
    for trace in traces:
        events_from_trace(trace)  # Validate probe/decision agreement before export.
        raw_events = trace.get("events") or []
        decision_states = {
            str(item.get("state_hash"))
            for item in raw_events
            if item.get("kind") == "counterfactual_decision"
        }
        meta = trace.get("meta") or {}
        for probe in raw_events:
            state_hash = str(probe.get("state_hash") or "")
            if (
                probe.get("kind") != "counterfactual_probe"
                or probe.get("qualified") is not True
                or state_hash not in decision_states
            ):
                continue
            identity = {
                "trajectory_id": trace.get("trajectory_id"),
                "state_hash": state_hash,
            }
            rows.append(
                {
                    "schema_version": 1,
                    "kind": "counterfactual_judge_evidence",
                    "evidence_id": _sha(identity),
                    "trace_id": trace.get("trace_id"),
                    "trajectory_id": trace.get("trajectory_id"),
                    "record_id": meta.get("record_id"),
                    "context_text": meta.get("context_text"),
                    "probe": probe,
                }
            )
    return sorted(rows, key=lambda row: row["evidence_id"])


def write_decision_events(path: Path | str, events: Iterable[DecisionEventV1]) -> int:
    path = Path(path)
    rows = sorted(events, key=lambda event: event.event_id)
    if len({event.event_id for event in rows}) != len(rows):
        raise ValueError("duplicate decision event ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        for event in rows:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return len(rows)


def write_counterfactual_evidence(
    path: Path | str, rows: Iterable[dict[str, Any]]
) -> int:
    path = Path(path)
    evidence = sorted(rows, key=lambda row: row["evidence_id"])
    if len({row["evidence_id"] for row in evidence}) != len(evidence):
        raise ValueError("duplicate counterfactual evidence ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        for row in evidence:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return len(evidence)


def decision_event_manifest(
    events: Iterable[DecisionEventV1],
    *,
    dataset_id: str,
    records_path: str = "events.jsonl",
    source_trace_ids: Iterable[str] = (),
    source_record_fingerprint: str | None = None,
    source_record_fingerprints: Iterable[str] = (),
    evidence_path: str | None = None,
    evidence_rows: Iterable[dict[str, Any]] = (),
    min_train_signature_support: int = 1,
    require_signature_support: bool = False,
) -> dict[str, Any]:
    """Build an immutable, identity-homogeneous event-corpus manifest."""
    rows = sorted(events, key=lambda event: event.event_id)
    if not rows:
        raise ValueError("decision event corpus must not be empty")
    identities = {
        (
            event.policy_checkpoint_sha,
            event.tokenizer_sha,
            event.decode_config_hash,
        )
        for event in rows
    }
    if len(identities) != 1:
        raise ValueError("decision event corpus mixes policy identities")
    checkpoint_sha, tokenizer_sha, decode_hash = identities.pop()
    split_counts = {
        split: sum(event.split == split for event in rows)
        for split in ("train", "held_out")
    }
    split_groups = {
        split: len({event.group_id for event in rows if event.split == split})
        for split in ("train", "held_out")
    }
    signature_support = decision_signature_support(
        rows, min_train_support=min_train_signature_support
    )
    if require_signature_support and not signature_support["held_out_coverage"][
        "passed"
    ]:
        missing = signature_support["held_out_coverage"]["uncovered"]
        raise ValueError(
            f"decision event corpus lacks train support for {len(missing)} "
            "held-out signatures"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "decision_event_corpus",
        "dataset_id": dataset_id,
        "immutable": True,
        "records": records_path,
        "record_count": len(rows),
        "group_count": len({event.group_id for event in rows}),
        "content_fingerprint": _sha([event.to_dict() for event in rows]),
        "policy_checkpoint_sha": checkpoint_sha,
        "tokenizer_sha": tokenizer_sha,
        "decode_config_hash": decode_hash,
        "evidence_kinds": {
            kind: sum(event.evidence_kind == kind for event in rows)
            for kind in ("constraint_shadow", "counterfactual")
        },
        "splits": split_counts,
        "split_groups": split_groups,
        "set_valued_events": sum(
            len(event.good_token_ids) > 1 or len(event.bad_token_ids) > 1
            for event in rows
        ),
        "source_trace_ids": sorted(set(source_trace_ids)),
        "decision_signature_support": signature_support,
    }
    source_fingerprints = sorted(
        {
            *(
                [source_record_fingerprint]
                if source_record_fingerprint is not None
                else []
            ),
            *(str(value) for value in source_record_fingerprints if value),
        }
    )
    if source_fingerprints:
        payload["source_record_fingerprint"] = (
            source_fingerprints[0]
            if len(source_fingerprints) == 1
            else _sha(source_fingerprints)
        )
        payload["source_record_fingerprints"] = source_fingerprints
    evidence = sorted(evidence_rows, key=lambda row: row["evidence_id"])
    candidates = [
        candidate
        for row in evidence
        for candidate in (row.get("probe") or {}).get("candidates") or ()
    ]
    payload["qualified_judge_summary"] = {
        "probes": len(evidence),
        "candidates": len(candidates),
        "independent_judge_passed": sum(
            (candidate.get("judge") or {}).get("ok") is True
            for candidate in candidates
        ),
        "fully_verified": sum(
            candidate.get("verified") is True for candidate in candidates
        ),
    }
    if evidence_path is not None:
        payload["judge_evidence"] = evidence_path
        payload["judge_evidence_count"] = len(evidence)
        payload["judge_evidence_fingerprint"] = _sha(evidence)
    return payload


def write_decision_event_manifest(path: Path | str, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
# ── V2 state + action evidence contract (SLM-116 / LDI0-02) ─────────────

# Re-exported for consumers that import from this module.
__all__ = [
    "Split",
    "EvidenceKind",
    "split_for_group",
    "DecisionEventV1",
    "decision_signature",
    "decision_signature_metadata",
    "decision_support_signature",
    "decision_support_signature_metadata",
    "decision_signature_support",
    "objective_signature_support",
    "events_from_trace",
    "load_trace_rows",
    "load_decision_events",
    "write_decision_events",
    "counterfactual_evidence_from_traces",
    "write_counterfactual_evidence",
    "decision_event_manifest",
    "write_decision_event_manifest",
    "DecisionStateV2",
    "ActionOutcomeV2",
    "ObjectiveView",
    "DecisionEventV2",
    "compute_state_id",
    "merge_action_evidence",
    "materialize_objective_pareto",
    "materialize_objective_threshold",
    "materialize_objective_single_best_worst",
    "materialize_objective_set_partition",
    "materialize_constraint_shadow",
    "guard_semantic_view",
    "migrate_v1_to_v2",
    "load_decision_events_v2",
    "write_decision_events_v2",
    "load_decision_events_v1_or_v2",
    "decision_event_manifest_v2",
    "write_decision_event_manifest_v2",
    "objective_view_signature",
    "objective_view_support",
    "admit_semantic_corpus",
]


def _canonical_state_hash(
    *,
    architecture: str,
    context_text: str | None,
    context_ids: tuple[int, ...] | None,
    canvas_ids: tuple[int, ...] | None,
    decision_position: int,
    generation_step: int | None,
    legal_action_ids: tuple[int, ...],
    decision_kind: str,
    abstract_state_role: str,
    grammar_state_hash: str,
    policy_checkpoint_sha: str,
    tokenizer_sha: str,
    decode_config_hash: str,
    verifier_bundle_hash: str,
    group_id: str,
) -> str:
    """Stable identity over the exact model state and immutable runtime hashes.

    Excludes sampled labels, rollout outcomes, ordinal file position, and
    candidate order. A reordered or augmented action table must not change this
    value.
    """
    payload: dict[str, Any] = {
        "architecture": architecture,
        "decision_position": int(decision_position),
        "generation_step": generation_step,
        "legal_action_ids": list(legal_action_ids),
        "decision_kind": str(decision_kind),
        "abstract_state_role": str(abstract_state_role),
        "grammar_state_hash": str(grammar_state_hash),
        "policy_checkpoint_sha": str(policy_checkpoint_sha),
        "tokenizer_sha": str(tokenizer_sha),
        "decode_config_hash": str(decode_config_hash),
        "verifier_bundle_hash": str(verifier_bundle_hash),
        "group_id": str(group_id),
    }
    if context_ids is not None:
        payload["context_ids"] = list(context_ids)
    elif context_text is not None:
        payload["context_text"] = str(context_text)
    if canvas_ids is not None:
        payload["canvas_ids"] = list(canvas_ids)
    return _sha(payload)


def compute_state_id(state: dict[str, Any]) -> str:
    """Canonical state_id from a dict payload."""
    return _canonical_state_hash(
        architecture=str(state["architecture"]),
        context_text=state.get("context_text"),
        context_ids=state.get("context_ids"),
        canvas_ids=state.get("canvas_ids"),
        decision_position=int(state["decision_position"]),
        generation_step=state.get("generation_step"),
        legal_action_ids=_ids(state["legal_action_ids"]),
        decision_kind=str(state["decision_kind"]),
        abstract_state_role=str(state.get("abstract_state_role") or ""),
        grammar_state_hash=str(state["grammar_state_hash"]),
        policy_checkpoint_sha=str(state["policy_checkpoint_sha"]),
        tokenizer_sha=str(state["tokenizer_sha"]),
        decode_config_hash=str(state["decode_config_hash"]),
        verifier_bundle_hash=str(state["verifier_bundle_hash"]),
        group_id=str(state["group_id"]),
    )


@dataclass(frozen=True)
class DecisionStateV2:
    state_id: str
    group_id: str
    architecture: Literal["twotower", "causal"]
    context_text: str
    context_ids: tuple[int, ...] | None
    canvas_ids: tuple[int, ...] | None
    decision_position: int
    generation_step: int | None
    legal_action_ids: tuple[int, ...]
    decision_kind: str
    abstract_state_role: str
    grammar_state_hash: str
    policy_checkpoint_sha: str
    tokenizer_sha: str
    decode_config_hash: str
    verifier_bundle_hash: str
    split: Split

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "legal_action_ids", _ids(self.legal_action_ids)
        )
        if self.context_ids is not None:
            object.__setattr__(
                self, "context_ids", tuple(map(int, self.context_ids))
            )
        if self.canvas_ids is not None:
            object.__setattr__(
                self, "canvas_ids", tuple(map(int, self.canvas_ids))
            )
        if self.split != split_for_group(self.group_id):
            raise ValueError("split must be derived from group_id")
        if not self.policy_checkpoint_sha or not self.tokenizer_sha:
            raise ValueError("policy and tokenizer identity hashes are required")
        if not self.decode_config_hash or not self.verifier_bundle_hash:
            raise ValueError("decode and verifier bundle hashes are required")
        canonical = _canonical_state_hash(
            architecture=self.architecture,
            context_text=self.context_text or None,
            context_ids=self.context_ids,
            canvas_ids=self.canvas_ids,
            decision_position=self.decision_position,
            generation_step=self.generation_step,
            legal_action_ids=self.legal_action_ids,
            decision_kind=self.decision_kind,
            abstract_state_role=self.abstract_state_role,
            grammar_state_hash=self.grammar_state_hash,
            policy_checkpoint_sha=self.policy_checkpoint_sha,
            tokenizer_sha=self.tokenizer_sha,
            decode_config_hash=self.decode_config_hash,
            verifier_bundle_hash=self.verifier_bundle_hash,
            group_id=self.group_id,
        )
        if not self.state_id:
            object.__setattr__(self, "state_id", canonical)
        elif self.state_id != canonical:
            raise ValueError(
                f"state_id mismatch: provided {self.state_id} != canonical {canonical}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionOutcomeV2:
    state_id: str
    action_id: int
    legal: bool
    rollout_policy_sha: str
    continuation_seeds: tuple[int, ...]
    outcome_hashes: tuple[str, ...]
    verifier_vectors: tuple[dict[str, str], ...]
    reward_vectors: tuple[dict[str, float], ...]
    mean_value: float | None
    confidence_interval: tuple[float, float] | None
    evidence_ids: tuple[str, ...]
    evidence_confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", int(self.action_id))
        object.__setattr__(
            self,
            "continuation_seeds",
            tuple(map(int, self.continuation_seeds or ())),
        )
        object.__setattr__(
            self, "outcome_hashes", tuple(str(value) for value in self.outcome_hashes or ())
        )
        object.__setattr__(
            self,
            "verifier_vectors",
            tuple(dict(value) for value in self.verifier_vectors or ()),
        )
        object.__setattr__(
            self,
            "reward_vectors",
            tuple(dict(value) for value in self.reward_vectors or ()),
        )
        object.__setattr__(
            self, "evidence_ids", tuple(str(value) for value in self.evidence_ids or ())
        )
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1]")
        if len(self.continuation_seeds) != len(self.outcome_hashes):
            raise ValueError("continuation_seeds and outcome_hashes must match")
        if self.confidence_interval is not None and len(self.confidence_interval) != 2:
            raise ValueError("confidence_interval must be a 2-tuple")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveView:
    good_action_ids: tuple[int, ...]
    bad_action_ids: tuple[int, ...]
    ambiguous_action_ids: tuple[int, ...]
    unobserved_action_ids: tuple[int, ...]
    weights: dict[str, float]
    materializer_id: str
    materializer_config_hash: str
    trainable: bool = True

    def __post_init__(self) -> None:
        for field in ("good_action_ids", "bad_action_ids", "ambiguous_action_ids", "unobserved_action_ids"):
            object.__setattr__(self, field, _ids(getattr(self, field)))
        good = set(self.good_action_ids)
        bad = set(self.bad_action_ids)
        ambiguous = set(self.ambiguous_action_ids)
        if good & bad or good & ambiguous or bad & ambiguous:
            raise ValueError("objective partitions must be disjoint")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionEventV2:
    state: DecisionStateV2
    outcomes: tuple[ActionOutcomeV2, ...]
    evidence_kind: EvidenceKind
    version: int = 2

    def __post_init__(self) -> None:
        if self.version != 2:
            raise ValueError("unsupported decision event version")
        if not self.outcomes:
            raise ValueError("V2 event requires at least one action outcome")
        state_ids = {outcome.state_id for outcome in self.outcomes}
        if len(state_ids) != 1 or self.state.state_id not in state_ids:
            raise ValueError("all outcomes must share the state's state_id")
        legal = set(self.state.legal_action_ids)
        for outcome in self.outcomes:
            is_legal_action = outcome.action_id in legal
            if self.evidence_kind == "counterfactual":
                if not is_legal_action:
                    raise ValueError(
                        f"action {outcome.action_id} is not in the state's legal set"
                    )
                if not outcome.legal:
                    raise ValueError(
                        "counterfactual semantic outcomes must be verifier-legal"
                    )
            elif self.evidence_kind == "constraint_shadow" and not is_legal_action:
                if outcome.legal:
                    raise ValueError(
                        "constraint-shadow outcome outside the legal set must be marked illegal"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "evidence_kind": self.evidence_kind,
            "state": self.state.to_dict(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionEventV2":
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown decision event v2 fields: {sorted(unknown)}")
        state = DecisionStateV2(**value["state"])
        outcomes = tuple(
            ActionOutcomeV2(**outcome) for outcome in value.get("outcomes", ())
        )
        return cls(
            state=state,
            outcomes=outcomes,
            evidence_kind=value["evidence_kind"],
            version=int(value.get("version", 2)),
        )


def merge_action_evidence(
    events: Iterable[DecisionEventV2],
) -> list[DecisionEventV2]:
    """Append-only deduplication of action evidence by content identity.

    Two samples for the same exact state merge into one state table; row order
    does not affect the state identity.
    """
    by_state: dict[str, dict[str, Any]] = {}
    for event in events:
        sid = event.state.state_id
        if sid not in by_state:
            by_state[sid] = {
                "state": event.state,
                "evidence_kind": event.evidence_kind,
                "outcomes": {},
            }
        state_entry = by_state[sid]
        if event.state != state_entry["state"]:
            raise ValueError(f"conflicting state metadata for state_id {sid}")
        if event.evidence_kind != state_entry["evidence_kind"]:
            raise ValueError(
                f"conflicting evidence_kind for state_id {sid}"
            )
        for outcome in event.outcomes:
            key = _sha(outcome.to_dict())
            state_entry["outcomes"][key] = outcome
    return [
        DecisionEventV2(
            state=entry["state"],
            outcomes=tuple(sorted(entry["outcomes"].values(), key=lambda o: o.action_id)),
            evidence_kind=entry["evidence_kind"],
        )
        for entry in by_state.values()
    ]


# ── Materializers ────────────────────────────────────────────────────────


def _average_rewards(
    reward_vectors: tuple[dict[str, float], ...],
) -> dict[str, float]:
    if not reward_vectors:
        return {}
    keys = sorted({key for vector in reward_vectors for key in vector})
    return {
        key: sum(vector.get(key, 0.0) for vector in reward_vectors) / len(reward_vectors)
        for key in keys
    }


def _materializer_config_hash(
    materializer_id: str, config: dict[str, Any]
) -> str:
    return _sha({"materializer_id": materializer_id, "config": config})


def materialize_objective_pareto(
    event: DecisionEventV2,
    *,
    metric_thresholds: dict[str, float],
    require_all_metrics: bool = True,
) -> ObjectiveView:
    """Pareto pass/fail over named verifier metrics.

    An action is good iff it passes every threshold; bad iff it fails at least
    one. Actions with no usable reward evidence are ambiguous.
    """
    config = {"metric_thresholds": dict(metric_thresholds), "require_all_metrics": require_all_metrics}
    materializer_id = "pareto_v1"
    config_hash = _materializer_config_hash(materializer_id, config)
    legal = set(event.state.legal_action_ids)
    observed: dict[int, dict[str, float]] = {}
    for outcome in event.outcomes:
        rewards = _average_rewards(outcome.reward_vectors)
        if rewards:
            observed[outcome.action_id] = rewards
    good: set[int] = set()
    bad: set[int] = set()
    ambiguous: set[int] = set()
    for action_id, rewards in observed.items():
        if require_all_metrics and not all(
            metric in rewards for metric in metric_thresholds
        ):
            ambiguous.add(action_id)
            continue
        if all(
            rewards.get(metric, float("-inf")) >= threshold
            for metric, threshold in metric_thresholds.items()
        ):
            good.add(action_id)
        else:
            bad.add(action_id)
    unobserved = legal - set(observed)
    ambiguous = (ambiguous | (set(observed) - good - bad)) & legal
    return ObjectiveView(
        good_action_ids=tuple(sorted(good)),
        bad_action_ids=tuple(sorted(bad)),
        ambiguous_action_ids=tuple(sorted(ambiguous)),
        unobserved_action_ids=tuple(sorted(unobserved)),
        weights={"uniform": 1.0},
        materializer_id=materializer_id,
        materializer_config_hash=config_hash,
    )


def materialize_objective_threshold(
    event: DecisionEventV2,
    *,
    threshold: float,
    min_confidence_lower: float | None = None,
) -> ObjectiveView:
    """Thresholded scalar value with confidence requirements."""
    config = {"threshold": threshold, "min_confidence_lower": min_confidence_lower}
    materializer_id = "threshold_v1"
    config_hash = _materializer_config_hash(materializer_id, config)
    legal = set(event.state.legal_action_ids)
    good: set[int] = set()
    bad: set[int] = set()
    ambiguous: set[int] = set()
    observed: set[int] = set()
    for outcome in event.outcomes:
        observed.add(outcome.action_id)
        mean = outcome.mean_value
        if mean is None:
            ambiguous.add(outcome.action_id)
            continue
        lower = None
        if outcome.confidence_interval is not None:
            lower = min(outcome.confidence_interval)
        if mean >= threshold:
            if min_confidence_lower is None or (
                lower is not None and lower >= min_confidence_lower
            ):
                good.add(outcome.action_id)
            else:
                ambiguous.add(outcome.action_id)
        else:
            bad.add(outcome.action_id)
    unobserved = legal - observed
    return ObjectiveView(
        good_action_ids=tuple(sorted(good)),
        bad_action_ids=tuple(sorted(bad)),
        ambiguous_action_ids=tuple(sorted(ambiguous)),
        unobserved_action_ids=tuple(sorted(unobserved)),
        weights={"uniform": 1.0},
        materializer_id=materializer_id,
        materializer_config_hash=config_hash,
    )


def materialize_objective_single_best_worst(
    event: DecisionEventV2,
) -> ObjectiveView:
    """Single-best / single-worst control."""
    materializer_id = "single_best_worst_v1"
    config_hash = _materializer_config_hash(materializer_id, {})
    legal = set(event.state.legal_action_ids)
    scored: dict[int, float] = {}
    for outcome in event.outcomes:
        if outcome.mean_value is not None:
            scored[outcome.action_id] = outcome.mean_value
        elif outcome.reward_vectors:
            averaged = _average_rewards(outcome.reward_vectors)
            if averaged:
                scored[outcome.action_id] = sum(averaged.values()) / len(averaged)
    good: set[int] = set()
    bad: set[int] = set()
    ambiguous: set[int] = set()
    if len(scored) >= 2:
        best_value = max(scored.values())
        worst_value = min(scored.values())
        if best_value != worst_value:
            good = {aid for aid, value in scored.items() if value == best_value}
            bad = {aid for aid, value in scored.items() if value == worst_value}
            ambiguous = (set(scored) - good - bad) & legal
        else:
            ambiguous = set(scored) & legal
    elif scored:
        ambiguous = set(scored) & legal
    unobserved = legal - set(scored)
    return ObjectiveView(
        good_action_ids=tuple(sorted(good)),
        bad_action_ids=tuple(sorted(bad)),
        ambiguous_action_ids=tuple(sorted(ambiguous)),
        unobserved_action_ids=tuple(sorted(unobserved)),
        weights={"uniform": 1.0},
        materializer_id=materializer_id,
        materializer_config_hash=config_hash,
    )


def _dominates(
    left: dict[str, float], right: dict[str, float]
) -> bool:
    no_worse = all(left.get(name, 0.0) >= right.get(name, 0.0) for name in set(left) | set(right))
    better = any(left.get(name, 0.0) > right.get(name, 0.0) for name in set(left) | set(right))
    return no_worse and better


def materialize_objective_set_partition(
    event: DecisionEventV2,
    *,
    metric_thresholds: dict[str, float] | None = None,
) -> ObjectiveView:
    """Set-valued good/bad partitions via Pareto frontier."""
    config = {"metric_thresholds": dict(metric_thresholds or {})}
    materializer_id = "set_partition_v1"
    config_hash = _materializer_config_hash(materializer_id, config)
    legal = set(event.state.legal_action_ids)
    scored: dict[int, dict[str, float]] = {}
    for outcome in event.outcomes:
        rewards = _average_rewards(outcome.reward_vectors)
        if rewards:
            scored[outcome.action_id] = rewards
    failed: set[int] = set()
    if metric_thresholds:
        for action_id, rewards in scored.items():
            if any(
                rewards.get(metric, float("-inf")) < threshold
                for metric, threshold in metric_thresholds.items()
            ):
                failed.add(action_id)
    eligible = {aid: rewards for aid, rewards in scored.items() if aid not in failed}
    frontier: set[int] = set()
    for action_id, rewards in eligible.items():
        if not any(
            other_id != action_id and _dominates(other_rewards, rewards)
            for other_id, other_rewards in eligible.items()
        ):
            frontier.add(action_id)
    good = frontier
    bad = failed | {
        action_id
        for action_id, rewards in eligible.items()
        if any(
            frontier_id != action_id and _dominates(frontier_rewards, rewards)
            for frontier_id, frontier_rewards in eligible.items()
            if frontier_id in frontier
        )
    }
    ambiguous = (set(scored) - good - bad) & legal
    unobserved = legal - set(scored)
    return ObjectiveView(
        good_action_ids=tuple(sorted(good)),
        bad_action_ids=tuple(sorted(bad)),
        ambiguous_action_ids=tuple(sorted(ambiguous)),
        unobserved_action_ids=tuple(sorted(unobserved)),
        weights={"uniform": 1.0},
        materializer_id=materializer_id,
        materializer_config_hash=config_hash,
    )


def materialize_constraint_shadow(
    event: DecisionEventV2,
) -> ObjectiveView:
    """Diagnostic view for constraint-shadow evidence; explicitly non-semantic."""
    materializer_id = "constraint_shadow_diagnostic_v1"
    config_hash = _materializer_config_hash(materializer_id, {})
    good = {outcome.action_id for outcome in event.outcomes if outcome.legal}
    bad = {
        outcome.action_id
        for outcome in event.outcomes
        if not outcome.legal and outcome.action_id in event.state.legal_action_ids
    }
    observed = {outcome.action_id for outcome in event.outcomes}
    unobserved = set(event.state.legal_action_ids) - observed
    return ObjectiveView(
        good_action_ids=tuple(sorted(good)),
        bad_action_ids=tuple(sorted(bad)),
        ambiguous_action_ids=(),
        unobserved_action_ids=tuple(sorted(unobserved)),
        weights={"uniform": 1.0},
        materializer_id=materializer_id,
        materializer_config_hash=config_hash,
        trainable=False,
    )


def guard_semantic_view(view: ObjectiveView) -> None:
    """Raise if the view comes from a non-semantic constraint-shadow materializer."""
    if "constraint_shadow" in view.materializer_id:
        raise ValueError(
            "constraint-shadow objective view is diagnostic-only and cannot be "
            "consumed by semantic trainers"
        )


def objective_view_signature(
    event: DecisionEventV2, view: ObjectiveView
) -> str:
    """Objective-support signature for a V2 state + materialized view.

    Includes the materializer identity and the good/bad partition. A corpus can
    therefore pass state support while failing objective support when the sampled
    negatives differ — the E284 blocker.
    """
    return _sha(
        {
            "materializer_id": view.materializer_id,
            "materializer_config_hash": view.materializer_config_hash,
            "decision_kind": event.state.decision_kind,
            "legal_action_ids": list(event.state.legal_action_ids),
            "good_action_ids": list(view.good_action_ids),
            "bad_action_ids": list(view.bad_action_ids),
        }
    )[:16]


def objective_view_support(
    items: Iterable[tuple[DecisionEventV2, ObjectiveView]], *, min_train_support: int = 1
) -> dict[str, Any]:
    """Held-out V2 objective-view coverage.

    ``items`` pairs each V2 state with the objective view materialized for it. A
    held-out objective signature is covered when at least ``min_train_support``
    train examples share it.
    """
    if min_train_support < 1:
        raise ValueError("minimum train signature support must be positive")
    rows = list(items)
    counts: dict[str, dict[str, int]] = {"train": {}, "held_out": {}}
    metadata: dict[str, dict[str, Any]] = {}
    for event, view in rows:
        signature = objective_view_signature(event, view)
        bucket = counts[event.state.split]
        bucket[signature] = bucket.get(signature, 0) + 1
        metadata[signature] = {
            "materializer_id": view.materializer_id,
            "materializer_config_hash": view.materializer_config_hash,
            "decision_kind": event.state.decision_kind,
            "legal_action_ids": list(event.state.legal_action_ids),
            "good_action_ids": list(view.good_action_ids),
            "bad_action_ids": list(view.bad_action_ids),
        }
    held = set(counts["held_out"])
    covered = sorted(
        signature for signature in held if counts["train"].get(signature, 0) >= min_train_support
    )
    uncovered = sorted(held - set(covered))
    return {
        "minimum_train_support": min_train_support,
        "counts": {
            split: dict(sorted(split_counts.items())) for split, split_counts in counts.items()
        },
        "metadata": dict(sorted(metadata.items())),
        "held_out_coverage": {
            "covered": covered,
            "uncovered": uncovered,
            "passed": not uncovered,
        },
    }


def admit_semantic_corpus(
    items: Iterable[tuple[DecisionEventV2, ObjectiveView]],
    *,
    materializer_id: str,
    materializer_config_hash: str | None = None,
    min_train_support: int = 1,
) -> dict[str, Any]:
    """Fail closed before semantic training; return the objective-support report.

    Refuses when any view is non-trainable (e.g. a constraint-shadow diagnostic),
    when a view's materializer does not match the requested ID or config hash, or
    when any held-out objective signature lacks train support.
    """
    rows = list(items)
    report = objective_view_support(rows, min_train_support=min_train_support)
    non_trainable = [event.state.state_id for event, view in rows if not view.trainable]
    if non_trainable:
        raise ValueError(
            "semantic admission refused: "
            f"{len(non_trainable)} non-trainable view(s) (e.g. constraint_shadow) "
            "cannot supervise a semantic objective"
        )
    mismatched = sorted(
        {view.materializer_id for event, view in rows if view.materializer_id != materializer_id}
    )
    if mismatched:
        raise ValueError(
            "semantic admission refused: corpus materializer(s) "
            f"{mismatched} do not match the requested objective {materializer_id!r}"
        )
    if materializer_config_hash is not None:
        config_mismatched = sorted(
            {
                view.materializer_config_hash
                for event, view in rows
                if view.materializer_config_hash != materializer_config_hash
            }
        )
        if config_mismatched:
            raise ValueError(
                "semantic admission refused: corpus materializer config hash(es) "
                f"{config_mismatched} do not match the requested config hash "
                f"{materializer_config_hash!r}"
            )
    if not report["held_out_coverage"]["passed"]:
        missing = report["held_out_coverage"]["uncovered"]
        raise ValueError(
            "semantic admission refused: corpus lacks train support for "
            f"{len(missing)} held-out objective signature(s); repair objective "
            "support before training (E284 blocker)"
        )
    return report


def materialize_v1_from_v2(
    event: DecisionEventV2,
    view: ObjectiveView,
    *,
    trajectory_id: str = "v2-materialized",
    seed: int = 0,
) -> DecisionEventV1:
    """Materialize a V1-shaped exact-state event from a V2 state + objective view."""
    guard_semantic_view(view)
    if not view.good_action_ids or not view.bad_action_ids:
        raise ValueError(
            "materialized V1 view requires non-empty good and bad action sets"
        )
    if set(view.good_action_ids) & set(view.bad_action_ids):
        raise ValueError("materialized good and bad actions must be disjoint")
    if not set(view.good_action_ids).issubset(event.state.legal_action_ids):
        raise ValueError("materialized good actions must be verifier-legal")
    identity = {
        "state_id": event.state.state_id,
        "materializer_id": view.materializer_id,
        "materializer_config_hash": view.materializer_config_hash,
    }
    return DecisionEventV1(
        event_id=_sha(identity),
        group_id=event.state.group_id,
        context_text=event.state.context_text,
        canvas_ids=event.state.canvas_ids or (),
        position=event.state.decision_position,
        good_token_ids=view.good_action_ids,
        bad_token_ids=view.bad_action_ids,
        legal_token_ids=event.state.legal_action_ids,
        evidence_kind=event.evidence_kind,
        evidence_confidence=max(
            (outcome.evidence_confidence for outcome in event.outcomes),
            default=1.0,
        ),
        decision_kind=event.state.decision_kind,
        split=event.state.split,
        policy_checkpoint_sha=event.state.policy_checkpoint_sha,
        tokenizer_sha=event.state.tokenizer_sha,
        decode_config_hash=event.state.decode_config_hash,
        seed=seed,
        trajectory_id=trajectory_id,
    )


# ── V1 migration ─────────────────────────────────────────────────────────


def migrate_v1_to_v2(event: DecisionEventV1) -> DecisionEventV2:
    """One-way V1 → V2 migration that never fabricates evidence.

    Semantic counterfactuals become complete-enough action evidence; constraint
    shadows remain incomplete legality diagnostics.
    """
    state = DecisionStateV2(
        state_id="",
        group_id=event.group_id,
        architecture="twotower",
        context_text=event.context_text,
        context_ids=None,
        canvas_ids=event.canvas_ids,
        decision_position=event.position,
        generation_step=None,
        legal_action_ids=event.legal_token_ids,
        decision_kind=event.decision_kind,
        abstract_state_role="",
        grammar_state_hash=decision_signature(event),
        policy_checkpoint_sha=event.policy_checkpoint_sha,
        tokenizer_sha=event.tokenizer_sha,
        decode_config_hash=event.decode_config_hash,
        verifier_bundle_hash=event.decode_config_hash,
        split=event.split,
    )
    outcomes: list[ActionOutcomeV2] = []
    for action_id in event.good_token_ids:
        outcomes.append(
            ActionOutcomeV2(
                state_id=state.state_id,
                action_id=action_id,
                legal=True,
                rollout_policy_sha=event.policy_checkpoint_sha,
                continuation_seeds=(),
                outcome_hashes=(),
                verifier_vectors=(),
                reward_vectors=(),
                mean_value=None,
                confidence_interval=None,
                evidence_ids=(event.event_id,),
                evidence_confidence=event.evidence_confidence,
            )
        )
    for action_id in event.bad_token_ids:
        is_legal = action_id in event.legal_token_ids
        outcomes.append(
            ActionOutcomeV2(
                state_id=state.state_id,
                action_id=action_id,
                legal=is_legal,
                rollout_policy_sha=event.policy_checkpoint_sha,
                continuation_seeds=(),
                outcome_hashes=(),
                verifier_vectors=(),
                reward_vectors=(),
                mean_value=None,
                confidence_interval=None,
                evidence_ids=(event.event_id,),
                evidence_confidence=event.evidence_confidence,
            )
        )
    return DecisionEventV2(
        state=state,
        outcomes=tuple(outcomes),
        evidence_kind=event.evidence_kind,
    )


# ── V2 I/O and manifest ──────────────────────────────────────────────────


def load_decision_events_v2(path: Path | str) -> list[DecisionEventV2]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            DecisionEventV2.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]


def write_decision_events_v2(
    path: Path | str, events: Iterable[DecisionEventV2]
) -> int:
    path = Path(path)
    rows = merge_action_evidence(events)
    if len({event.state.state_id for event in rows}) != len(rows):
        raise ValueError("duplicate state ids after merge")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        for event in sorted(rows, key=lambda event: event.state.state_id):
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return len(rows)


def load_decision_events_v1_or_v2(path: Path | str) -> list[DecisionEventV1 | DecisionEventV2]:
    """Dispatch on the first row's version field."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        return []
    first = json.loads(lines[0])
    version = first.get("version", 1)
    if version == 2:
        return [DecisionEventV2.from_dict(json.loads(line)) for line in lines]
    return [DecisionEventV1.from_dict(json.loads(line)) for line in lines]


def decision_event_manifest_v2(
    events: Iterable[DecisionEventV2],
    *,
    dataset_id: str,
    records_path: str = "events.jsonl",
    source_trace_ids: Iterable[str] = (),
    source_record_fingerprint: str | None = None,
    source_record_fingerprints: Iterable[str] = (),
    evidence_path: str | None = None,
) -> dict[str, Any]:
    """Manifest with separate state, action-evidence, and objective fingerprints."""
    rows = merge_action_evidence(events)
    if not rows:
        raise ValueError("decision event corpus must not be empty")
    identities = {
        (
            event.state.policy_checkpoint_sha,
            event.state.tokenizer_sha,
            event.state.decode_config_hash,
            event.state.verifier_bundle_hash,
        )
        for event in rows
    }
    if len(identities) != 1:
        raise ValueError("decision event corpus mixes policy identities")
    (
        checkpoint_sha,
        tokenizer_sha,
        decode_hash,
        verifier_hash,
    ) = identities.pop()
    split_counts = {
        split: sum(event.state.split == split for event in rows)
        for split in ("train", "held_out")
    }
    split_groups = {
        split: len({event.state.group_id for event in rows if event.state.split == split})
        for split in ("train", "held_out")
    }
    states = [event.state.to_dict() for event in rows]
    # Drop state_id from the state content fingerprint? No: state_id is canonical
    # and derived from the same fields, so including it is harmless and stable.
    state_fingerprint = _sha(states)
    evidence_fingerprint = _sha(
        [outcome.to_dict() for event in rows for outcome in event.outcomes]
    )
    objective_view = [
        materialize_objective_single_best_worst(event).to_dict()
        for event in rows
    ]
    objective_fingerprint = _sha(objective_view)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "decision_event_corpus",
        "dataset_id": dataset_id,
        "immutable": True,
        "records": records_path,
        "record_count": len(rows),
        "group_count": len({event.state.group_id for event in rows}),
        "state_fingerprint": state_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "objective_fingerprint": objective_fingerprint,
        "policy_checkpoint_sha": checkpoint_sha,
        "tokenizer_sha": tokenizer_sha,
        "decode_config_hash": decode_hash,
        "verifier_bundle_hash": verifier_hash,
        "evidence_kinds": {
            kind: sum(event.evidence_kind == kind for event in rows)
            for kind in ("constraint_shadow", "counterfactual")
        },
        "splits": split_counts,
        "split_groups": split_groups,
        "source_trace_ids": sorted(set(source_trace_ids)),
    }
    source_fingerprints = sorted(
        {
            *(
                [source_record_fingerprint]
                if source_record_fingerprint is not None
                else []
            ),
            *(str(value) for value in source_record_fingerprints if value),
        }
    )
    if source_fingerprints:
        payload["source_record_fingerprint"] = (
            source_fingerprints[0]
            if len(source_fingerprints) == 1
            else _sha(source_fingerprints)
        )
        payload["source_record_fingerprints"] = source_fingerprints
    if evidence_path is not None:
        payload["evidence_path"] = evidence_path
    return payload


def write_decision_event_manifest_v2(
    path: Path | str, manifest: dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
