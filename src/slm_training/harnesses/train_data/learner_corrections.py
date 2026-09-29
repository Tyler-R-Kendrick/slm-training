"""Training-only corrective proposals for explicitly known symbolic copy tasks.

This oracle's input IS a program (AST-to-AST ladder rung), not an arbitrary NL
instruction. Parser success cannot establish the intended answer. Publication
still belongs to DATA's immutable readiness/admission workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from slm_training.data.readiness_lineage import leakage_findings
from slm_training.data.record_admission import assert_training_record
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.measurement_identity import content_digest, declared_root


def copy_fixture_records():
    from slm_training.dsl.schema import ExampleRecord

    programs = (
        'root = TextContent(":slot_0")',
        'root = Button(":slot_0")',
        'root = Card([TextContent(":slot_0")])',
        'root = Stack([Button(":slot_0")], "row")',
    )
    rows = []
    for index, program in enumerate(programs):
        oracle = CopyProgramOracle(program, f"copy-family-{index}")
        rows.append(
            ExampleRecord(
                id=f"copy-{index}",
                prompt=oracle.prompt,
                openui=program,
                placeholders=[":slot_0"],
                split="train" if index < 2 else "smoke",
                source="declared_public_AST_copy_fixture",
                meta={"root_family_id": oracle.root_id},
            )
        )
    return rows[:2], rows[2:]


def publish_fixture(root, name, rows, kind, *, write_json, feedback=None):
    from slm_training.data.publication import publish_tree
    from slm_training.data.record_admission import assert_training_record
    from slm_training.data.store import DataStore, write_common_manifest
    from slm_training.dsl.schema import write_jsonl

    staging = root / "staging" / name
    staging.mkdir(parents=True)
    if kind == "train":
        for row in rows:
            assert_training_record(row)
    records_path = staging / (
        "records.jsonl" if kind == "train" else "suites/smoke/records.jsonl"
    )
    records_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records_path, rows)
    write_json(
        staging / "manifest.json",
        {
            "kind": kind,
            "dataset_id": name,
            "exposure": "public_regression",
            "sampling_policy": "fixed_AST_copy_fixture/v1",
        },
    )
    if kind == "train":
        write_json(
            staging / "quality_report.json",
            {
                "record_n": len(rows),
                "admission": "actual_training_record_contract",
                "warnings": [],
                "per_root_n": {
                    r: sum(row.meta["root_family_id"] == r for row in rows)
                    for r in sorted({row.meta["root_family_id"] for row in rows})
                },
            },
        )
        write_json(
            staging / "synthesis_feedback.json",
            feedback or {"recommendations": [], "experiment_candidates": []},
        )
        (staging / "rejected.jsonl").write_text("")
    write_common_manifest(staging, kind=kind, dataset_id=name)
    destination = DataStore().path(kind, name)
    publish_tree(staging, destination, metadata={"kind": kind, "dataset_id": name})
    return destination.resolve()


@dataclass(frozen=True)
class LearnerState:
    canvas: tuple[int, ...]
    unknown: tuple[bool, ...]
    checkpoint_sha256: str
    tokenizer_sha256: str
    seed: int

    def __post_init__(self):
        if not self.canvas or len(self.canvas) != len(self.unknown):
            raise ValueError("full diffusion canvas and unknown mask must align")
        if any(type(token) is not int or token < 0 for token in self.canvas):
            raise ValueError("canvas must contain token IDs")
        if any(type(value) is not bool for value in self.unknown):
            raise ValueError("unknown mask must contain booleans")
        for value in (self.checkpoint_sha256, self.tokenizer_sha256):
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError("checkpoint/tokenizer content digest required")

    def to_dict(self):
        return {
            "canvas": list(self.canvas),
            "unknown": list(self.unknown),
            "checkpoint_sha256": self.checkpoint_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "seed": self.seed,
            "state_kind": "nonprefix_diffusion_canvas/v1",
        }


@dataclass(frozen=True)
class CopyProgramOracle:
    """A deterministic task definition, not a parser pretending to know intent."""

    program: str
    root_id: str

    @property
    def prompt(self):
        return "Copy this symbolic OpenUI program exactly:\n" + self.program

    def target(self, record):
        if record.prompt != self.prompt or record.openui != self.program:
            raise ValueError("oracle abstains: input is not its declared copy task")
        if declared_root(record) != self.root_id:
            raise ValueError("oracle root does not match training task")
        return self.program


def corrective_example(
    record: ExampleRecord,
    state: LearnerState,
    prediction: str,
    oracle: CopyProgramOracle,
):
    if record.split != "train":
        raise ValueError("learner corrections require training-only inputs")
    target = oracle.target(record)
    assert_training_record(record)
    if prediction == target:
        return None
    from slm_training.dsl.lang_core import validate

    validate(prediction)  # Require a legal-but-wrong learner result, not intent proof.
    payload = state.to_dict()
    result = replace(
        record,
        id=record.id + "__correction_" + content_digest(payload)[:16],
        prompt=record.prompt
        + "\nLearner diffusion state:\n"
        + json.dumps(
            {key: payload[key] for key in ("canvas", "unknown", "state_kind")},
            sort_keys=True,
        ),
        source="learner_state_correction",
        meta={
            **record.meta,
            "learner_correction": {
                "oracle": "copy_program/v1",
                "oracle_input_sha256": content_digest(record.prompt),
                "source_record_sha256": content_digest(record.to_dict()),
                "prediction_sha256": content_digest(prediction),
                "state": payload,
                "claim": "training_only_AST_copy_not_OpenUI_intent",
                "label_status": "verified_task",
            },
        },
    )
    assert_training_record(result)
    return result


def corrective_mixture(original, observations, *, scored_suites, training_ancestors):
    """Keep all replay data; return candidates plus explicit admission/refusal.

    observations = (record, full-state, prediction, registered copy oracle).
    The caller supplies controller-resolved scored suites and ancestor records.
    DATA must independently revalidate this closure before publication.
    """
    if not original or not scored_suites:
        raise ValueError("original replay and actual scored suites are required")
    accepted, refused = [], []
    seen = set()
    original_ids = {record.id for record in original}
    for record, state, prediction, oracle in observations:
        try:
            if record.id not in original_ids:
                raise ValueError("learner source absent from declared training replay")
            candidate = corrective_example(record, state, prediction, oracle)
            if candidate is None:
                refused.append({"id": record.id, "reason": "already_correct"})
                continue
            digest = content_digest(candidate.to_dict())
            if digest in seen:
                refused.append({"id": record.id, "reason": "duplicate_state"})
                continue
            seen.add(digest)
            accepted.append(candidate)
        except ValueError as exc:
            refused.append({"id": record.id, "reason": str(exc)})
    leakage = leakage_findings(
        [*original, *training_ancestors, *accepted], scored_suites
    )
    if leakage:
        raise ValueError("corrective training/ancestor leakage: " + json.dumps(leakage))
    return [*original, *accepted], {
        "original_n": len(original),
        "corrective_n": len(accepted),
        "refused": refused,
        "leakage": leakage,
        "publication_authorized": False,
        "per_root_corrective_n": {
            root: sum(declared_root(row) == root for row in accepted)
            for root in sorted({declared_root(row) for row in accepted})
        },
    }
