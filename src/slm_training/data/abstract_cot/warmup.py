"""AP-018 (SLM-306): bottleneck warm-up records from SemanticPlanV1 and verbal plans.

Decision (Linear DSH-adjacent AP2 issue AP-018): create *privileged-plan*
warm-up examples for the Abstract-CoT causal pilot while forcing the eventual
program-generation head to depend only on the prompt plus abstract codebook
tokens -- never directly on the privileged plan. This module only *builds the
data*; it does not train anything and does not change any production decode
path. The block-structured bottleneck attention that actually enforces the
``target`` segment's exclusion of ``privileged_plan`` is AP-019's job -- this
module emits the typed contract (:class:`SegmentLayoutV1`) that AP-019 must
honor, so the constraint is documented and replayable from iteration zero.

Pipeline, per already-verified ``ExampleRecord`` row:

1. Parse the record's OpenUI program into a ``ProgramSpec`` and extract its
   canonical, gold-provenance ``SemanticPlanV1`` (existing
   ``OpenUISemanticPlanExtractor`` -- CAP2/AP-029 authority, not reinvented
   here).
2. Derive a fixed-width factor tensorization of that plan (existing CAP2-05 /
   AP-029 ``tensorize_semantic_plan`` bridge) plus an optional, fully
   deterministic natural-language verbalization computed from the same
   canonicalized plan.
3. Seed an iteration-0 *abstract span*: a short sequence of reserved,
   non-interpretable ``AbstractPlanV1`` codebook tokens sampled uniformly at
   random from a seed keyed only by the record id and a fixed global seed --
   never derived from the plan or the target, so no gold information leaks
   into the abstract span at this iteration. AP-020 (on-policy
   self-distillation) is the only issue authorized to replace this random
   seed with a model-produced span.
4. Emit ``[prompt; privileged_plan; abstract_span; target]`` records with
   explicit segment/attention/loss metadata (:class:`SegmentLayoutV1`).

Locked-test exclusion: any source row whose id or gold-plan fingerprint
appears in the ``locked_test`` partition of
``src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl``
(the existing decontaminated promotion holdout, see
``slm_training.data.locked_eval_manifest``) is excluded with a typed reason
rather than silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.canonicalize import canonicalize_plan, plan_factor_fingerprints
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.dsl.abstract_plan import AbstractPlanV1
from slm_training.dsl.pack import get_pack
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.semantic_plan_factors import (
    ProgramFactorTensorizerConfig,
    tensorize_semantic_plan,
)

__all__ = [
    "SCHEMA",
    "SEGMENT_ORDER",
    "DEFAULT_ATTENTION_BLOCK",
    "DEFAULT_LOSS_SEGMENTS",
    "EXCLUSION_REASONS",
    "SegmentLayoutV1",
    "AbstractCotWarmupConfig",
    "AbstractCotWarmupRecordV1",
    "WarmupExclusionV1",
    "AbstractCotWarmupCorpusV1",
    "verbalize_semantic_plan",
    "build_abstract_cot_warmup_corpus",
    "load_locked_test_keys",
    "DEFAULT_LOCKED_MANIFEST_PATH",
]

SCHEMA = "abstract_cot_warmup/v1"

# ``[prompt; privileged_plan; abstract_span; target]`` -- the AP-018 record
# layout named directly in the issue's Implementation contract.
SEGMENT_ORDER: tuple[str, ...] = ("prompt", "privileged_plan", "abstract_span", "target")

# Bottleneck attention contract: which segments each segment may attend to.
# ``target`` deliberately excludes ``privileged_plan`` -- this is the exact
# mechanism that forces the final program to depend only on the prompt plus
# abstract tokens (the issue's Decision). AP-019 implements the real masked
# attention/loss; this is the frozen, replayable contract it must honor.
DEFAULT_ATTENTION_BLOCK: dict[str, tuple[str, ...]] = {
    "prompt": ("prompt",),
    "privileged_plan": ("prompt", "privileged_plan"),
    "abstract_span": ("prompt", "privileged_plan", "abstract_span"),
    "target": ("prompt", "abstract_span", "target"),
}
DEFAULT_LOSS_SEGMENTS: tuple[str, ...] = ("target",)

EXCLUSION_REASONS = frozenset(
    {
        "locked_test_excluded",
        "empty_prompt_or_openui",
        "openui_parse_failed",
        "plan_extraction_failed",
    }
)

DEFAULT_LOCKED_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "data"
    / "eval"
    / "manifests"
    / "abstract_planning_locked_v1.jsonl"
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SegmentLayoutV1:
    """Typed segment/attention/loss metadata for one warm-up record.

    This is the contract AP-019's block-structured bottleneck attention must
    replay exactly: ``attention_block[segment]`` lists every segment that
    ``segment`` may attend to (inclusive of itself), and ``loss_segments``
    lists which segments receive a training loss at this iteration.
    """

    order: tuple[str, ...] = SEGMENT_ORDER
    attention_block: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_ATTENTION_BLOCK)
    )
    loss_segments: tuple[str, ...] = DEFAULT_LOSS_SEGMENTS

    def __post_init__(self) -> None:
        if tuple(self.order) != SEGMENT_ORDER:
            raise ValueError(f"unsupported segment order: {self.order!r}")
        if set(self.attention_block) != set(SEGMENT_ORDER):
            raise ValueError("attention_block must cover exactly the four segments")
        for segment, allowed in self.attention_block.items():
            if segment not in allowed:
                raise ValueError(f"segment {segment!r} must attend to itself")
            unknown = set(allowed) - set(SEGMENT_ORDER)
            if unknown:
                raise ValueError(f"attention_block[{segment!r}] has unknown segments: {unknown}")
        if "privileged_plan" in self.attention_block.get("target", ()):
            raise ValueError(
                "target segment must never attend to privileged_plan -- that is "
                "the AP-018 bottleneck invariant this schema exists to enforce"
            )
        unknown_loss = set(self.loss_segments) - set(SEGMENT_ORDER)
        if unknown_loss:
            raise ValueError(f"loss_segments has unknown segments: {unknown_loss}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "attention_block": {k: list(v) for k, v in self.attention_block.items()},
            "loss_segments": list(self.loss_segments),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SegmentLayoutV1":
        return cls(
            order=tuple(value.get("order", SEGMENT_ORDER)),
            attention_block={
                str(k): tuple(v) for k, v in dict(value.get("attention_block", {})).items()
            }
            or dict(DEFAULT_ATTENTION_BLOCK),
            loss_segments=tuple(value.get("loss_segments", DEFAULT_LOSS_SEGMENTS)),
        )


@dataclass(frozen=True)
class AbstractCotWarmupConfig:
    """Fixed, deterministic knobs for the AP-018 warm-up record builder."""

    abstract_plan: AbstractPlanV1 = field(default_factory=AbstractPlanV1)
    abstract_span_length: int = 3  # matches AbstractPlanV1's default T (rounds)
    seed: int = 0
    include_verbalization: bool = True
    factor_config: ProgramFactorTensorizerConfig = field(
        default_factory=ProgramFactorTensorizerConfig
    )
    pack_id: str = "openui"
    locked_manifest_path: Path | None = DEFAULT_LOCKED_MANIFEST_PATH
    max_records: int | None = None
    segments: SegmentLayoutV1 = field(default_factory=SegmentLayoutV1)

    def __post_init__(self) -> None:
        if self.abstract_span_length < 1:
            raise ValueError("abstract_span_length must be >= 1")
        if self.abstract_span_length > self.abstract_plan.max_slot_count:
            raise ValueError("abstract_span_length must not exceed max_slot_count")
        if self.max_records is not None and self.max_records < 1:
            raise ValueError("max_records must be >= 1 when set")


@dataclass(frozen=True)
class AbstractCotWarmupRecordV1:
    """One ``[prompt; privileged_plan; abstract_span; target]`` warm-up row."""

    schema: str
    record_id: str
    source_record_id: str
    prompt: str
    target: str
    privileged_plan_factors: dict[str, list[float]]
    privileged_plan_verbalization: str | None
    plan_fingerprint: str
    abstract_span_tokens: tuple[str, ...]
    abstract_span_token_ids: tuple[int, ...]
    abstract_span_source: str
    codebook_version: int
    segments: SegmentLayoutV1
    binder_reference_bucket: str
    source_content_hash: str
    verifier_hash: str
    iteration: int = 0
    split: str = "train"

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported AbstractCotWarmupRecordV1 schema: {self.schema!r}")
        # Only iteration-0 random seeding is authorized by this issue; AP-020
        # is the only issue permitted to introduce an on-policy source.
        if self.iteration == 0 and self.abstract_span_source != "iteration_0_random":
            raise ValueError(
                "iteration 0 records must use abstract_span_source='iteration_0_random'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "record_id": self.record_id,
            "source_record_id": self.source_record_id,
            "prompt": self.prompt,
            "target": self.target,
            "privileged_plan_factors": self.privileged_plan_factors,
            "privileged_plan_verbalization": self.privileged_plan_verbalization,
            "plan_fingerprint": self.plan_fingerprint,
            "abstract_span_tokens": list(self.abstract_span_tokens),
            "abstract_span_token_ids": list(self.abstract_span_token_ids),
            "abstract_span_source": self.abstract_span_source,
            "codebook_version": self.codebook_version,
            "segments": self.segments.to_dict(),
            "binder_reference_bucket": self.binder_reference_bucket,
            "source_content_hash": self.source_content_hash,
            "verifier_hash": self.verifier_hash,
            "iteration": self.iteration,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AbstractCotWarmupRecordV1":
        if str(value.get("schema")) != SCHEMA:
            raise ValueError("unsupported AbstractCotWarmupRecordV1 schema")
        return cls(
            schema=str(value["schema"]),
            record_id=str(value["record_id"]),
            source_record_id=str(value["source_record_id"]),
            prompt=str(value["prompt"]),
            target=str(value["target"]),
            privileged_plan_factors={
                str(k): [float(x) for x in v]
                for k, v in dict(value["privileged_plan_factors"]).items()
            },
            privileged_plan_verbalization=(
                str(value["privileged_plan_verbalization"])
                if value.get("privileged_plan_verbalization") is not None
                else None
            ),
            plan_fingerprint=str(value["plan_fingerprint"]),
            abstract_span_tokens=tuple(value["abstract_span_tokens"]),
            abstract_span_token_ids=tuple(int(x) for x in value["abstract_span_token_ids"]),
            abstract_span_source=str(value["abstract_span_source"]),
            codebook_version=int(value["codebook_version"]),
            segments=SegmentLayoutV1.from_dict(value["segments"]),
            binder_reference_bucket=str(value["binder_reference_bucket"]),
            source_content_hash=str(value["source_content_hash"]),
            verifier_hash=str(value["verifier_hash"]),
            iteration=int(value.get("iteration", 0)),
            split=str(value.get("split", "train")),
        )


@dataclass(frozen=True)
class WarmupExclusionV1:
    """One typed, non-silent exclusion of a source row from the warm-up corpus."""

    source_record_id: str
    reason: str

    def __post_init__(self) -> None:
        if self.reason not in EXCLUSION_REASONS:
            raise ValueError(f"unknown exclusion reason: {self.reason!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"source_record_id": self.source_record_id, "reason": self.reason}


@dataclass(frozen=True)
class AbstractCotWarmupCorpusV1:
    """Full builder result: accepted records plus a non-silent exclusion ledger."""

    schema: str
    codebook_version: int
    records: tuple[AbstractCotWarmupRecordV1, ...]
    exclusions: tuple[WarmupExclusionV1, ...]
    binder_bucket_counts: dict[str, int]
    eligible_count: int
    accepted_count: int
    acceptance_rate: float
    corpus_fingerprint: str
    emitted_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "codebook_version": self.codebook_version,
            "eligible_count": self.eligible_count,
            "accepted_count": self.accepted_count,
            "emitted_count": self.emitted_count,
            "acceptance_rate": self.acceptance_rate,
            "binder_bucket_counts": self.binder_bucket_counts,
            "corpus_fingerprint": self.corpus_fingerprint,
            "exclusions": [item.to_dict() for item in self.exclusions],
        }

    def write_jsonl(self, path: Path) -> None:
        """Write only the emitted records (see ``records``), one per line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(_canonical_json(record.to_dict()))
                handle.write("\n")


def verbalize_semantic_plan(plan: SemanticPlanV1) -> str:
    """Deterministic, template-based verbalization of a canonical plan.

    This is not an LLM paraphrase: identical canonical structure always
    produces byte-identical text, which is required for the warm-up
    iteration-0 contract (no nondeterministic gold-adjacent text).
    """
    canonical = canonicalize_plan(plan)
    archetype = canonical.archetype.id or "unspecified archetype"
    ordered_roles = sorted(canonical.role_slots, key=lambda slot: slot.role_id)
    if ordered_roles:
        role_desc = ", ".join(
            f"{slot.component_family or 'component'}"
            f"[{slot.min_cardinality if slot.min_cardinality is not None else 0}"
            f"-{slot.max_cardinality if slot.max_cardinality is not None else (slot.min_cardinality or 0)}]"
            for slot in ordered_roles
        )
    else:
        role_desc = "no declared roles"
    depth = canonical.topology.depth_bounds
    depth_desc = f"depth {depth[0]}-{depth[1]}" if depth is not None else "depth unbounded"
    binder_desc = f"{len(canonical.bindings)} binding(s) over {len(canonical.symbols)} symbol(s)"
    return (
        f"Archetype: {archetype}. Roles: {role_desc}. Topology: {depth_desc}. "
        f"Binders: {binder_desc}."
    )


def _binder_reference_bucket(plan: SemanticPlanV1) -> str:
    count = len(plan.bindings)
    if count >= 3:
        return "3+"
    return str(count)


def _iteration0_abstract_span(
    record_id: str, config: AbstractCotWarmupConfig
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Sample an iteration-0 abstract span uniformly from a record-keyed seed.

    Deterministic and reproducible regardless of processing order: the RNG is
    seeded only from ``(config.seed, record_id)``, never from plan/target
    content, so no gold information leaks into the span at this iteration.
    """
    plan = config.abstract_plan
    rng = random.Random(f"{config.seed}:{record_id}")
    slot_tokens = plan.slot_tokens
    slot_ids = plan.slot_token_ids
    indices = [rng.randrange(plan.slot_count) for _ in range(config.abstract_span_length)]
    begin_id, end_id = plan.delimiter_token_ids
    tokens = (plan.begin_token, *(slot_tokens[i] for i in indices), plan.end_token)
    ids = (begin_id, *(slot_ids[i] for i in indices), end_id)
    return tokens, ids


def load_locked_test_keys(path: Path | None) -> tuple[set[str], set[str]]:
    """Return (locked_test record ids, locked_test gold-plan fingerprints).

    Returns two empty sets when ``path`` is ``None`` or missing, rather than
    raising -- the locked manifest is an optional decontamination input, not
    a hard runtime dependency of this data builder.
    """
    if path is None or not path.exists():
        return set(), set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    ids: set[str] = set()
    fingerprints: set[str] = set()
    pack = get_pack("openui")
    for row in rows:
        if not isinstance(row, dict) or row.get("partition") != "locked_test":
            continue
        record = row.get("record") or {}
        record_id = record.get("id")
        if record_id:
            ids.add(str(record_id))
        openui = record.get("openui")
        if not openui:
            continue
        try:
            spec = ProgramSpec.from_openui(
                id=str(record_id or "locked"),
                openui=str(openui),
                facts={},
                program_family_id="abstract_planning_locked_v1",
                lineage_id=str(record_id or "locked"),
                split_group_id=str(record_id or "locked"),
            )
            plan = OpenUISemanticPlanExtractor().extract(spec, pack)
            fingerprints.add(plan_factor_fingerprints(plan)["exact"])
        except ValueError:
            continue
    return ids, fingerprints


class _OpenUiParseFailed(ValueError):
    """Raised when ``ProgramSpec.from_openui`` rejects a source row's program."""


class _PlanExtractionFailed(ValueError):
    """Raised when the gold-plan extractor rejects an already-parsed program."""


def _parse_source(source: ExampleRecord) -> ProgramSpec:
    try:
        return ProgramSpec.from_openui(
            id=source.id,
            openui=source.openui,
            facts={},
            program_family_id=str((source.meta or {}).get("program_family_id") or source.source),
            lineage_id=str((source.meta or {}).get("lineage_id") or source.id),
            split_group_id=str((source.meta or {}).get("split_group_id") or source.id),
            split=source.split,
        )
    except ValueError as exc:
        raise _OpenUiParseFailed(str(exc)) from exc


def _extract_plan(spec: ProgramSpec, pack: Any) -> SemanticPlanV1:
    try:
        return OpenUISemanticPlanExtractor().extract(spec, pack)
    except ValueError as exc:
        raise _PlanExtractionFailed(str(exc)) from exc


def _build_one_record(
    source: ExampleRecord,
    config: AbstractCotWarmupConfig,
    pack: Any,
) -> AbstractCotWarmupRecordV1:
    spec = _parse_source(source)
    plan = _extract_plan(spec, pack)
    factors = tensorize_semantic_plan(plan, config.factor_config)
    factor_lists = {name: [float(v) for v in tensor.tolist()] for name, tensor in factors.items()}
    fingerprints = plan_factor_fingerprints(plan)
    verbalization = verbalize_semantic_plan(plan) if config.include_verbalization else None
    tokens, token_ids = _iteration0_abstract_span(source.id, config)
    source_content_hash = _sha256(
        {
            "id": source.id,
            "prompt": source.prompt,
            "openui": source.openui,
            "placeholders": list(source.placeholders),
        }
    )
    verifier_hash = _sha256(
        {
            "source_content_hash": source_content_hash,
            "plan_fingerprint": fingerprints["exact"],
            "contract_id": spec.contract_id,
        }
    )
    return AbstractCotWarmupRecordV1(
        schema=SCHEMA,
        record_id=f"apcw_v1_{source.id}",
        source_record_id=source.id,
        prompt=source.prompt,
        target=source.openui,
        privileged_plan_factors=factor_lists,
        privileged_plan_verbalization=verbalization,
        plan_fingerprint=fingerprints["exact"],
        abstract_span_tokens=tokens,
        abstract_span_token_ids=token_ids,
        abstract_span_source="iteration_0_random",
        codebook_version=config.abstract_plan.codebook_version,
        segments=config.segments,
        binder_reference_bucket=_binder_reference_bucket(plan),
        source_content_hash=source_content_hash,
        verifier_hash=verifier_hash,
        iteration=0,
        split=source.split,
    )


def build_abstract_cot_warmup_corpus(
    records: Iterable[ExampleRecord],
    config: AbstractCotWarmupConfig | None = None,
) -> AbstractCotWarmupCorpusV1:
    """Build the full AP-018 warm-up corpus from already-verified rows.

    Every input row is scored into exactly one of: accepted (has a valid
    plan) or excluded-with-typed-reason (:class:`WarmupExclusionV1`) -- no
    row is dropped silently. ``config.max_records`` (if set) bounds only how
    many *accepted* rows are emitted into ``records`` (selected by a
    binder-bucket round-robin so the emitted subset stays balanced across
    binder/reference complexity); ``eligible_count``/``accepted_count``/
    ``acceptance_rate`` are always measured over the full input, not the
    emitted subset.
    """
    cfg = config or AbstractCotWarmupConfig()
    pack = get_pack(cfg.pack_id)
    locked_ids, locked_fingerprints = load_locked_test_keys(cfg.locked_manifest_path)

    accepted: list[AbstractCotWarmupRecordV1] = []
    exclusions: list[WarmupExclusionV1] = []
    eligible_count = 0

    for source in records:
        eligible_count += 1
        if source.id in locked_ids:
            exclusions.append(WarmupExclusionV1(source.id, "locked_test_excluded"))
            continue
        if not source.prompt.strip() or not source.openui.strip():
            exclusions.append(WarmupExclusionV1(source.id, "empty_prompt_or_openui"))
            continue
        try:
            record = _build_one_record(source, cfg, pack)
        except _OpenUiParseFailed:
            exclusions.append(WarmupExclusionV1(source.id, "openui_parse_failed"))
            continue
        except _PlanExtractionFailed:
            exclusions.append(WarmupExclusionV1(source.id, "plan_extraction_failed"))
            continue
        if record.plan_fingerprint in locked_fingerprints:
            exclusions.append(WarmupExclusionV1(source.id, "locked_test_excluded"))
            continue
        accepted.append(record)

    accepted_count = len(accepted)
    acceptance_rate = accepted_count / eligible_count if eligible_count else 0.0

    buckets: dict[str, list[AbstractCotWarmupRecordV1]] = {}
    for record in accepted:
        buckets.setdefault(record.binder_reference_bucket, []).append(record)
    binder_bucket_counts = {key: len(value) for key, value in sorted(buckets.items())}

    if cfg.max_records is None or accepted_count <= cfg.max_records:
        emitted = list(accepted)
    else:
        # Deterministic round-robin across binder buckets (sorted bucket key,
        # then by record_id within a bucket) so the emitted subset stays
        # balanced across binder/reference complexity rather than favoring
        # whatever bucket happened to come first in input order.
        ordered_buckets = [sorted(buckets[key], key=lambda r: r.record_id) for key in sorted(buckets)]
        emitted = []
        cursors = [0] * len(ordered_buckets)
        while len(emitted) < cfg.max_records and any(
            cursor < len(bucket) for cursor, bucket in zip(cursors, ordered_buckets)
        ):
            for idx, bucket in enumerate(ordered_buckets):
                if len(emitted) >= cfg.max_records:
                    break
                if cursors[idx] < len(bucket):
                    emitted.append(bucket[cursors[idx]])
                    cursors[idx] += 1

    corpus_fingerprint = _sha256(
        {
            "codebook_version": cfg.abstract_plan.codebook_version,
            "record_ids": sorted(record.record_id for record in emitted),
            "plan_fingerprints": sorted(record.plan_fingerprint for record in emitted),
        }
    )

    return AbstractCotWarmupCorpusV1(
        schema=SCHEMA,
        codebook_version=cfg.abstract_plan.codebook_version,
        records=tuple(emitted),
        exclusions=tuple(exclusions),
        binder_bucket_counts=binder_bucket_counts,
        eligible_count=eligible_count,
        accepted_count=accepted_count,
        acceptance_rate=acceptance_rate,
        corpus_fingerprint=corpus_fingerprint,
        emitted_count=len(emitted),
    )
