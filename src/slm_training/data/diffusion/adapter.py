"""Pure, online corruption policies for masked-diffusion OpenUI training."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol

POLICIES = (
    "uniform",
    "contiguous",
    "statement",
    "ast_subtree",
    "reference",
    "edit_local",
    "disjoint",
    "all_mask",
    "expansion",
    "contraction",
    "reorder",
)
DEFAULT_LENGTH_BUCKETS = (32, 64, 96, 128, 192, 256, 384, 512)


class TokenizerLike(Protocol):
    pad_id: int
    bos_id: int
    eos_id: int
    mask_id: int
    id_to_token: Mapping[int, str]


@dataclass(frozen=True)
class DiffusionConfig:
    """Policy mixture and canvas limits; no corpus rows are materialized."""

    policies: tuple[str, ...] = POLICIES
    mask_min: float = 0.15
    mask_max: float = 0.85
    overallocate: int = 8
    length_buckets: tuple[int, ...] = DEFAULT_LENGTH_BUCKETS
    max_length: int | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.policies) - set(POLICIES))
        if not self.policies or unknown:
            raise ValueError(f"invalid diffusion policies: {unknown or self.policies}")
        if not 0.0 <= self.mask_min <= self.mask_max <= 1.0:
            raise ValueError("mask rates must satisfy 0 <= min <= max <= 1")
        if self.overallocate < 1:
            raise ValueError("overallocate must be >= 1")
        if (
            not self.length_buckets
            or tuple(sorted(self.length_buckets)) != self.length_buckets
        ):
            raise ValueError("length_buckets must be non-empty and sorted")


@dataclass(frozen=True)
class DiffusionCorruption:
    """One clean target and its in-memory diffusion state."""

    clean_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    noisy_ids: tuple[int, ...]
    predict_mask: tuple[bool, ...]
    insertion_mask: tuple[bool, ...]
    deletion_mask: tuple[bool, ...]
    policy: str
    source_length: int
    target_length: int
    canvas_length: int
    length_bucket: int
    length_bucket_index: int
    pad_id: int
    aux_labels: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {
            len(self.target_ids),
            len(self.noisy_ids),
            len(self.predict_mask),
            len(self.insertion_mask),
            len(self.deletion_mask),
            self.canvas_length,
        }
        if len(lengths) != 1:
            raise ValueError("diffusion canvas fields must align")
        for name, labels in self.aux_labels.items():
            if len(labels) != self.canvas_length:
                raise ValueError(f"aux label {name!r} is not canvas-aligned")

    def reconstruct(
        self, predicted_ids: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        """Apply predictions and remove learned deletion slots."""
        if predicted_ids is not None and len(predicted_ids) != self.canvas_length:
            raise ValueError("predicted_ids must match canvas_length")
        values = list(self.noisy_ids)
        for index, predict in enumerate(self.predict_mask):
            if predict:
                values[index] = (
                    int(predicted_ids[index])
                    if predicted_ids is not None
                    else self.target_ids[index]
                )
        return tuple(value for value in values if value != self.pad_id)


@dataclass(frozen=True)
class DiffusionBatch:
    rows: tuple[DiffusionCorruption, ...]

    @property
    def policies(self) -> tuple[str, ...]:
        return tuple(row.policy for row in self.rows)


def length_bucket(
    length: int, buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS
) -> tuple[int, int]:
    """Return ``(bucket_index, bucket_upper_bound)`` with overflow clamped."""
    if length < 0 or not buckets:
        raise ValueError("length must be >= 0 and buckets must be non-empty")
    for index, upper in enumerate(buckets):
        if length <= upper:
            return index, int(upper)
    return len(buckets) - 1, int(buckets[-1])


def edit_token_indices(meta: Mapping[str, Any] | None) -> tuple[int, ...]:
    """Read the shared ProgramSpec edit metadata without imposing a new schema."""
    edit = (meta or {}).get("edit")
    if not isinstance(edit, Mapping):
        return ()
    raw = edit.get("changed_token_indices")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(sorted({int(value) for value in raw if int(value) >= 0}))
    span = edit.get("changed_span")
    if (
        isinstance(span, Sequence)
        and not isinstance(span, (str, bytes))
        and len(span) == 2
    ):
        lo, hi = int(span[0]), int(span[1])
        return tuple(range(max(0, lo), max(lo, hi)))
    return ()


def _surface(tokenizer: TokenizerLike, token_id: int) -> str:
    return str(tokenizer.id_to_token.get(int(token_id), ""))


def _trim_padding(ids: Sequence[int], pad_id: int) -> list[int]:
    values = [int(value) for value in ids]
    while values and values[-1] == pad_id:
        values.pop()
    return values


def _valid_positions(ids: Sequence[int], tokenizer: TokenizerLike) -> list[int]:
    return [
        index
        for index, value in enumerate(ids)
        if value not in {tokenizer.pad_id, tokenizer.bos_id}
    ]


def _statement_spans(
    ids: Sequence[int], tokenizer: TokenizerLike
) -> list[tuple[int, int]]:
    method = getattr(tokenizer, "statement_spans", None)
    if callable(method):
        spans = [tuple(map(int, span)) for span in method(list(ids))]
        if spans:
            return spans
    start = 1 if ids and ids[0] == tokenizer.bos_id else 0
    spans: list[tuple[int, int]] = []
    for index in range(start, len(ids)):
        value = ids[index]
        if value in {tokenizer.eos_id, tokenizer.pad_id}:
            if index > start:
                spans.append((start, index))
            break
        if "\n" in _surface(tokenizer, value):
            if index > start:
                spans.append((start, index))
            start = index + 1
    else:
        if start < len(ids):
            spans.append((start, len(ids)))
    return spans


def _subtree_spans(
    ids: Sequence[int], tokenizer: TokenizerLike
) -> list[tuple[int, int]]:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    for index, value in enumerate(ids):
        token = _surface(tokenizer, value)
        if token in {"(", "[", "{"}:
            stack.append((token, index))
        elif token in pairs:
            for stack_index in range(len(stack) - 1, -1, -1):
                opener, start = stack[stack_index]
                if opener == pairs[token]:
                    del stack[stack_index:]
                    if index > start + 1:
                        spans.append((start, index + 1))
                    break
    return spans


def _definition_positions(
    ids: Sequence[int], tokenizer: TokenizerLike
) -> dict[int, int]:
    definitions: dict[int, int] = {}
    for lo, hi in _statement_spans(ids, tokenizer):
        for index in range(lo, hi):
            if _surface(tokenizer, ids[index]) == "=":
                if index > lo:
                    definitions[int(ids[index - 1])] = index - 1
                break
    return definitions


def _reference_groups(ids: Sequence[int], tokenizer: TokenizerLike) -> list[list[int]]:
    definitions = _definition_positions(ids, tokenizer)
    groups: list[list[int]] = []
    for token_id, definition in definitions.items():
        positions = [
            index
            for index, value in enumerate(ids)
            if int(value) == token_id
            and value not in {tokenizer.bos_id, tokenizer.eos_id}
        ]
        if definition in positions and len(positions) > 1:
            groups.append(positions)
    return groups


def _contiguous_positions(
    valid: Sequence[int], count: int, rng: random.Random
) -> list[int]:
    if not valid:
        return []
    count = max(1, min(count, len(valid)))
    candidates = [
        start
        for start in range(len(valid) - count + 1)
        if valid[start + count - 1] - valid[start] == count - 1
    ]
    start = rng.choice(candidates) if candidates else 0
    return list(valid[start : start + count])


def _disjoint_positions(
    valid: Sequence[int], count: int, rng: random.Random
) -> list[int]:
    """Choose at least two mask islands when the sequence has room for a gap."""
    if len(valid) < 3:
        return list(valid)
    gap = rng.randrange(1, len(valid) - 1)
    anchors = {valid[gap - 1], valid[gap + 1]}
    candidates = [
        position for position in valid if position not in anchors | {valid[gap]}
    ]
    wanted = min(max(2, count), len(valid) - 1)
    anchors.update(rng.sample(candidates, min(wanted - 2, len(candidates))))
    return sorted(anchors)


def _aux_labels(
    target: Sequence[int],
    tokenizer: TokenizerLike,
    predict: Sequence[bool],
    *,
    target_length: int,
    error_category: str | None,
) -> dict[str, tuple[Any, ...]]:
    n = len(target)
    statement_boundary = [0] * n
    for lo, _ in _statement_spans(target, tokenizer):
        if 0 <= lo < n:
            statement_boundary[lo] = 1
    definitions = _definition_positions(target, tokenizer)
    definition_at = {position: token_id for token_id, position in definitions.items()}
    def_ids = set(definitions)
    node_type: list[str] = []
    production: list[str] = []
    identifier_role: list[str] = []
    component_type: list[str] = []
    arg_position: list[int] = []
    arg_stack: list[int] = []
    for index, value in enumerate(target):
        token = _surface(tokenizer, value)
        if token in {"(", "[", "{"}:
            arg_stack.append(0)
        current_arg = arg_stack[-1] if arg_stack else -1
        arg_position.append(current_arg)
        if index in definition_at:
            role = "definition"
        elif int(value) in def_ids:
            role = "use"
        else:
            role = "none"
        identifier_role.append(role)
        is_component = token[:1].isupper() and token.isidentifier()
        component_type.append(token if is_component else "")
        if value in {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}:
            kind = "special"
        elif is_component:
            kind = "component"
        elif role != "none":
            kind = "identifier"
        elif token.startswith(('"', "STR:", "LIT_")) or token[:1].isdigit():
            kind = "literal"
        else:
            kind = "structure"
        node_type.append(kind)
        if token == "=":
            production.append("assignment")
        elif is_component:
            production.append("call")
        elif token in {"[", "]"}:
            production.append("list")
        else:
            production.append("token")
        if token == "," and arg_stack:
            arg_stack[-1] += 1
        elif token in {")", "]", "}"} and arg_stack:
            arg_stack.pop()
    labels: dict[str, tuple[Any, ...]] = {
        "statement_boundary": tuple(statement_boundary),
        "ast_node_type": tuple(node_type),
        "grammar_production": tuple(production),
        "identifier_role": tuple(identifier_role),
        "component_type": tuple(component_type),
        "arg_position": tuple(arg_position),
        "expected_length": (target_length,) * n,
        "changed": tuple(int(value) for value in predict),
    }
    if error_category is not None:
        labels["error_category"] = (str(error_category),) * n
    return labels


def corrupt_tokens(
    clean_ids: Sequence[int],
    tokenizer: TokenizerLike,
    *,
    policy: str,
    rng: random.Random,
    mask_rate: float = 0.35,
    overallocate: int = 8,
    length_buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
    max_length: int | None = None,
    edit_indices: Sequence[int] = (),
    error_category: str | None = None,
) -> DiffusionCorruption:
    """Corrupt one clean sequence in memory using an explicit policy."""
    if policy not in POLICIES:
        raise ValueError(f"unknown diffusion policy {policy!r}")
    clean = _trim_padding(clean_ids, tokenizer.pad_id)
    if not clean:
        raise ValueError("clean_ids must contain a non-padding token")
    target = list(clean)
    noisy = list(clean)
    predict = [False] * len(clean)
    insertion = [False] * len(clean)
    deletion = [False] * len(clean)
    valid = _valid_positions(clean, tokenizer)
    count = max(1, min(len(valid), math.ceil(len(valid) * max(0.0, mask_rate))))
    source_length = len(clean)

    def mask_positions(positions: Sequence[int], *, inserted: bool = False) -> None:
        for index in positions:
            if 0 <= index < len(noisy) and index in valid:
                noisy[index] = tokenizer.mask_id
                predict[index] = True
                insertion[index] = inserted

    if policy == "uniform":
        mask_positions(rng.sample(valid, count))
    elif policy == "contiguous":
        mask_positions(_contiguous_positions(valid, count, rng))
    elif policy == "statement":
        spans = _statement_spans(clean, tokenizer)
        lo, hi = rng.choice(spans) if spans else (valid[0], valid[-1] + 1)
        mask_positions(range(lo, hi))
    elif policy == "ast_subtree":
        spans = _subtree_spans(clean, tokenizer)
        lo, hi = rng.choice(spans) if spans else (valid[0], valid[-1] + 1)
        mask_positions(range(lo, hi))
    elif policy == "reference":
        groups = _reference_groups(clean, tokenizer)
        mask_positions(
            rng.choice(groups) if groups else _contiguous_positions(valid, count, rng)
        )
    elif policy == "edit_local":
        local = [index for index in edit_indices if index in valid]
        mask_positions(local or _contiguous_positions(valid, count, rng))
    elif policy == "disjoint":
        mask_positions(_disjoint_positions(valid, count, rng))
    elif policy == "all_mask":
        mask_positions(valid)
    elif policy == "expansion":
        positions = _contiguous_positions(valid, count, rng)
        mask_positions(positions, inserted=True)
        source_length -= len(positions)
    elif policy == "contraction":
        capacity = max(0, (max_length or len(clean) + overallocate) - len(clean))
        extra = min(overallocate, count, capacity)
        if extra:
            spans = _statement_spans(clean, tokenizer)
            boundary = rng.choice([lo for lo, _ in spans] or [max(1, len(clean) - 1)])
            candidates = [
                clean[index] for index in valid if clean[index] != tokenizer.eos_id
            ]
            distractors = [rng.choice(candidates) for _ in range(extra)]
            target[boundary:boundary] = [tokenizer.pad_id] * extra
            noisy[boundary:boundary] = distractors
            predict[boundary:boundary] = [True] * extra
            insertion[boundary:boundary] = [False] * extra
            deletion[boundary:boundary] = [True] * extra
            source_length += extra
        else:
            mask_positions(_contiguous_positions(valid, count, rng))
    elif policy == "reorder":
        spans = _statement_spans(clean, tokenizer)
        if len(spans) >= 2:
            (lo1, hi1), (lo2, hi2) = sorted(rng.sample(spans, 2))
            noisy = (
                clean[:lo1]
                + clean[lo2:hi2]
                + clean[hi1:lo2]
                + clean[lo1:hi1]
                + clean[hi2:]
            )
        else:
            lo, hi = spans[0] if spans else (valid[0], valid[-1] + 1)
            noisy[lo:hi] = reversed(noisy[lo:hi])
        predict = [left != right for left, right in zip(noisy, target)]

    if not any(predict):
        mask_positions([valid[0]])
    bucket_index, bucket = length_bucket(len(clean), length_buckets)
    aux = _aux_labels(
        target,
        tokenizer,
        predict,
        target_length=len(clean),
        error_category=error_category,
    )
    return DiffusionCorruption(
        clean_ids=tuple(clean),
        target_ids=tuple(target),
        noisy_ids=tuple(noisy),
        predict_mask=tuple(predict),
        insertion_mask=tuple(insertion),
        deletion_mask=tuple(deletion),
        policy=policy,
        source_length=source_length,
        target_length=len(clean),
        canvas_length=len(target),
        length_bucket=bucket,
        length_bucket_index=bucket_index,
        pad_id=tokenizer.pad_id,
        aux_labels=aux,
    )


def align_token_edits(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    tokenizer: TokenizerLike,
    *,
    length_buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
    max_length: int | None = None,
    error_category: str | None = None,
) -> DiffusionCorruption:
    """Align a real short/long edit pair onto one insert/delete canvas."""
    source = _trim_padding(source_ids, tokenizer.pad_id)
    clean = _trim_padding(target_ids, tokenizer.pad_id)
    if not source or not clean:
        raise ValueError("source_ids and target_ids must contain non-padding tokens")

    target: list[int] = []
    noisy: list[int] = []
    predict: list[bool] = []
    insertion: list[bool] = []
    deletion: list[bool] = []

    def append(
        source_id: int, target_id: int, *, inserted: bool, deleted: bool
    ) -> None:
        noisy.append(source_id)
        target.append(target_id)
        predict.append(source_id != target_id)
        insertion.append(inserted)
        deletion.append(deleted)

    matcher = SequenceMatcher(a=source, b=clean, autojunk=False)
    for tag, source_lo, source_hi, target_lo, target_hi in matcher.get_opcodes():
        if tag == "equal":
            for source_id, target_id in zip(
                source[source_lo:source_hi], clean[target_lo:target_hi]
            ):
                append(source_id, target_id, inserted=False, deleted=False)
            continue
        if tag == "delete":
            for source_id in source[source_lo:source_hi]:
                append(source_id, tokenizer.pad_id, inserted=False, deleted=True)
            continue
        if tag == "insert":
            for target_id in clean[target_lo:target_hi]:
                append(tokenizer.mask_id, target_id, inserted=True, deleted=False)
            continue

        source_chunk = source[source_lo:source_hi]
        target_chunk = clean[target_lo:target_hi]
        paired = min(len(source_chunk), len(target_chunk))
        for index in range(paired):
            append(
                source_chunk[index],
                target_chunk[index],
                inserted=False,
                deleted=False,
            )
        for source_id in source_chunk[paired:]:
            append(source_id, tokenizer.pad_id, inserted=False, deleted=True)
        for target_id in target_chunk[paired:]:
            append(tokenizer.mask_id, target_id, inserted=True, deleted=False)

    if max_length is not None and len(target) > max_length:
        raise ValueError(
            f"aligned edit canvas length {len(target)} exceeds max_length {max_length}"
        )
    if not any(predict):
        valid = _valid_positions(target, tokenizer)
        index = valid[0] if valid else 0
        noisy[index] = tokenizer.mask_id
        predict[index] = True
    if len(clean) > len(source):
        policy = "expansion"
    elif len(clean) < len(source):
        policy = "contraction"
    else:
        policy = "reorder"
    bucket_index, bucket = length_bucket(len(clean), length_buckets)
    return DiffusionCorruption(
        clean_ids=tuple(clean),
        target_ids=tuple(target),
        noisy_ids=tuple(noisy),
        predict_mask=tuple(predict),
        insertion_mask=tuple(insertion),
        deletion_mask=tuple(deletion),
        policy=policy,
        source_length=len(source),
        target_length=len(clean),
        canvas_length=len(target),
        length_bucket=bucket,
        length_bucket_index=bucket_index,
        pad_id=tokenizer.pad_id,
        aux_labels=_aux_labels(
            target,
            tokenizer,
            predict,
            target_length=len(clean),
            error_category=error_category,
        ),
    )


def corrupt_batch(
    rows: Sequence[Sequence[int]],
    tokenizer: TokenizerLike,
    *,
    config: DiffusionConfig,
    rng: random.Random,
    metadata: Sequence[Mapping[str, Any] | None] | None = None,
) -> DiffusionBatch:
    """Sample a fresh corruption per row on every call."""
    output: list[DiffusionCorruption] = []
    for index, row in enumerate(rows):
        meta = metadata[index] if metadata and index < len(metadata) else None
        output.append(
            corrupt_tokens(
                row,
                tokenizer,
                policy=rng.choice(config.policies),
                rng=rng,
                mask_rate=rng.uniform(config.mask_min, config.mask_max),
                overallocate=config.overallocate,
                length_buckets=config.length_buckets,
                max_length=config.max_length,
                edit_indices=edit_token_indices(meta),
                error_category=(meta or {}).get("error_category"),
            )
        )
    return DiffusionBatch(tuple(output))
