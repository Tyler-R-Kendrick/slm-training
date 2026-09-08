"""Identity-compatible diagnostic evidence for the existing posterior selector."""

from __future__ import annotations

import math
import hashlib
import json
from collections import defaultdict
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def contract_digest(value):
    """Canonical advisory payload identity, checked against runtime at the boundary."""
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class SearchIdentity(Contract):
    schema_version: Literal["search_identity/v1"] = "search_identity/v1"
    endpoint: str = Field(min_length=1)
    endpoint_version: str = Field(min_length=1)
    units: str = Field(min_length=1)
    direction: Literal["maximize", "minimize"]
    estimator: str = Field(min_length=1)
    suite_digest: Digest
    selection_digest: Digest
    data_digest: Digest
    ancestor_digest: Digest
    model_digest: Digest
    host_digest: Digest
    fidelity_digest: Digest
    analysis_digest: Digest
    independence: Literal["conditional_on_ancestor", "independent_initializations"]


class SearchEffect(Contract):
    schema_version: Literal["search_effect/v1"] = "search_effect/v1"
    identity: SearchIdentity
    slug: str = Field(min_length=1)
    treatment_id: Digest
    replicate_id: str = Field(min_length=1)
    comparison_id: Digest
    attempt_id: str = Field(min_length=1)
    complete: bool
    control: float = Field(strict=True)
    candidate: float = Field(strict=True)
    paired_case_ids: tuple[str, ...] = Field(min_length=1)
    paired_root_ids: tuple[str, ...] = Field(min_length=1)
    required_case_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def paired_coverage(self):
        if len(set(self.required_case_ids)) != len(self.required_case_ids):
            raise ValueError("locked case identities must be unique")
        if len(self.paired_root_ids) != len(self.paired_case_ids):
            raise ValueError("one root identity per paired case required")
        if len(set(self.paired_case_ids)) != len(self.paired_case_ids):
            raise ValueError("replayed cases cannot add inferential weight")
        if len(set(self.paired_root_ids)) != len(self.paired_root_ids):
            raise ValueError(
                "dependent roots require a separately declared blocked analysis"
            )
        if self.complete and set(self.paired_case_ids) != set(self.required_case_ids):
            raise ValueError("complete comparison requires all locked pairs")
        if self.identity.units == "rate" and not (
            0 <= self.control <= 1 and 0 <= self.candidate <= 1
        ):
            raise ValueError("rates must be in [0, 1]")
        if not math.isfinite(self.candidate - self.control):
            raise ValueError("effect subtraction overflow")
        return self

    @property
    def benefit(self):
        delta = self.candidate - self.control
        return delta if self.identity.direction == "maximize" else -delta


def compatible_effects(
    raw_effects, identity: SearchIdentity
) -> tuple[list[SearchEffect], list[str]]:
    """Retries and legacy aggregate counts are never independent comparisons."""
    accepted: dict[tuple[str, str], SearchEffect] = {}
    comparisons = {}
    rejected = []
    for raw in raw_effects:
        try:
            row = SearchEffect.model_validate(
                raw.model_dump() if isinstance(raw, SearchEffect) else raw
            )
        except ValueError:
            rejected.append("invalid_search_effect")
            continue
        if row.identity != identity or not row.complete:
            rejected.append(
                "incompatible_identity" if row.identity != identity else "incomplete"
            )
            continue
        key = (row.treatment_id, row.replicate_id)
        if row.comparison_id in comparisons and comparisons[row.comparison_id] != key:
            raise ValueError("comparison relabelled as another replicate")
        old = accepted.get(key)
        if old is not None:
            if old.model_dump(exclude={"attempt_id"}) != row.model_dump(
                exclude={"attempt_id"}
            ):
                raise ValueError("conflicting retry evidence for one planned replicate")
            rejected.append("duplicate_retry")
            continue
        accepted[key] = row
        comparisons[row.comparison_id] = key
    return list(accepted.values()), rejected


def rank_compatible(
    candidates,
    *,
    posterior,
    identity=None,
    effects=(),
    exploration_c=1.0,
    prior_scale=0.05,
    rotation_order=None,
):
    """Existing UCB formula; no evidence means its exploration prior, not a null."""
    if (
        not math.isfinite(exploration_c)
        or exploration_c < 0
        or not math.isfinite(prior_scale)
        or prior_scale <= 0
    ):
        raise ValueError("invalid UCB parameters")
    rows = []
    if identity is not None:
        identity = SearchIdentity.model_validate(
            identity.model_dump() if isinstance(identity, SearchIdentity) else identity
        )
        rows = compatible_effects(effects, identity)[0]
    by_slug = defaultdict(list)
    for row in rows:
        by_slug[row.slug].append(row.benefit)
    rotation = {slug: i for i, slug in enumerate(rotation_order or candidates)}

    def key(slug):
        values = by_slug[slug]
        mean = sum(values) / len(values) if values else 0.0
        score = posterior(
            n=float(len(values)),
            mean=mean,
            m2=sum((value - mean) ** 2 for value in values),
            prior_scale=prior_scale,
        )
        return (-score.ucb(exploration_c), rotation.get(slug, len(rotation)), slug)

    return sorted(dict.fromkeys(candidates), key=key)


def record_search_effect(store, effect: SearchEffect):
    """Use the existing campaign artifact/event owners; never publish a promotion."""
    effect = SearchEffect.model_validate(effect.model_dump())
    artifact = store.write_artifact("search_effects", effect)
    store.append_event(
        "search_effect_recorded",
        artifact_sha256=artifact.stem,
        idempotency_key=f"search-effect:{effect.comparison_id}:{effect.attempt_id}",
        detail={
            "claim_class": "diagnostic_only",
            "comparison_id": effect.comparison_id,
        },
    )
    return artifact


def effect_from_loss_reports(identity: SearchIdentity, lineage, control, candidate):
    """Actual loss-suite producer to UCB consumer, paired by locked case identity.

    The coordinator supplies the resolved experimental lineage, never a worker
    verdict. Averaging is over cases; token counts are not independent units.
    """
    selection = control["selection"]
    if selection != candidate["selection"]:
        raise ValueError("loss selections differ")
    if (
        control["estimator_id"] != candidate["estimator_id"]
        or control["estimator_id"] != identity.estimator
    ):
        raise ValueError("loss estimator differs from search identity")
    if selection["selection_sha256"] != identity.selection_digest:
        raise ValueError("search selection identity mismatch")
    ids = selection["selected_record_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("nonempty unique selected cases required")
    if len(selection["input_sha256s"]) != len(ids) or len(
        selection["selected_root_ids"]
    ) != len(ids):
        raise ValueError("selection identities must align")
    arms = []
    paired_identities = []
    for report in (control, candidate):
        rows = _loss_arm(report, selection, identity)
        paired_identities.append(
            [(rows[key]["seed"], rows[key]["evaluator_sha256"]) for key in ids]
        )
        arms.append(sum(rows[key]["nll"] / len(ids) for key in ids))
    if paired_identities[0] != paired_identities[1]:
        raise ValueError("paired randomness/evaluator differs")
    return SearchEffect(
        identity=identity,
        **lineage,
        complete=True,
        control=arms[0],
        candidate=arms[1],
        paired_case_ids=tuple(ids),
        required_case_ids=tuple(ids),
        paired_root_ids=tuple(selection["selected_root_ids"]),
    )


def _loss_arm(report, selection, identity):
    ids = selection["selected_record_ids"]
    required = set(ids)
    # The canonical producer also emits auxiliary OOD-only rows with nll=None.
    # Keep every selected row (even an invalid/missing loss) and unexpected broad
    # observations so neither missingness nor extra broad coverage is hidden.
    broad = [
        row
        for row in report["per_record"]
        if row["id"] in required or row.get("nll") is not None
    ]
    rows = {row["id"]: row for row in broad}
    if len(rows) != len(broad) or set(rows) != required:
        raise ValueError("incomplete or duplicated loss rows")
    for index, record_id in enumerate(ids):
        row = rows[record_id]
        if (
            row["input_sha256"] != selection["input_sha256s"][index]
            or row["root_id"] != selection["selected_root_ids"][index]
            or row["units"] != identity.units
            or row["estimator_id"] != identity.estimator
            or row["selection_sha256"] != identity.selection_digest
        ):
            raise ValueError("loss row identity mismatch")
        _validate_loss_value(row)
    return rows


def _validate_loss_value(row):
    count = row["masked_tokens"]
    if type(count) is not int or count <= 0:
        raise ValueError("positive integer masked-token denominator required")
    for key in ("nll", "nll_sum"):
        value = row[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative denoising loss required")
    if not math.isclose(row["nll"], row["nll_sum"] / count, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("loss numerator/denominator mismatch")
