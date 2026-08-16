"""Data-adequacy ladder: measure marginal utility of training data.

The direct answer to "generate more or stop?": train the same recipe on
nested subsets of one immutable corpus (echoing the generator's
``scaling_ladder`` nested-prefix semantics), measure the eval metric per
rung, and classify the marginal gain of the last doubling. The
classification — not a capacity prior — is the only admissible saturation
evidence for ``autoresearch.sample_adequacy``.

Decidability is enforced, never assumed: a ``flat`` classification may only
be claimed when the eval suite is powered for the preregistered minimum
detectable effect (``autoresearch.power.required_n_for_effect``). An
underpowered or variance-free measurement is ``undecidable`` — the ladder
refuses to convert an undecidable eval into a stop signal, which is the
failure mode that stalled hill climbing at n=3 smoke suites.

Artifacts are ``data_adequacy_ladder/v1`` JSON; runner:
``scripts/run_scaling_ladder.py --family data-adequacy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from slm_training.autoresearch.power import required_n_for_effect
from slm_training.harness_core.lineage.records import content_sha

DATA_ADEQUACY_LADDER_SCHEMA = "data_adequacy_ladder/v1"

LadderClassification = Literal["rising", "flat", "undecidable"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RungMeasurement(_Strict):
    """One trained rung: records used, metric observed, eval suite size."""

    records: int = Field(ge=1)
    metric: float
    suite_n: int = Field(ge=1)
    seed: int = 0


class MarginalGainVerdict(_Strict):
    classification: LadderClassification
    reason: str
    last_gain_per_100_records: float | None
    mde: float
    sd: float | None
    min_suite_n: int
    required_suite_n: int | None


def plan_nested_rungs(
    record_count: int, *, rung_count: int = 3, min_records: int = 16
) -> tuple[int, ...]:
    """Nested rung sizes by halving from the full corpus, smallest first."""

    if record_count < min_records:
        raise ValueError(
            f"corpus of {record_count} records is below min_records={min_records}"
        )
    rungs = [record_count]
    while len(rungs) < rung_count and rungs[-1] // 2 >= min_records:
        rungs.append(rungs[-1] // 2)
    return tuple(sorted(rungs))


def materialize_nested_subsets(
    train_dir: Path, out_root: Path, rungs: Sequence[int]
) -> list[tuple[int, Path]]:
    """Write first-N-record nested subsets of an immutable train snapshot.

    Subsets are wiring fixtures for ladder measurement, never promotion
    corpora: each manifest records the parent fingerprint, its rung size,
    and ``claim_class: fixture``.
    """

    records_path = train_dir / "records.jsonl"
    lines = records_path.read_text(encoding="utf-8").splitlines()
    manifest_path = train_dir / "manifest.json"
    parent_fingerprint = None
    if manifest_path.is_file():
        parent = json.loads(manifest_path.read_text(encoding="utf-8"))
        parent_fingerprint = parent.get("content_fingerprint")
    subsets: list[tuple[int, Path]] = []
    for rung in sorted(dict.fromkeys(int(r) for r in rungs)):
        if rung > len(lines):
            raise ValueError(
                f"rung {rung} exceeds corpus size {len(lines)} in {train_dir}"
            )
        subset_dir = out_root / f"rung_{rung:06d}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(lines[:rung]) + "\n"
        (subset_dir / "records.jsonl").write_text(payload, encoding="utf-8")
        (subset_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "data_adequacy_ladder_subset/v1",
                    "parent_train_dir": str(train_dir),
                    "parent_content_fingerprint": parent_fingerprint,
                    "record_count": rung,
                    "nested_prefix": True,
                    "claim_class": "fixture",
                    "promotion_authorized": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        subsets.append((rung, subset_dir))
    return subsets


def marginal_gains_per_100(
    measurements: Sequence[RungMeasurement], *, higher_is_better: bool = True
) -> list[float]:
    """Metric improvement per +100 records between adjacent rungs."""

    ordered = sorted(measurements, key=lambda m: m.records)
    if len(ordered) < 2:
        return []
    gains: list[float] = []
    for prev, curr in zip(ordered, ordered[1:]):
        delta = curr.metric - prev.metric
        if not higher_is_better:
            delta = -delta
        span = curr.records - prev.records
        if span <= 0:
            raise ValueError("rung record counts must strictly increase")
        gains.append(delta * 100.0 / span)
    return gains


def classify_marginal_gain(
    measurements: Sequence[RungMeasurement],
    *,
    mde: float,
    sd: float | None,
    alpha: float = 0.05,
    power: float = 0.8,
    higher_is_better: bool = True,
) -> MarginalGainVerdict:
    """Classify the last rung-to-rung gain, refusing undecidable claims.

    ``flat`` (the saturation signal) requires: at least two rungs, a
    positive ``sd`` estimate for the eval metric, and every suite size at or
    above ``required_n_for_effect(mde, sd)``. Anything less is
    ``undecidable`` — never ``flat``.
    """

    if mde <= 0:
        raise ValueError("mde must be positive")
    gains = marginal_gains_per_100(measurements, higher_is_better=higher_is_better)
    min_suite = min((m.suite_n for m in measurements), default=0)
    if not gains:
        return MarginalGainVerdict(
            classification="undecidable",
            reason="fewer than two rungs measured",
            last_gain_per_100_records=None,
            mde=mde,
            sd=sd,
            min_suite_n=min_suite,
            required_suite_n=None,
        )
    last_gain = gains[-1]
    if sd is None or sd <= 0:
        return MarginalGainVerdict(
            classification="undecidable",
            reason="no positive sd estimate for the eval metric",
            last_gain_per_100_records=last_gain,
            mde=mde,
            sd=sd,
            min_suite_n=min_suite,
            required_suite_n=None,
        )
    required = required_n_for_effect(mde, sd, alpha, power, paired=True)
    if min_suite < required:
        return MarginalGainVerdict(
            classification="undecidable",
            reason=(
                f"suite n={min_suite} is below the powered floor "
                f"n={required} for mde={mde}"
            ),
            last_gain_per_100_records=last_gain,
            mde=mde,
            sd=sd,
            min_suite_n=min_suite,
            required_suite_n=required,
        )
    classification: LadderClassification = (
        "flat" if last_gain <= mde else "rising"
    )
    return MarginalGainVerdict(
        classification=classification,
        reason=(
            f"last gain {last_gain:.6f} per 100 records vs mde={mde} at "
            f"powered suite n>={required}"
        ),
        last_gain_per_100_records=last_gain,
        mde=mde,
        sd=sd,
        min_suite_n=min_suite,
        required_suite_n=required,
    )


def build_ladder_artifact(
    *,
    train_dir: str,
    measurements: Sequence[RungMeasurement],
    verdict: MarginalGainVerdict,
    metric_name: str,
    higher_is_better: bool,
    version_stamp: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": DATA_ADEQUACY_LADDER_SCHEMA,
        "train_dir": train_dir,
        "metric_name": metric_name,
        "higher_is_better": higher_is_better,
        "measurements": [m.model_dump(mode="json") for m in measurements],
        "marginal_gains_per_100_records": marginal_gains_per_100(
            measurements, higher_is_better=higher_is_better
        ),
        "verdict": verdict.model_dump(mode="json"),
        "claim_class": "fixture",
        "promotion_authorized": False,
    }
    if version_stamp is not None:
        payload["version_stamp"] = dict(version_stamp)
    payload["content_sha256"] = content_sha(
        {k: v for k, v in payload.items() if k != "content_sha256"}
    )
    return payload


def load_ladder_classification(path: Path) -> tuple[bool | None, str]:
    """Read an artifact into (marginal_gain_flat, source) for adequacy.

    ``flat`` → ``True``; ``rising`` → ``False``; ``undecidable`` → ``None``
    (no saturation evidence — the adequacy verdict must not claim it).
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != DATA_ADEQUACY_LADDER_SCHEMA:
        raise ValueError(f"not a {DATA_ADEQUACY_LADDER_SCHEMA} artifact: {path}")
    classification = str((payload.get("verdict") or {}).get("classification"))
    flat: bool | None
    if classification == "flat":
        flat = True
    elif classification == "rising":
        flat = False
    else:
        flat = None
    return flat, str(path)


__all__ = [
    "DATA_ADEQUACY_LADDER_SCHEMA",
    "MarginalGainVerdict",
    "RungMeasurement",
    "build_ladder_artifact",
    "classify_marginal_gain",
    "load_ladder_classification",
    "marginal_gains_per_100",
    "materialize_nested_subsets",
    "plan_nested_rungs",
]
