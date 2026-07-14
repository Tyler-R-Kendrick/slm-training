"""P1b mixture manifest + online sampling tests."""

from __future__ import annotations

import random
from pathlib import Path

from slm_training.data.mixture import (
    MixtureManifest,
    default_base_weights,
    fit_weight_regression,
    load_mixture_manifest,
    local_probe_candidates,
    mixture_hash,
    propose_from_fit,
    sample_mixture_batch,
    write_mixture_manifest,
)
from slm_training.dsl.schema import ExampleRecord


def _rec(rid: str, family: str) -> ExampleRecord:
    return ExampleRecord(
        id=rid,
        prompt=f"p-{rid}",
        openui='root = Button(":x")',
        split="train",
        source=family,
        meta={"source_family": family},
    )


def test_mixture_normalize_and_hash(tmp_path: Path) -> None:
    m = MixtureManifest(
        mixture_id="m1",
        weights={"rico_real": 2.0, "human_curated": 2.0},
    ).normalized()
    assert abs(m.weights["rico_real"] - 0.5) < 1e-9
    path = write_mixture_manifest(tmp_path / "m1.json", m)
    loaded = load_mixture_manifest(path)
    assert mixture_hash(loaded) == mixture_hash(m)


def test_sample_mixture_batch_respects_weights() -> None:
    records = [_rec(f"r{i}", "rico_real") for i in range(20)] + [
        _rec(f"h{i}", "human_curated") for i in range(20)
    ]
    rng = random.Random(0)
    batch = sample_mixture_batch(
        records,
        weights={"rico_real": 0.9, "human_curated": 0.1},
        batch_size=200,
        rng=rng,
    )
    rico = sum(1 for r in batch if r.meta["source_family"] == "rico_real")
    assert rico > 140  # strongly skewed


def test_sample_mixture_batch_deterministic_with_rng() -> None:
    records = [_rec(f"r{i}", "rico_real") for i in range(10)]
    a = sample_mixture_batch(
        records, weights={"rico_real": 1.0}, batch_size=5, rng=random.Random(7)
    )
    b = sample_mixture_batch(
        records, weights={"rico_real": 1.0}, batch_size=5, rng=random.Random(7)
    )
    assert [r.id for r in a] == [r.id for r in b]


def test_local_probes_and_regression_propose() -> None:
    base = default_base_weights()
    probes = local_probe_candidates(base)
    assert len(probes) >= 4
    rows = []
    for i, probe in enumerate(probes[:6]):
        # Fake NLL rises with paraphrase weight.
        nll = 1.0 + 2.0 * float(probe.weights.get("prompt_paraphrase", 0.0))
        rows.append({"weights": probe.weights, "weighted_nll": nll + 0.01 * i})
    fit = fit_weight_regression(rows)
    assert "coefficients" in fit
    proposals = propose_from_fit(fit, base=base, n=2)
    assert len(proposals) == 2
