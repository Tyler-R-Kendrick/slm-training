"""P1b mixture manifest + online sampling tests."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from slm_training.data.mixture import (
    DEFAULT_TASK_WEIGHTS,
    NEW_FAMILIES,
    MixtureManifest,
    corpus_diagnostics,
    default_base_weights,
    fit_weight_regression,
    load_mixture_manifest,
    local_probe_candidates,
    mixture_hash,
    propose_from_fit,
    sample_mixture_batch,
    task_group,
    write_mixture_manifest,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.catalog import KNOWN_FAMILIES


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


def test_default_mix_and_probes_cover_new_families() -> None:
    base = default_base_weights()
    assert set(NEW_FAMILIES) <= set(base)
    assert set(KNOWN_FAMILIES) <= set(base)
    probes = local_probe_candidates(base, task_weights=DEFAULT_TASK_WEIGHTS)
    probed = {
        probe.mixture_id.removeprefix("local_").rsplit("_", 1)[0] for probe in probes
    }
    assert set(NEW_FAMILIES) <= probed
    assert all(probe.mixture_id.rsplit("_", 1)[-1] != "1" for probe in probes)


def test_task_balanced_sampling_ignores_row_count_skew() -> None:
    records = [
        ExampleRecord(
            id=f"g{i}",
            prompt=f"g{i}",
            openui='root = Button(":x")',
            meta={"source_family": "programspec_generated", "task": "generation"},
        )
        for i in range(100)
    ] + [
        ExampleRecord(
            id="repair",
            prompt="repair",
            openui='root = Button(":x")',
            meta={"source_family": "corruption_repair", "task": "repair"},
        )
    ]
    batch = sample_mixture_batch(
        records,
        weights={"programspec_generated": 0.5, "corruption_repair": 0.5},
        task_weights={"generation": 0.5, "repair_completion_inpaint": 0.5},
        batch_size=1000,
        rng=random.Random(9),
    )
    generation = sum(row.meta["task"] == "generation" for row in batch)
    assert 430 < generation < 570


def test_capacity_aware_sampling_limits_repeats_to_pool_cycles() -> None:
    records = [
        ExampleRecord(
            id=f"generation-{i}",
            prompt="generate",
            openui='root = Button(":x")',
            meta={"source_family": "programspec_generated", "task": "generation"},
        )
        for i in range(20)
    ] + [
        ExampleRecord(
            id="single-repair",
            prompt="repair",
            openui='root = Button(":x")',
            meta={"source_family": "corruption_repair", "task": "repair"},
        )
    ]

    batch = sample_mixture_batch(
        records,
        weights={"programspec_generated": 0.5, "corruption_repair": 0.5},
        task_weights={"generation": 0.5, "repair_completion_inpaint": 0.5},
        batch_size=21,
        rng=random.Random(9),
        sampling_policy="capacity_aware",
    )

    assert len({record.id for record in batch}) == 21
    assert sum(record.id == "single-repair" for record in batch) == 1


def test_capacity_aware_sampling_is_deterministic_and_cycles() -> None:
    records = [_rec(f"r{i}", "rico_real") for i in range(3)]
    first = sample_mixture_batch(
        records,
        weights={"rico_real": 1.0},
        batch_size=7,
        rng=random.Random(4),
        sampling_policy="capacity_aware",
    )
    second = sample_mixture_batch(
        records,
        weights={"rico_real": 1.0},
        batch_size=7,
        rng=random.Random(4),
        sampling_policy="capacity_aware",
    )

    assert [record.id for record in first] == [record.id for record in second]
    assert len({record.id for record in first[:3]}) == 3
    assert len({record.id for record in first[3:6]}) == 3


def test_quota_capacity_aware_sampling_preserves_task_allocation() -> None:
    records = [
        ExampleRecord(
            id=f"generation-{i}",
            prompt="generate",
            openui='root = Button(":x")',
            meta={"source_family": "programspec_generated", "task": "generation"},
        )
        for i in range(20)
    ] + [
        ExampleRecord(
            id=f"repair-{i}",
            prompt="repair",
            openui='root = Button(":x")',
            meta={"source_family": "corruption_repair", "task": "repair"},
        )
        for i in range(20)
    ] + [
        ExampleRecord(
            id=f"edit-{i}",
            prompt="edit",
            openui='root = Button(":x")',
            meta={"source_family": "edit_trajectory", "task": "edit"},
        )
        for i in range(20)
    ]

    batch = sample_mixture_batch(
        records,
        weights={
            "programspec_generated": 1.0,
            "corruption_repair": 1.0,
            "edit_trajectory": 1.0,
        },
        task_weights={
            "generation": 1.0,
            "repair_completion_inpaint": 1.0,
            "patch_edit": 1.0,
        },
        batch_size=32,
        rng=random.Random(9),
        sampling_policy="quota_capacity_aware",
    )

    counts = Counter(task_group(record.meta["task"]) for record in batch)
    assert sorted(counts.values()) == [10, 11, 11]
    assert len({record.id for record in batch}) == 32


def test_manifest_v1_loads_without_task_policy(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"mixture_id":"legacy","version":1,"weights":{"rico_real":1}}\n',
        encoding="utf-8",
    )
    assert load_mixture_manifest(path).task_weights is None


def test_loads_canonical_pipeline_mixture_envelope(tmp_path: Path) -> None:
    path = tmp_path / "mixture.json"
    path.write_text(
        '{"manifest":{"mixture_id":"canonical","version":2,'
        '"weights":{"human_curated":1},'
        '"task_weights":{"generation":1}},"diagnostics":{}}\n',
        encoding="utf-8",
    )

    manifest = load_mixture_manifest(path)

    assert manifest.mixture_id == "canonical"
    assert manifest.weights == {"human_curated": 1.0}
    assert manifest.task_weights == {"generation": 1.0}


def test_corpus_diagnostics_reports_task_and_structure_coverage() -> None:
    rows = [
        ExampleRecord(
            id="a",
            prompt="a",
            openui='root = Button(":x")',
            meta={
                "source_family": "programspec_generated",
                "task": "generation",
                "program_family_id": "pf-a",
            },
        ),
        ExampleRecord(
            id="b",
            prompt="b",
            openui='root = Card([Button(":x")])',
            meta={"source_family": "visual_edit", "task": "edit"},
        ),
    ]
    report = corpus_diagnostics(rows, configured_weights=default_base_weights())
    assert report["task_group_counts"] == {"generation": 1, "patch_edit": 1}
    assert report["unique_program_families"] == 1
    assert report["observed_component_counts"]["Button"] == 2


def test_regression_handles_normalized_simplex_with_intercept() -> None:
    rows = [
        {"weights": {"a": 0.2, "b": 0.8}, "weighted_nll": 1.8},
        {"weights": {"a": 0.5, "b": 0.5}, "weighted_nll": 1.5},
        {"weights": {"a": 0.8, "b": 0.2}, "weighted_nll": 1.2},
    ]
    fit = fit_weight_regression(rows)
    assert any(abs(value) > 0.01 for value in fit["coefficients"].values())
