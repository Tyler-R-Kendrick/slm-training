from dataclasses import replace

from slm_training.data.flow.bridge_corpus import load_corpus
from slm_training.harnesses.experiments.slm198_bridge_curriculum import (
    ARMS,
    BridgeCurriculumManifestV1,
    BridgeCurriculumSampler,
    build_manifest,
    compute_difficulties,
    run_matrix,
)
from slm_training.harnesses.experiments.slm197_direct_bridge_policy import DEFAULT_CORPUS


def _fixture():
    rows, candidate_sets, manifest = load_corpus(DEFAULT_CORPUS)
    return [row for row in rows if row.split == "train"], candidate_sets, manifest


def test_manifests_have_matched_final_support_and_target_first_contract() -> None:
    rows, _, source = _fixture()
    manifests = [
        build_manifest(
            rows,
            arm=arm,
            seed=0,
            epochs=3,
            source_content_fingerprint=source["content_fingerprint"],
        )
        for arm in ARMS
    ]
    assert len({manifest.final_support_digest for manifest in manifests}) == 1
    assert all(manifest.total_exposures == 3 * len(rows) for manifest in manifests)
    assert not manifests[0].target_first
    assert all(manifest.target_first for manifest in manifests[1:])
    assert BridgeCurriculumManifestV1.from_dict(manifests[2].to_dict()) == manifests[2]


def test_difficulty_is_deterministic_and_confirmation_independent() -> None:
    rows, candidate_sets, _ = _fixture()
    first = compute_difficulties(rows, candidate_sets)
    second = compute_difficulties(tuple(reversed(rows)), candidate_sets)
    assert {key: value.to_dict() for key, value in first.items()} == {
        key: value.to_dict() for key, value in second.items()
    }


def test_sampler_resume_and_anti_curriculum_are_exact() -> None:
    rows, candidate_sets, source = _fixture()
    difficulty = compute_difficulties(rows, candidate_sets)
    forward = build_manifest(
        rows,
        arm="length_curriculum",
        seed=7,
        epochs=2,
        source_content_fingerprint=source["content_fingerprint"],
    )
    anti = build_manifest(
        rows,
        arm="anti_curriculum",
        seed=7,
        epochs=2,
        source_content_fingerprint=source["content_fingerprint"],
    )
    sampler = BridgeCurriculumSampler(rows, forward, difficulty)
    first = next(sampler).row_id
    resumed = BridgeCurriculumSampler.resume(rows, forward, difficulty, sampler.state_dict())
    suffix = [row.row_id for row in resumed]
    assert [first, *suffix] == [
        row.row_id for row in BridgeCurriculumSampler(rows, forward, difficulty)
    ]
    assert [
        row.row_id for row in BridgeCurriculumSampler(rows, anti, difficulty)
    ] != [row.row_id for row in BridgeCurriculumSampler(rows, forward, difficulty)]
    first_forward = next(
        iter(BridgeCurriculumSampler(rows, forward, difficulty))
    )
    first_anti = next(iter(BridgeCurriculumSampler(rows, anti, difficulty)))
    assert (
        difficulty[first_anti.row_id].candidate_count
        > difficulty[first_forward.row_id].candidate_count
    )


def test_target_first_sampler_balances_skewed_row_multiplicity() -> None:
    rows, candidate_sets, source = _fixture()
    base = rows[0]
    skewed = [
        replace(
            base,
            row_id=f"target-a-{index}",
            bridge_id=f"path-a-{index // 2}",
            target_cluster_id="target-a",
        )
        for index in range(4)
    ] + [
        replace(
            rows[1],
            row_id="target-b-0",
            bridge_id="path-b-0",
            target_cluster_id="target-b",
        )
    ]
    difficulty = compute_difficulties(skewed, candidate_sets)
    uniform = build_manifest(
        skewed,
        arm="uniform_rows",
        seed=0,
        epochs=2,
        source_content_fingerprint=source["content_fingerprint"],
        difficulty=difficulty,
    )
    balanced = build_manifest(
        skewed,
        arm="uniform_targets",
        seed=0,
        epochs=2,
        source_content_fingerprint=source["content_fingerprint"],
        difficulty=difficulty,
    )
    uniform_targets = [
        row.target_cluster_id
        for row in BridgeCurriculumSampler(skewed, uniform, difficulty)
    ]
    balanced_targets = [
        row.target_cluster_id
        for row in BridgeCurriculumSampler(skewed, balanced, difficulty)
    ]
    assert uniform_targets.count("target-a") == 8
    assert balanced_targets.count("target-a") == 5
    assert balanced_targets.count("target-b") == 5


def test_matrix_reconciles_exposure_and_rejects_fixture_claim() -> None:
    report = run_matrix(seeds=(0,), epochs=1)
    assert all(report["matched_controls"].values())
    assert report["confirmation"]["status"] == "blocked"
    assert report["confirmation"]["selected_arm"] is None
    assert report["checkpoint"]["written"] is False
    assert report["honest_verdict"] == "reject_curriculum_fixture_indistinguishable"
    for arm in ARMS:
        run = report["arms"][arm]["runs"][0]
        assert run["training"]["exposures"] == report["recipe"]["train_rows"]
        assert run["evaluation"]["candidate_membership"]["exact"]
        assert "traces" not in run["free_running"]
        assert len(run["free_running"]["trace_digest"]) == 64


def test_matrix_rejects_over_cap() -> None:
    try:
        run_matrix(seeds=(0,), epochs=1, max_wall_minutes=3.1)
    except ValueError as exc:
        assert "max_wall_minutes" in str(exc)
    else:
        raise AssertionError("over-cap matrix must fail")
