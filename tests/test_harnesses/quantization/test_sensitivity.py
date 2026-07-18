"""Regression tests for CAP3-04 quantization sensitivity profiling."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch")

from slm_training.harnesses.quantization.calibration import CalibrationCorpusManifest, CalibrationSample
from slm_training.harnesses.quantization.sensitivity import (
    GroupingPolicy,
    ParameterGroup,
    _iter_group_params,
    compute_gradient_proxy,
    default_grouping_policy,
    profile_group_sensitivity,
)
from slm_training.models.local_action_head import LocalFlatHead, StateContext


class ToyLocalModel(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.local_head = LocalFlatHead(hidden_dim)


def _make_samples(n: int = 8) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    actions = ["a:0", "a:1", "a:2"]
    for i in range(n):
        legal = tuple(actions[: (i % 3) + 1])
        selected = legal[i % len(legal)]
        samples.append(
            CalibrationSample(
                trace_id=f"t-{i}",
                state_fingerprint=f"s-{i % 4}",
                state_signature_version="1",
                legal_action_ids=legal,
                selected_action_id=selected,
                target_action_ids=(selected,),
                top1_margin=0.5,
                posterior_entropy_bits=0.0,
                scope_signature="scope",
                template_signature="tpl",
                production_weight=1.0,
                bin_id=None,
                sensitivity_score=None,
                verification_outcome=None,
            )
        )
    return samples


def _make_manifest(samples: list[CalibrationSample]) -> CalibrationCorpusManifest:
    return CalibrationCorpusManifest(
        schema_version="cap3-02.v1",
        source_trace_ids=[s.trace_id for s in samples],
        checkpoint_id="toy",
        teacher_id="toy",
        state_signature_version="1",
        sample_count=len(samples),
        sampling_strategy="uniform_state",
        inclusion_rules={},
        exclusion_rules={},
        coverage_fields={},
        raw_production_frequency_weights={},
        bin_edges=None,
        calibration_split_hashes=[],
        test_split_hashes=[],
        no_test_leakage_asserted=False,
    )


@pytest.fixture
def model() -> nn.Module:
    return ToyLocalModel(hidden_dim=16)


@pytest.fixture
def samples() -> list[CalibrationSample]:
    return _make_samples()


@pytest.fixture
def manifest(samples: list[CalibrationSample]) -> CalibrationCorpusManifest:
    return _make_manifest(samples)


def test_default_policy_matches_local_head_groups(model: nn.Module) -> None:
    policy = default_grouping_policy()
    scorer = next((g for g in policy.groups if g.group_id == "local_head/scorer"), None)
    emb = next((g for g in policy.groups if g.group_id == "local_head/embeddings"), None)
    assert scorer is not None
    assert emb is not None
    # Warm embeddings so they exist.
    ctx = StateContext(state_family_id="test")
    model.local_head.score(torch.randn(1, 16), ctx, ["a:0", "a:1"])
    scorer_params = [n for n, _, _ in _iter_group_params(model, scorer)]
    emb_params = [n for n, _, _ in _iter_group_params(model, emb)]
    assert any("scorer.weight" in n for n in scorer_params)
    assert len(emb_params) >= 2


def test_custom_group_policy_matches_expected_modules(model: nn.Module) -> None:
    policy = GroupingPolicy(
        version="test-v1",
        groups=(
            ParameterGroup(group_id="scorer", path_patterns=(r"local_head\.scorer",), quantize_kind="linear_weight"),
            ParameterGroup(group_id="embeddings", param_name_patterns=(r"local_head\.action_embeddings\.",), quantize_kind="embedding_dict"),
        ),
    )
    ctx = StateContext(state_family_id="test")
    model.local_head.score(torch.randn(1, 16), ctx, ["a:0"])
    scorer_group = policy.groups[0]
    emb_group = policy.groups[1]
    assert any("scorer.weight" in n for n, _, _ in _iter_group_params(model, scorer_group))
    assert any("action_embeddings" in n for n, _, _ in _iter_group_params(model, emb_group))


def test_profile_group_sensitivity_returns_ok_points(model: nn.Module, manifest: CalibrationCorpusManifest, samples: list[CalibrationSample]) -> None:
    policy = GroupingPolicy(
        version="test-v1",
        groups=(
            ParameterGroup(group_id="scorer", path_patterns=(r"local_head\.scorer",), quantize_kind="linear_weight"),
            ParameterGroup(group_id="embeddings", param_name_patterns=(r"local_head\.action_embeddings\.",), quantize_kind="embedding_dict"),
        ),
    )
    report = profile_group_sensitivity(
        model,
        manifest,
        samples,
        policy,
        formats=("ternary", "int4"),
        group_size=8,
        hidden_dim=16,
    )
    assert report.version == "cap3-04-v1"
    assert len(report.points) == 4  # 2 groups x 2 formats
    ok_points = [p for p in report.points if p.status == "ok"]
    assert len(ok_points) == 4
    assert {p.group_id for p in ok_points} == {"scorer", "embeddings"}


def test_baseline_parameters_restored_after_profiling(model: nn.Module, manifest: CalibrationCorpusManifest, samples: list[CalibrationSample]) -> None:
    policy = GroupingPolicy(
        version="test-v1",
        groups=(
            ParameterGroup(group_id="embeddings", param_name_patterns=(r"local_head\.action_embeddings\.",), quantize_kind="embedding_dict"),
        ),
    )
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    profile_group_sensitivity(
        model,
        manifest,
        samples,
        policy,
        formats=("ternary", "int4"),
        group_size=8,
        hidden_dim=16,
    )
    for n, p in model.named_parameters():
        assert torch.allclose(p.data, before[n])


def test_excluded_group_reported_with_reason(model: nn.Module, manifest: CalibrationCorpusManifest, samples: list[CalibrationSample]) -> None:
    policy = GroupingPolicy(
        version="test-v1",
        groups=(
            ParameterGroup(
                group_id="norms",
                param_name_patterns=(r"\.norm",),
                quantize_kind="none",
                exclusion_reason="test exclusion",
            ),
        ),
    )
    report = profile_group_sensitivity(
        model,
        manifest,
        samples,
        policy,
        formats=("ternary",),
        group_size=8,
        hidden_dim=16,
    )
    assert len(report.points) == 1
    assert report.points[0].status == "excluded"
    assert "test exclusion" in report.points[0].notes


def test_gradient_proxy_runs_for_groups(model: nn.Module, samples: list[CalibrationSample]) -> None:
    policy = GroupingPolicy(
        version="test-v1",
        groups=(
            ParameterGroup(group_id="scorer", path_patterns=(r"local_head\.scorer",), quantize_kind="linear_weight"),
            ParameterGroup(group_id="embeddings", param_name_patterns=(r"local_head\.action_embeddings\.",), quantize_kind="embedding_dict"),
        ),
    )
    proxies = compute_gradient_proxy(model, samples, policy, hidden_dim=16)
    assert "scorer" in proxies
    assert "embeddings" in proxies
    assert all(v >= 0.0 for v in proxies.values())
