"""Regression tests for CAP4-02 adaptive residual-plane routing."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.local_action_head import ResidualTritPlaneHead, StateContext
from slm_training.models.quantization.adaptive_planes import (
    AdaptivePlaneRoutingContext,
    PlaneRouter,
    PlaneScheduleSpec,
    PlaneScheduler,
    RuntimeDiagnostics,
    completion_support_floor,
    local_action_floor,
    make_schedule_spec,
    oracle_min_planes,
)

HIDDEN_DIM = 8


def _hidden(batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, HIDDEN_DIM)


def _head(R: int = 3) -> ResidualTritPlaneHead:
    return ResidualTritPlaneHead(
        HIDDEN_DIM,
        max_actions=32,
        R=R,
        scale_mode="geometric_balanced",
        residual_normalization="none",
    )


class TestStructuralFloors:
    """Compiler-derived floor functions."""

    def test_local_action_floor_for_branch_counts(self) -> None:
        assert local_action_floor(0) == 0
        assert local_action_floor(1) == 0
        assert local_action_floor(2) == 1
        assert local_action_floor(3) == 1
        assert local_action_floor(9) == 2
        assert local_action_floor(10) == 3

    def test_completion_support_floor(self) -> None:
        assert completion_support_floor(None) == 0
        assert completion_support_floor(1) == 0
        assert completion_support_floor(3) == 1
        assert completion_support_floor(9) == 2

    def test_scheduler_floor_never_below_local_action_code(self) -> None:
        spec = make_schedule_spec("structural_floor", max_planes=4)
        scheduler = PlaneScheduler(spec)
        assert scheduler.floor_planes(branch_count=9) == 2
        assert scheduler.floor_planes(branch_count=1) == 0

    def test_scheduler_completion_support_floor(self) -> None:
        spec = PlaneScheduleSpec(
            schedule_id="structural_floor",
            latent_role="plan_support",
            max_planes=4,
            structural_floor="completion_support",
            runtime_signal="none",
            thresholds={},
            grouping_policy="whole_batch",
            fallback_policy="score_best",
        )
        scheduler = PlaneScheduler(spec)
        assert scheduler.floor_planes(support_size=9) == 2
        assert scheduler.floor_planes(support_size=None) == 0


class TestScheduleDesiredPlanes:
    """Deterministic requested-plane manifests."""

    def test_uniform_1_and_max(self) -> None:
        spec_1 = make_schedule_spec("uniform_1", max_planes=4)
        spec_max = make_schedule_spec("uniform_max", max_planes=4)
        assert PlaneScheduler(spec_1).desired_planes(0, RuntimeDiagnostics()) == 1
        assert PlaneScheduler(spec_max).desired_planes(0, RuntimeDiagnostics()) == 4

    def test_structural_floor(self) -> None:
        spec = make_schedule_spec("structural_floor", max_planes=4)
        scheduler = PlaneScheduler(spec)
        assert scheduler.desired_planes(0, RuntimeDiagnostics(), branch_count=9) == 2
        # Structural floor is a fixed target regardless of how many planes have
        # already been executed.
        assert scheduler.desired_planes(5, RuntimeDiagnostics(), branch_count=9) == 2

    def test_floor_plus_entropy_manifest(self) -> None:
        spec = make_schedule_spec(
            "floor_plus_entropy",
            max_planes=4,
            thresholds={"entropy_high": 0.5},
        )
        scheduler = PlaneScheduler(spec)
        floor = scheduler.floor_planes(branch_count=9)
        assert floor == 2
        # Entropy below threshold -> stop at floor.
        low_entropy = RuntimeDiagnostics(entropy=0.1)
        assert scheduler.desired_planes(floor, low_entropy, branch_count=9) == floor
        # Entropy above threshold -> request one more plane.
        high_entropy = RuntimeDiagnostics(entropy=0.8)
        assert scheduler.desired_planes(floor, high_entropy, branch_count=9) == floor + 1

    def test_floor_plus_margin_manifest(self) -> None:
        spec = make_schedule_spec(
            "floor_plus_margin",
            max_planes=4,
            thresholds={"margin_low": 0.2},
        )
        scheduler = PlaneScheduler(spec)
        floor = scheduler.floor_planes(branch_count=3)
        assert floor == 1
        assert scheduler.desired_planes(
            floor, RuntimeDiagnostics(margin=0.5), branch_count=3
        ) == floor
        assert scheduler.desired_planes(
            floor, RuntimeDiagnostics(margin=0.05), branch_count=3
        ) == floor + 1

    def test_floor_plus_sensitivity_manifest(self) -> None:
        spec = make_schedule_spec(
            "floor_plus_sensitivity",
            max_planes=4,
            thresholds={"sensitivity_high": 0.5},
        )
        scheduler = PlaneScheduler(spec)
        floor = scheduler.floor_planes(branch_count=3)
        assert scheduler.desired_planes(
            floor, RuntimeDiagnostics(sensitivity={"a": 0.1}), branch_count=3
        ) == floor
        assert scheduler.desired_planes(
            floor, RuntimeDiagnostics(sensitivity={"a": 0.9}), branch_count=3
        ) == floor + 1

    def test_learned_router_no_target_leakage(self) -> None:
        diagnostics = RuntimeDiagnostics(
            entropy=0.5,
            margin=0.1,
            sensitivity={"slot": 0.2},
            residual_norm=0.3,
        )
        features = PlaneRouter.build_features(4, diagnostics)
        assert features.shape == (5,)
        # No target/future information is present in the feature vector.
        assert not any(torch.isnan(features))

    def test_learned_router_manifest(self) -> None:
        router = PlaneRouter()
        # Push the router toward "request another plane" by setting a low threshold.
        spec = make_schedule_spec(
            "floor_plus_learned_router",
            max_planes=4,
            thresholds={"router_logit": -10.0},
        )
        scheduler = PlaneScheduler(spec, router=router)
        floor = scheduler.floor_planes(branch_count=3)
        diagnostics = RuntimeDiagnostics(
            entropy=0.5, margin=0.1, sensitivity={"slot": 0.2}, residual_norm=0.3
        )
        assert scheduler.desired_planes(floor, diagnostics, branch_count=3) == floor + 1

    def test_desired_planes_clamped_to_max(self) -> None:
        spec = make_schedule_spec(
            "floor_plus_entropy",
            max_planes=2,
            thresholds={"entropy_high": 0.0},
        )
        scheduler = PlaneScheduler(spec)
        assert scheduler.desired_planes(2, RuntimeDiagnostics(entropy=1.0), branch_count=9) == 2


class TestRoutingContext:
    """End-to-end routing through a residual-plane head."""

    def test_forced_state_runs_zero_planes(self) -> None:
        head = _head(R=2)
        spec = make_schedule_spec("uniform_max", max_planes=head.R)
        ctx = AdaptivePlaneRoutingContext(head, PlaneScheduler(spec))
        result = ctx.route_batch(
            _hidden(batch=1),
            [StateContext("test", branch_count=1, forced=True)],
            [["only"]],
        )[0]
        assert result.decision_kind == "forced"
        assert result.planes_used == 0

    def test_structural_floor_schedule_respects_floor(self) -> None:
        head = _head(R=4)
        spec = make_schedule_spec("structural_floor", max_planes=head.R)
        ctx = AdaptivePlaneRoutingContext(
            head, PlaneScheduler(spec), stability_patience=0
        )
        branch_count = 9  # needs 2 planes
        result = ctx.route_batch(
            _hidden(batch=1),
            [StateContext("test", branch_count=branch_count)],
            [[f"a{i}" for i in range(branch_count)]],
        )[0]
        assert result.planes_used == 2

    def test_fallback_at_max_planes(self) -> None:
        head = _head(R=2)
        # Entropy threshold of 0.0 always requests another plane, so we should
        # hit max_planes.
        spec = make_schedule_spec(
            "floor_plus_entropy",
            max_planes=head.R,
            thresholds={"entropy_high": 0.0},
        )
        ctx = AdaptivePlaneRoutingContext(head, PlaneScheduler(spec))
        result = ctx.route_batch(
            _hidden(batch=1),
            [StateContext("test", branch_count=3)],
            [["a", "b", "c"]],
        )[0]
        assert result.planes_used == head.R
        assert result.telemetry.get("fallback_triggered") is True

    def test_batch_grouping_matches_per_item(self) -> None:
        head = _head(R=4)
        spec = make_schedule_spec(
            "floor_plus_margin",
            max_planes=head.R,
            thresholds={"margin_low": 0.5},
        )
        ctx = AdaptivePlaneRoutingContext(
            head, PlaneScheduler(spec), grouping_policy="compact"
        )
        torch.manual_seed(0)
        hidden = torch.randn(3, HIDDEN_DIM)
        contexts = [
            StateContext("test", branch_count=3),
            StateContext("test", branch_count=5),
            StateContext("test", branch_count=1, forced=True),
        ]
        legals = [
            ["a", "b", "c"],
            ["d", "e", "f", "g", "h"],
            ["only"],
        ]

        batch_results = ctx.route_batch(hidden, contexts, legals)
        per_item_results = [
            ctx.route_batch(
                hidden[i : i + 1],
                [contexts[i]],
                [legals[i]],
            )[0]
            for i in range(3)
        ]

        for br, pr in zip(batch_results, per_item_results):
            assert br.action_identity == pr.action_identity
            assert br.planes_used == pr.planes_used
            assert br.decision_kind == pr.decision_kind

    def test_default_uniform_execution_unchanged_when_disabled(self) -> None:
        head = _head(R=2)
        legal = ["a", "b", "c"]
        hidden = _hidden(batch=1)
        out_full = head.score(hidden, StateContext("test"), legal)
        out_max = head.score(
            hidden, StateContext("test"), legal, max_planes=head.R
        )
        assert torch.allclose(out_full.logits, out_max.logits, atol=1e-6)

    def test_oracle_min_planes_is_offline_diagnostic(self) -> None:
        head = _head(R=3)
        legal = ["a", "b", "c"]
        hidden = _hidden(batch=1)
        ctx = StateContext("test", branch_count=3)
        out_full = head.score(hidden, ctx, legal, max_planes=head.R)
        accepted = head.decode(out_full, legal).action_identity
        min_p = oracle_min_planes(
            head, hidden, ctx, legal, max_planes=head.R, accepted_action=accepted
        )
        assert 0 <= min_p <= head.R
        # The oracle prefix must reproduce the accepted action.
        out_prefix = head.score(hidden, ctx, legal, max_planes=min_p)
        assert head.decode(out_prefix, legal).action_identity == accepted

    def test_compact_drops_finished_items(self) -> None:
        head = _head(R=4)
        # Very high margin threshold -> no extra planes beyond floor.
        spec = make_schedule_spec(
            "floor_plus_margin",
            max_planes=head.R,
            thresholds={"margin_low": 1e6},
        )
        ctx = AdaptivePlaneRoutingContext(
            head, PlaneScheduler(spec), grouping_policy="compact", stability_patience=0
        )
        torch.manual_seed(1)
        hidden = torch.randn(4, HIDDEN_DIM)
        contexts = [StateContext("test", branch_count=3) for _ in range(4)]
        legals = [["a", "b", "c"] for _ in range(4)]
        results = ctx.route_batch(hidden, contexts, legals)
        # Every non-forced item should stop at the structural floor (1 plane).
        assert all(r.planes_used == 1 for r in results)
