"""Tests for CAP1-04 conditional task rate and posterior diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.analysis.arity.conditional_rate import (
    TaskDistortionSpec,
    analyze_conditional_rate,
    blahut_arimoto_rate_distortion,
    conditional_entropy,
    entropy,
    fano_lower_bound,
    mutual_information,
    posterior_effective_support,
)
from slm_training.dsl.analysis.arity.task_quotient import AlignedActionRecord


def _records_for_state(
    fingerprint: str, actions: dict[str, int]
) -> list[AlignedActionRecord]:
    records: list[AlignedActionRecord] = []
    total = sum(actions.values())
    for action_id, count in actions.items():
        records.append(
            AlignedActionRecord(
                state_fingerprint=fingerprint,
                action_id=action_id,
                aligned_family=action_id,
                probability=count / total,
            )
        )
    return records


def test_entropy_uniform() -> None:
    dist = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    assert entropy(dist) == 2.0


def test_entropy_deterministic() -> None:
    assert entropy({"a": 1.0}) == 0.0


def test_conditional_entropy_forced_actions() -> None:
    # Two states, each with a single forced action -> H(A|Q) = 0.
    conditional = {"s1": {"a": 1.0}, "s2": {"b": 1.0}}
    assert conditional_entropy(conditional) == 0.0


def test_conditional_entropy_mixed_state() -> None:
    # One state uniform over 4 actions; one forced.
    conditional = {"s1": {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}, "s2": {"a": 1.0}}
    # Equal empirical weighting -> 0.5 * 2 + 0.5 * 0 = 1.0
    assert conditional_entropy(conditional) == 1.0


def test_mutual_information_independence() -> None:
    # Uniform 2x2 joint -> I(X;Y) = 0.
    joint = {
        ("x1", "y1"): 0.25,
        ("x1", "y2"): 0.25,
        ("x2", "y1"): 0.25,
        ("x2", "y2"): 0.25,
    }
    assert mutual_information(joint) == 0.0


def test_mutual_information_perfect_correlation() -> None:
    joint = {("x1", "y1"): 0.5, ("x2", "y2"): 0.5}
    assert mutual_information(joint) == 1.0


def test_fano_bound_zero_entropy() -> None:
    bound = fano_lower_bound(0.0, 4)
    assert bound.lower_bound_error == 0.0
    assert bound.exact is True


def test_fano_bound_large_entropy() -> None:
    # Uniform over 4 actions -> H = 2. Bound should be positive.
    bound = fano_lower_bound(2.0, 4)
    assert bound.lower_bound_error > 0.0
    assert bound.lower_bound_error <= 1.0


def test_posterior_effective_support() -> None:
    conditional = {
        "s1": {"a": 0.5, "b": 0.5},  # exp(1) = 2.0
        "s2": {"a": 1.0},  # 1.0
    }
    support = posterior_effective_support(conditional)
    assert support.per_state["s1"] == 2.0
    assert support.per_state["s2"] == 1.0
    assert support.mean == 1.5
    assert support.min == 1.0
    assert support.max == 2.0


def test_blahut_arimoto_endpoints() -> None:
    # Two-source Bernoulli(0.5), reproduction = {0, 1}, Hamming distortion.
    source = {"0": 0.5, "1": 0.5}
    repro = ["0", "1"]
    distortion = {
        ("0", "0"): 0.0,
        ("0", "1"): 1.0,
        ("1", "0"): 1.0,
        ("1", "1"): 0.0,
    }
    points = blahut_arimoto_rate_distortion(
        source, distortion, repro, betas=[0.001, 1.0, 100.0]
    )
    assert len(points) >= 2
    # Low beta -> low distortion, high rate (near 1 bit).
    low_dist = points[0]
    assert low_dist.distortion < 0.1
    assert low_dist.rate_bits > 0.9
    # High beta -> higher distortion, lower rate (near 0 bits).
    high_dist = points[-1]
    assert high_dist.distortion > 0.4
    assert high_dist.rate_bits < 0.1


def test_blahut_arimoto_monotonic() -> None:
    source = {"a": 0.6, "b": 0.4}
    repro = ["a", "b"]
    distortion = {
        ("a", "a"): 0.0,
        ("a", "b"): 1.0,
        ("b", "a"): 1.0,
        ("b", "b"): 0.0,
    }
    points = blahut_arimoto_rate_distortion(
        source, distortion, repro, betas=[0.01, 0.1, 1.0, 10.0, 100.0]
    )
    rates = [p.rate_bits for p in points]
    dists = [p.distortion for p in points]
    # After Pareto pruning, rate is non-increasing as distortion increases.
    assert rates == sorted(rates, reverse=True)
    assert dists == sorted(dists)


def test_analyze_conditional_rate_report_keys() -> None:
    records: list[AlignedActionRecord] = []
    records.extend(_records_for_state("s1", {"a": 1, "b": 1}))
    records.extend(_records_for_state("s2", {"a": 1, "c": 1}))
    spec = TaskDistortionSpec(spec_id="test", policy_metric="tv", policy_tolerance=0.5)
    report = analyze_conditional_rate(records, spec)
    data = report.to_dict()
    assert set(data) >= {
        "spec",
        "state_count",
        "action_alphabet_size",
        "conditional_entropy_bits",
        "mutual_information_bits",
        "fano_bounds",
        "posterior_support",
        "rate_distortion_curve",
        "estimated",
    }
    assert report.state_count == 2
    assert report.action_alphabet_size == 3


def test_cli_smoke(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "state_fingerprint": f"s{i}",
                    "action_id": chr(ord("a") + i),
                }
            )
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    md = tmp_path / "report.md"
    from scripts.analyze_conditional_rate import main

    rc = main(
        [
            "--records",
            str(records_path),
            "--out",
            str(out),
            "--markdown-out",
            str(md),
            "--policy-metric",
            "tv",
            "--policy-tolerance",
            "0.0",
        ]
    )
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["state_count"] == 2
    assert md.exists()
