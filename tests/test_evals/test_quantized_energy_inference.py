"""Regression tests for quantized energy inference comparison (CAP4-03)."""

from __future__ import annotations

import math


from slm_training.evals.quantized_energy_inference import (
    EnergyProblem,
    EnergyQuantizer,
    EnergyStage,
    LegalAction,
    ScoreSemantics,
    compare_quantized_energy_inference,
    evaluate_format,
)
from slm_training.models.quantization.formats import (
    binary_format,
    fp16_format,
    int4_format,
    ternary_format,
)


def _problem(energies: list[list[float]]) -> EnergyProblem:
    stages = []
    for i, row in enumerate(energies):
        actions = tuple(
            LegalAction(action_id=f"s{i}-a{j}", local_energy=e)
            for j, e in enumerate(row)
        )
        stages.append(EnergyStage(stage_id=f"s{i}", actions=actions))
    return EnergyProblem(
        problem_id="test",
        stages=tuple(stages),
        semantics=ScoreSemantics.ADDITIVE_EDGE,
    )


def test_greedy_and_exact_match_when_local_optima_compose() -> None:
    problem = _problem([[1.0, 2.0], [1.0, 3.0]])
    result = compare_quantized_energy_inference(problem, formats=(fp16_format(),))
    fr = result.format_results[0]
    assert fr.greedy.path == fr.exact.path == ("s0-a0", "s1-a0")
    assert math.isclose(fr.exact.total_quantized_energy, 2.0)


def test_exact_enumerates_all_paths() -> None:
    # For independent additive stages greedy and exact agree, but exact still
    # enumerates the full path space and reports the true optimum/tie class.
    problem = _problem([[0.1, 0.2], [10.0, 0.0]])
    result = compare_quantized_energy_inference(problem, formats=(fp16_format(),))
    fr = result.format_results[0]
    assert fr.greedy.path == fr.exact.path
    assert fr.exact.path_count_considered == 4


def test_binary_quantization_collapses_scores() -> None:
    problem = _problem([[0.1, 0.2, 0.3, 0.4]])
    result = compare_quantized_energy_inference(problem, formats=(binary_format(),))
    fr = result.format_results[0]
    # All positive energies collapse to one binary level; greedy and exact tie.
    assert fr.greedy.total_quantized_energy == fr.exact.total_quantized_energy


def test_tie_class_size_detected() -> None:
    # Two symmetric paths with identical energies: exact should report a tie.
    problem = _problem([[1.0, 1.0], [2.0, 2.0]])
    result = compare_quantized_energy_inference(problem, formats=(fp16_format(),))
    fr = result.format_results[0]
    assert fr.exact.tie_class_size == 4


def test_quantizer_preserves_membership() -> None:
    fmt = ternary_format()
    quantizer = EnergyQuantizer.calibrate(fmt, [1.0, -1.0, 0.5])
    assert quantizer.scale > 0
    # Energies map to one of the ternary levels scaled.
    q = quantizer.quantize(0.9)
    assert any(math.isclose(q, level * quantizer.scale) for level in (-1.0, 0.0, 1.0))


def test_unknown_action_excluded_from_totals() -> None:
    stage = EnergyStage(
        stage_id="s0",
        actions=(
            LegalAction(action_id="a", local_energy=1.0, known=True),
            LegalAction(action_id="b", local_energy=0.0, known=False),
        ),
    )
    problem = EnergyProblem(problem_id="unk", stages=(stage,))
    result = evaluate_format(problem, EnergyQuantizer.calibrate(fp16_format(), [1.0]))
    assert result.greedy.path == ("a",)
    assert result.exact.path == ("a",)


def test_compare_includes_all_default_formats() -> None:
    problem = _problem([[1.0, 2.0], [3.0, 4.0]])
    result = compare_quantized_energy_inference(problem)
    ids = {fr.quantizer.fmt.format_id for fr in result.format_results}
    assert ids == {
        "fp16",
        "binary",
        "ternary",
        "symmetric_four_level",
        "learned_four_level_zero",
        "int4",
    }


def test_round_trip_dict() -> None:
    problem = _problem([[1.0, 2.0]])
    result = compare_quantized_energy_inference(problem, formats=(int4_format(),))
    data = result.to_dict()
    assert data["problem"]["problem_id"] == "test"
    assert data["format_results"][0]["greedy"]["mode"] == "greedy_local"
    assert data["format_results"][0]["exact"]["mode"] == "exact_viterbi"
