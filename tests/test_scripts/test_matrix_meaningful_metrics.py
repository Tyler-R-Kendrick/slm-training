from scripts.run_grammar_matrix import (
    _halving_score,
    _summarize_board as summarize_grammar_board,
)
from scripts.run_quality_matrix import _summarize_board as summarize_quality_board


def _board() -> dict:
    return {
        "suites": {
            "smoke": {
                "n": 4,
                "parse_rate": 1.0,
                "syntax_parse_rate": 1.0,
                "meaningful_program_rate": 0.5,
                "meaningful_program_v1_rate": 0.5,
                "binding_aware_meaningful_v2_rate_strict": 0.25,
                "binding_aware_meaningful_v2_rate_coverage_conditioned": 1 / 3,
                "binding_aware_meaningful_v2_coverage": 0.75,
                "meaningful_metric_primary": "meaningful_program_v1",
                "meaningful_metric_versions": {
                    "meaningful_program_v1": "1.0.0",
                    "binding_aware_meaningful_v2": {
                        "metric_name": "binding_aware_meaningful_v2",
                        "metric_version": "2.0.0",
                        "coverage": 0.75,
                    },
                },
                "placeholder_fidelity": 0.5,
                "structural_similarity": 0.5,
                "reward_score": 0.5,
            }
        }
    }


def test_matrix_summaries_preserve_versioned_meaningful_metrics() -> None:
    for summarize in (summarize_quality_board, summarize_grammar_board):
        metrics = summarize(_board())["suites"]["smoke"]
        assert metrics["meaningful_program_v1_rate"] == 0.5
        assert metrics["binding_aware_meaningful_v2_rate_strict"] == 0.25
        assert metrics["binding_aware_meaningful_v2_rate_coverage_conditioned"] == 1 / 3
        assert metrics["binding_aware_meaningful_v2_coverage"] == 0.75
        assert metrics["meaningful_metric_primary"] == "meaningful_program_v1"
        assert metrics["meaningful_metric_versions"]["binding_aware_meaningful_v2"][
            "metric_version"
        ] == "2.0.0"


def test_grammar_halving_uses_v1_not_syntax_or_v2() -> None:
    suites = _board()["suites"]
    suites["smoke"]["syntax_parse_rate"] = 0.0
    suites["smoke"]["binding_aware_meaningful_v2_rate_strict"] = 0.0
    assert _halving_score(suites, "smoke") == 2.75


def test_grammar_halving_supports_legacy_parse_rate() -> None:
    legacy = {"smoke": {"parse_rate": 0.5, "meaningful_program_rate": None}}
    assert _halving_score(legacy, "smoke") == 1.0
