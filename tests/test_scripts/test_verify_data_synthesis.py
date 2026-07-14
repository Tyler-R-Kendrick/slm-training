from scripts.verify_data_synthesis import (
    MATCHED_MATRIX_KEYS,
    MATCHED_RESULT_KEYS,
    _matched_quality_check,
    _patch,
    _reverse_patch,
)
from slm_training.data.edits import apply_patch


def test_patch_round_trip_from_record_metadata() -> None:
    before = 'root = Stack([cta])\ncta = Button(":cta.old")'
    patch = _patch(
        {
            "operations": [
                {
                    "kind": "replace",
                    "name": "cta",
                    "before": 'Button(":cta.old")',
                    "after": 'Button(":cta.new")',
                }
            ]
        }
    )
    assert apply_patch(before, patch) == (
        'root = Stack([cta])\ncta = Button(":cta.new")'
    )
    assert apply_patch(apply_patch(before, patch), _reverse_patch(patch)) == before


def test_add_patch_round_trip_preserves_insertion_index() -> None:
    before = 'root = Stack([a])\na = TextContent(":a")'
    patch = _patch(
        {
            "collect_unreachable": False,
            "operations": [
                {
                    "kind": "add",
                    "name": "b",
                    "after": 'TextContent(":b")',
                    "index": 1,
                }
            ],
        }
    )

    assert apply_patch(apply_patch(before, patch), _reverse_patch(patch)) == before


def test_remove_patch_round_trip_uses_previous_index() -> None:
    before = 'root = Stack([a])\nunused = TextContent(":unused")\na = TextContent(":a")'
    patch = _patch(
        {
            "collect_unreachable": False,
            "operations": [
                {
                    "kind": "remove",
                    "name": "unused",
                    "before": 'TextContent(":unused")',
                    "previous_index": 1,
                }
            ],
        }
    )

    assert apply_patch(apply_patch(before, patch), _reverse_patch(patch)) == before


def _matrix_summary(*, experiment: str, fingerprint: str, held: float, rico: float):
    summary = {key: "same" for key in MATCHED_MATRIX_KEYS}
    result = {key: "same" for key in MATCHED_RESULT_KEYS}
    result.update(
        {
            "id": experiment,
            "train_content_fingerprint": fingerprint,
            "suites": {
                "held_out": {"n": 5, "placeholder_fidelity": held},
                "rico_held": {"n": 4, "placeholder_fidelity": rico},
            },
        }
    )
    summary["results"] = [result]
    return summary


def test_matched_quality_requires_same_experiment_and_both_suite_gains() -> None:
    baseline = _matrix_summary(
        experiment="E53", fingerprint="fixture", held=0.0, rico=0.0
    )
    champion = _matrix_summary(
        experiment="E53", fingerprint="integrated", held=0.2, rico=0.1
    )
    assert _matched_quality_check(baseline, champion)["status"] == "pass"

    champion["results"][0]["id"] = "E50"
    assert _matched_quality_check(baseline, champion)["status"] == "fail"
    champion["results"][0]["id"] = "E53"
    champion["results"][0]["suites"]["rico_held"]["placeholder_fidelity"] = 0.0
    assert _matched_quality_check(baseline, champion)["status"] == "fail"
