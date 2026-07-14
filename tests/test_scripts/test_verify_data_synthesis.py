from scripts.verify_data_synthesis import _patch, _reverse_patch
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
