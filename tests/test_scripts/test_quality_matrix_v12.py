from dataclasses import asdict
from pathlib import Path

from scripts.run_quality_matrix import _v11_experiments, _v12_experiments


def test_v12_registers_a2_asap_row_matched_to_e255() -> None:
    rows = _v12_experiments(Path("outputs/data/train/v1"))
    assert [row.eid for row in rows] == ["E259"]
    (asap_row,) = rows
    assert asap_row.asap_decode is True
    # Decode-only lever: the row is eval-only and must be routed through a
    # frozen E255 checkpoint (--parent) so the comparison isolates
    # asap_decode exactly.
    assert asap_row.initialization == "eval_only"
    assert "asap_decode" in (asap_row.runtime_override_fields or frozenset())

    # Matched against the v11 E255 control on every field except the lever
    # and the eval-only routing.
    (control,) = [r for r in _v11_experiments(asap_row.train_dir) if r.eid == "E255"]
    a, b = asdict(control), asdict(asap_row)
    assert {k for k in a if a[k] != b[k]} == {
        "eid",
        "run_id",
        "description",
        "asap_decode",
        "initialization",
        "runtime_override_fields",
    }
    # Same decode family as the control: parallel MaskGIT, lexer tokenizer.
    assert not asap_row.grammar_ltr_primary
    assert asap_row.output_tokenizer == "lexer"
    assert asap_row.mask_pattern == "diffusion"
