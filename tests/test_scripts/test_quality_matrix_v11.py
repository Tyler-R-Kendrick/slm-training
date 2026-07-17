from dataclasses import asdict
from pathlib import Path

from scripts.run_quality_matrix import _v11_experiments


def test_v11_registers_matched_b4_adaptation_pair() -> None:
    rows = _v11_experiments(Path("outputs/data/train/v1"))
    assert [row.eid for row in rows] == ["E255", "E256", "E257"]
    control, adapted, relative = rows
    assert control.denoiser_backend == "scratch"
    assert adapted.denoiser_backend == "hf"
    # Matched pairs: each lever row differs from the E255 control ONLY in its
    # single lever (plus identity fields), so any scoreboard delta is
    # attributable to that lever.
    a, b, c = asdict(control), asdict(adapted), asdict(relative)
    assert {k for k in a if a[k] != b[k]} == {
        "eid",
        "run_id",
        "description",
        "denoiser_backend",
    }
    assert {k for k in a if a[k] != c[k]} == {
        "eid",
        "run_id",
        "description",
        "bind_encoding",
    }
    assert relative.bind_encoding == "relative"
    # Parallel MaskGIT decode keeps the 135M-backbone eval tractable and
    # identical across the set.
    assert all(not row.grammar_ltr_primary for row in rows)
    assert all(row.mask_pattern == "diffusion" for row in rows)
