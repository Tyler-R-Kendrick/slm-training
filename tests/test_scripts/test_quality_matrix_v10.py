from dataclasses import asdict
from pathlib import Path

from scripts.run_quality_matrix import _v10_experiments


def test_v10_registers_matched_b4_adaptation_pair() -> None:
    rows = _v10_experiments(Path("outputs/data/train/v1"))
    assert [row.eid for row in rows] == ["E255", "E256"]
    control, adapted = rows
    assert control.denoiser_backend == "scratch"
    assert adapted.denoiser_backend == "hf"
    # Matched pair: the rows differ ONLY in the denoiser backbone (plus
    # identity fields), so any scoreboard delta is attributable to the
    # AR-adaptation lever.
    a, b = asdict(control), asdict(adapted)
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"eid", "run_id", "description", "denoiser_backend"}
    # Parallel MaskGIT decode keeps the 135M-backbone eval tractable and
    # identical across the pair.
    assert all(not row.grammar_ltr_primary for row in rows)
    assert all(row.mask_pattern == "diffusion" for row in rows)
