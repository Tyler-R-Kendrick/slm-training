from dataclasses import asdict
from pathlib import Path

from scripts.run_quality_matrix import _v9_experiments, _v13_experiments


def test_v13_registers_e268_matched_to_a5_baseline() -> None:
    rows = _v13_experiments(Path("outputs/data/train/v1"))
    assert [row.eid for row in rows] == ["E268"]
    (asap,) = rows
    assert asap.asap_reweight is True
    assert asap.initialization == "eval_only"

    # Matched to the A5 lattice-campaign baseline (E240): E268 differs from the
    # E240 strict compiler-tree control ONLY by the ASAp re-weighting flag (plus
    # identity fields), so any scoreboard delta is attributable to that lever.
    e240 = next(r for r in _v9_experiments(Path("outputs/data/train/v1")) if r.eid == "E240")
    a, b = asdict(e240), asdict(asap)
    assert {k for k in a if a[k] != b[k]} == {
        "eid",
        "run_id",
        "description",
        "asap_reweight",
    }
    # ASAp is a decode-time override on the shared frozen eval-only checkpoint.
    assert "asap_reweight" in asap.runtime_override_fields
