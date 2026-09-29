"""Generated identity metamorphisms; requires the declared dev dependency."""

import pytest

pytest.importorskip("hypothesis", reason="declared Hypothesis dev dependency unavailable")
from hypothesis import given, strategies as st

from tests.test_autoresearch.test_experiment_identity import design, identity


@given(st.integers(min_value=0, max_value=2**31 - 1))
def test_seed_relabeling_preserves_treatment(seed):
    d = design()
    original = identity(d)
    d["candidate"].update(seed=seed, attempt_id=f"attempt-{seed}")
    assert identity(d) == original

