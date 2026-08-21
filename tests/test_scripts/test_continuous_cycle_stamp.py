"""The continuous closeout must stamp cycle records with the eval-comparability
version components, so their deltas land in a real cross-version partition of
the evidence ledger instead of the catch-all ``unstamped`` bucket.

Torch-free: exercises only the closeout payload builder and the ledger miner.
"""

from __future__ import annotations

from types import SimpleNamespace

import scripts.run_autotrain_continuous as driver
from slm_training.autoresearch.evidence_ledger import (
    EVAL_KEY_COMPONENTS,
    eval_key_from_stamp,
    extract_observations,
)


def _handoff() -> SimpleNamespace:
    # `_render_continuous_cycle_docs` only reads these attributes off the
    # handoff; a lightweight stand-in keeps the test independent of the full
    # AutotrainCycleHandoffV1 construction (and of torch).
    return SimpleNamespace(
        cycle_index=7,
        cycle_role="screening",
        cycle_intent="screening",
        primary_metric="smoke.structural_similarity",
        evidence_class="fixture",
        reasons=[],
    )


def _delivery() -> dict:
    return {
        "positive": False,
        "stack_layer": False,
        "measurement_complete": True,
        "primary_metric": "smoke.structural_similarity",
        "control_metrics": {"smoke.structural_similarity": 0.0575},
        "candidate_metrics": {"smoke.structural_similarity": 0.1742},
        # candidate id token carries the arm slug the miner recovers.
        "reasons": [
            "quality_metric_win: c20260810-demo-c7-container-close "
            "structural_similarity control=0.0575 candidate=0.1742"
        ],
    }


def test_cycle_record_carries_eval_stamp() -> None:
    _md, payload = driver._render_continuous_cycle_docs(
        campaign_id="c20260810-demo-c7",
        loop_id="loop-demo",
        handoff=_handoff(),
        delivery=_delivery(),
    )
    assert payload["schema"] == "continuous_cycle_results/v1"
    hill = payload.get("hillclimb")
    assert isinstance(hill, dict)
    assert "went_well" in hill and "went_wrong" in hill and "speculate" in hill
    stamp = payload.get("version_stamp")
    assert isinstance(stamp, dict), "cycle record must carry a version_stamp"
    assert stamp.get("stamp_schema")
    for cid in EVAL_KEY_COMPONENTS:
        assert cid in stamp["components"], f"missing eval component {cid}"

    key = eval_key_from_stamp(payload)
    assert key is not None, "stamped record must resolve a non-null eval_key"
    for cid in EVAL_KEY_COMPONENTS:
        assert f"{cid}=" in key


def test_stamped_record_mines_into_a_real_partition() -> None:
    """End-to-end: the closeout payload, fed to the ledger miner, produces an
    observation whose eval_key is the stamped partition — never ``unstamped``."""
    _md, payload = driver._render_continuous_cycle_docs(
        campaign_id="c20260810-demo-c7",
        loop_id="loop-demo",
        handoff=_handoff(),
        delivery=_delivery(),
    )
    observations = extract_observations(payload, source="test")
    assert observations, "miner should recover an observation from the record"
    obs = observations[0]
    assert obs.slug == "container-close"
    assert obs.eval_key is not None and obs.eval_key != "unstamped"
    assert obs.eval_key == eval_key_from_stamp(payload)
    # the delta is the real screening delta, carried into the right partition.
    assert obs.delta is not None and abs(obs.delta - (0.1742 - 0.0575)) < 1e-9
