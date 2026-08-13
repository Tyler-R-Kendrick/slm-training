"""Timeout retirements keep data-generation in both identity hashes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from slm_training.autoresearch.climb_policy import load_climb_policy

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_autotrain_continuous.py"
)
_SPEC = importlib.util.spec_from_file_location("run_autotrain_continuous", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

_GENERATION = {"unique_root_target": 8, "data_only": True}


def test_timeout_retirement_matches_current_data_generation(tmp_path: Path) -> None:
    root = tmp_path / "autoresearch"
    loop_id = "loop-1"
    campaign_id = "continuous-loop-test-c1"
    camp = root / campaign_id
    camp.mkdir(parents=True)
    control_id = "c-test-control"
    candidate_id = "c-test-data"
    (camp / "campaign.json").write_text(
        json.dumps({"campaign_id": campaign_id, "loop_id": loop_id})
    )
    (camp / "matrix-proposal.json").write_text(
        json.dumps(
            {
                "hypotheses": [
                    {
                        "experiment": {
                            "experiment_id": control_id,
                            "knobs": {
                                "train_version": "wf_smoke_v2",
                                "eval_version": "e_test",
                                "binder_arity_loss_weight": 0.0,
                                "binder_arity_decode_weight": 0.0,
                            },
                        }
                    },
                    {
                        "experiment": {
                            "experiment_id": candidate_id,
                            "knobs": {
                                "train_version": "wf_smoke_v2",
                                "eval_version": "e_test",
                                "binder_arity_loss_weight": 1.0,
                                "binder_arity_decode_weight": 1.0,
                                "data_generation": _GENERATION,
                            },
                        }
                    },
                ]
            }
        )
    )
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "cycle_index": 1,
                "candidate_id": candidate_id,
                "control_id": control_id,
            }
        )
    )
    (camp / "cycle_handoff.json").write_text(
        json.dumps(
            {
                "loop_id": loop_id,
                "cycle_index": 1,
                "cycle_intent": "retry_measurement",
                "primary_metric": "smoke.structural_similarity",
                "reasons": [f"candidate_runtime_unblock_reproduced:{candidate_id}"],
            }
        )
    )

    policy = load_climb_policy()
    kwargs = dict(
        policy=policy,
        train_version="wf_smoke_v2",
        eval_version="e_test",
        primary_metric="smoke.structural_similarity",
        direction="increase",
        claim_class="diagnostic",
    )
    matched, _ = _mod._sync_reproduced_timeout_retirements(
        root, loop_id, campaign_id, data_generation=_GENERATION, **kwargs
    )
    assert matched == {"binder-arity"}

    mismatched, _ = _mod._sync_reproduced_timeout_retirements(
        root, loop_id, campaign_id, data_generation=None, **kwargs
    )
    assert mismatched == set()
