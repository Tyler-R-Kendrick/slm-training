"""Characterization pin for the ship-gate payload (harness-core extraction).

The inputs and expected payloads below were captured from
``evaluate_ship_gates`` on 2026-07-19 *before* the generic check loop moved to
``slm_training.harness_core.gate_engine``. The OpenUI policy owner
(``slm_training.harnesses.model_build.ship_gates``) must keep producing these
payloads byte-for-byte; any diff means the extraction stopped being purely
structural (docs/design/harness-core.md).
"""

from __future__ import annotations

import json

from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates

CASES = json.loads(r"""
{
  "default_mixed": {
    "thresholds": null,
    "suites": {
      "smoke": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41,
        "fallback_count": 0
      },
      "held_out": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.11,
        "component_type_recall": 0.48,
        "reward_score": 0.41,
        "fallback_count": 0
      }
    }
  },
  "default_fallback_and_legacy": {
    "thresholds": null,
    "suites": {
      "smoke": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41,
        "fallback_count": null
      },
      "held_out": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41,
        "fallback_count": 3
      },
      "adversarial": {
        "n": 4,
        "parse_rate": 0.5,
        "fallback_count": 0
      },
      "ood": {
        "n": null,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41,
        "fallback_count": 0
      },
      "rico_held": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41,
        "fallback_count": 0
      }
    }
  },
  "custom_thresholds": {
    "thresholds": {
      "smoke": {
        "meaningful_program_rate": 0.5,
        "min_n": 10,
        "reward_score": 0.2
      },
      "extra_suite": {
        "structural_similarity": 0.4
      }
    },
    "suites": {
      "smoke": {
        "n": 12,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41,
        "fallback_count": 0
      }
    }
  }
}
""")

EXPECTED = json.loads(r"""
{
  "default_mixed": {
    "policy": {
      "smoke": {
        "meaningful_program_rate": 0.66,
        "structural_similarity": 0.35,
        "component_type_recall": 0.35,
        "placeholder_fidelity": 0.25,
        "reward_score": 0.3
      },
      "held_out": {
        "meaningful_program_rate": 0.4,
        "structural_similarity": 0.3,
        "component_type_recall": 0.3,
        "placeholder_fidelity": 0.15
      },
      "adversarial": {
        "meaningful_program_rate": 0.25,
        "structural_similarity": 0.25,
        "component_type_recall": 0.2
      },
      "ood": {
        "meaningful_program_rate": 0.25,
        "structural_similarity": 0.25,
        "component_type_recall": 0.2
      },
      "rico_held": {
        "meaningful_program_rate": 0.1,
        "structural_similarity": 0.2,
        "component_type_recall": 0.15
      }
    },
    "meaningful_metric_policy": {
      "active_primary": "meaningful_program_v1",
      "threshold_version": "openui_ship_gates_v1",
      "meaningful_program_v1": {
        "version": "1.0.0",
        "wire_field": "meaningful_program_rate",
        "thresholds": "DEFAULT_SHIP_GATES"
      },
      "binding_aware_meaningful_v2": {
        "version": "2.0.0",
        "thresholds": null,
        "status": "candidate_pending_calibration"
      }
    },
    "actual": {
      "smoke": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41
      },
      "held_out": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.11,
        "component_type_recall": 0.48,
        "reward_score": 0.41
      }
    },
    "gates": {
      "smoke:certified_fallback": true,
      "smoke:insufficient_n": true,
      "smoke:meaningful_program_rate": true,
      "smoke:structural_similarity": true,
      "smoke:component_type_recall": true,
      "smoke:placeholder_fidelity": true,
      "smoke:reward_score": true,
      "held_out:certified_fallback": true,
      "held_out:insufficient_n": true,
      "held_out:meaningful_program_rate": true,
      "held_out:structural_similarity": false,
      "held_out:component_type_recall": true,
      "held_out:placeholder_fidelity": true,
      "adversarial:missing_suite": false,
      "ood:missing_suite": false,
      "rico_held:missing_suite": false
    },
    "failures": [
      "held_out:structural_similarity actual=0.11 need>=0.3",
      "adversarial:missing_suite",
      "ood:missing_suite",
      "rico_held:missing_suite"
    ],
    "pass": false,
    "note": "Honest ship gates require all policy suites and score structure only (meaningful_program_rate / structural_similarity / component_type_recall / placeholder_fidelity / reward_score). component_type_recall is the semantic-density floor: shorter-but-emptier output cannot pass on syntax alone. Syntax parse is reported separately and is not a learned-quality substitute. DESIGN.md style lint is never a ship gate. See docs/design/adversarial-review.md and docs/design/structure-only-eval.md."
  },
  "default_fallback_and_legacy": {
    "policy": {
      "smoke": {
        "meaningful_program_rate": 0.66,
        "structural_similarity": 0.35,
        "component_type_recall": 0.35,
        "placeholder_fidelity": 0.25,
        "reward_score": 0.3
      },
      "held_out": {
        "meaningful_program_rate": 0.4,
        "structural_similarity": 0.3,
        "component_type_recall": 0.3,
        "placeholder_fidelity": 0.15
      },
      "adversarial": {
        "meaningful_program_rate": 0.25,
        "structural_similarity": 0.25,
        "component_type_recall": 0.2
      },
      "ood": {
        "meaningful_program_rate": 0.25,
        "structural_similarity": 0.25,
        "component_type_recall": 0.2
      },
      "rico_held": {
        "meaningful_program_rate": 0.1,
        "structural_similarity": 0.2,
        "component_type_recall": 0.15
      }
    },
    "meaningful_metric_policy": {
      "active_primary": "meaningful_program_v1",
      "threshold_version": "openui_ship_gates_v1",
      "meaningful_program_v1": {
        "version": "1.0.0",
        "wire_field": "meaningful_program_rate",
        "thresholds": "DEFAULT_SHIP_GATES"
      },
      "binding_aware_meaningful_v2": {
        "version": "2.0.0",
        "thresholds": null,
        "status": "candidate_pending_calibration"
      }
    },
    "actual": {
      "smoke": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41
      },
      "held_out": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41
      },
      "adversarial": {
        "n": 4,
        "parse_rate": 0.5,
        "meaningful_program_rate": 0.5,
        "meaningful_program_v1_rate": null,
        "binding_aware_meaningful_v2_rate_strict": null,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": null,
        "binding_aware_meaningful_v2_coverage": null,
        "syntax_parse_rate": null,
        "placeholder_fidelity": null,
        "placeholder_validity": null,
        "structural_similarity": null,
        "component_type_recall": null,
        "reward_score": null
      },
      "ood": {
        "n": null,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41
      },
      "rico_held": {
        "n": 25,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41
      }
    },
    "gates": {
      "smoke:certified_fallback": false,
      "smoke:insufficient_n": true,
      "smoke:meaningful_program_rate": true,
      "smoke:structural_similarity": true,
      "smoke:component_type_recall": true,
      "smoke:placeholder_fidelity": true,
      "smoke:reward_score": true,
      "held_out:certified_fallback": false,
      "held_out:insufficient_n": true,
      "held_out:meaningful_program_rate": true,
      "held_out:structural_similarity": true,
      "held_out:component_type_recall": true,
      "held_out:placeholder_fidelity": true,
      "adversarial:certified_fallback": true,
      "adversarial:insufficient_n": false,
      "adversarial:meaningful_program_rate": true,
      "adversarial:structural_similarity": false,
      "adversarial:component_type_recall": false,
      "ood:certified_fallback": true,
      "ood:insufficient_n": false,
      "ood:meaningful_program_rate": true,
      "ood:structural_similarity": true,
      "ood:component_type_recall": true,
      "rico_held:certified_fallback": true,
      "rico_held:insufficient_n": true,
      "rico_held:meaningful_program_rate": true,
      "rico_held:structural_similarity": true,
      "rico_held:component_type_recall": true
    },
    "failures": [
      "smoke:certified_fallback unmeasured (fallback_count absent) need=0 for learned-quality claims",
      "held_out:certified_fallback actual=3 need=0 for learned-quality claims",
      "adversarial:insufficient_n actual=4 need>=20",
      "adversarial:structural_similarity actual=None need>=0.25",
      "adversarial:component_type_recall actual=None need>=0.2",
      "ood:insufficient_n actual=None need>=20"
    ],
    "pass": false,
    "note": "Honest ship gates require all policy suites and score structure only (meaningful_program_rate / structural_similarity / component_type_recall / placeholder_fidelity / reward_score). component_type_recall is the semantic-density floor: shorter-but-emptier output cannot pass on syntax alone. Syntax parse is reported separately and is not a learned-quality substitute. DESIGN.md style lint is never a ship gate. See docs/design/adversarial-review.md and docs/design/structure-only-eval.md."
  },
  "custom_thresholds": {
    "policy": {
      "smoke": {
        "meaningful_program_rate": 0.5,
        "min_n": 10,
        "reward_score": 0.2
      },
      "extra_suite": {
        "structural_similarity": 0.4
      }
    },
    "meaningful_metric_policy": {
      "active_primary": "meaningful_program_v1",
      "threshold_version": "custom:dbd82a4d723df42a18568035edb42b92daefa49ca8a15d89bbe0c59b489c9692",
      "meaningful_program_v1": {
        "version": "1.0.0",
        "wire_field": "meaningful_program_rate",
        "thresholds": "request_thresholds"
      },
      "binding_aware_meaningful_v2": {
        "version": "2.0.0",
        "thresholds": null,
        "status": "candidate_pending_calibration"
      }
    },
    "actual": {
      "smoke": {
        "n": 12,
        "parse_rate": 0.91,
        "meaningful_program_rate": 0.72,
        "meaningful_program_v1_rate": 0.72,
        "binding_aware_meaningful_v2_rate_strict": 0.55,
        "binding_aware_meaningful_v2_rate_coverage_conditioned": 0.61,
        "binding_aware_meaningful_v2_coverage": 0.83,
        "syntax_parse_rate": 0.93,
        "placeholder_fidelity": 0.44,
        "placeholder_validity": 0.87,
        "structural_similarity": 0.52,
        "component_type_recall": 0.48,
        "reward_score": 0.41
      }
    },
    "gates": {
      "smoke:certified_fallback": true,
      "smoke:insufficient_n": true,
      "smoke:meaningful_program_rate": true,
      "smoke:reward_score": true,
      "extra_suite:missing_suite": false
    },
    "failures": [
      "extra_suite:missing_suite"
    ],
    "pass": false,
    "note": "Honest ship gates require all policy suites and score structure only (meaningful_program_rate / structural_similarity / component_type_recall / placeholder_fidelity / reward_score). component_type_recall is the semantic-density floor: shorter-but-emptier output cannot pass on syntax alone. Syntax parse is reported separately and is not a learned-quality substitute. DESIGN.md style lint is never a ship gate. See docs/design/adversarial-review.md and docs/design/structure-only-eval.md."
  }
}
""")


def test_case_names_match() -> None:
    assert set(CASES) == set(EXPECTED)


def test_payloads_match_pre_extraction_golden() -> None:
    for name, case in CASES.items():
        payload = evaluate_ship_gates(case["suites"], thresholds=case["thresholds"])
        assert payload == EXPECTED[name], f"payload drifted for case {name!r}"


def test_payload_json_bytes_stable() -> None:
    """Key order and value formatting must survive, not just dict equality."""
    for name, case in CASES.items():
        payload = evaluate_ship_gates(case["suites"], thresholds=case["thresholds"])
        got = json.dumps(payload, indent=2)
        want = json.dumps(EXPECTED[name], indent=2)
        assert got == want, f"serialized payload drifted for case {name!r}"
