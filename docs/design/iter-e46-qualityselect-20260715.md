{
  "matrix": "quality-experiment-matrix-v5",
  "reference": "docs/design/quality-experiment-matrix.md",
  "gate_policy": {
    "smoke": {
      "parse_rate": 0.66,
      "structural_similarity": 0.35,
      "placeholder_fidelity": 0.25,
      "reward_score": 0.3
    },
    "held_out": {
      "parse_rate": 0.4,
      "structural_similarity": 0.3,
      "placeholder_fidelity": 0.15
    },
    "adversarial": {
      "parse_rate": 0.25,
      "structural_similarity": 0.25
    },
    "ood": {
      "parse_rate": 0.25,
      "structural_similarity": 0.25
    },
    "rico_held": {
      "parse_rate": 0.1,
      "structural_similarity": 0.2
    }
  },
  "device": "cpu",
  "batch_size": 8,
  "learning_rate": 0.0003,
  "seed": 0,
  "test_dir": "outputs/test_data/remediated",
  "design_md_in_context": false,
  "rico_eval_limit": 32,
  "suites": [
    "smoke"
  ],
  "steps": 128,
  "gen_steps": 8,
  "context_backend": "scratch",
  "matrix_set": "v5",
  "results": [
    {
      "id": "E46",
      "run_id": "qx_e46_champion",
      "pass": false,
      "failures": [
        "exception: SystemExit: 143"
      ],
      "suites": {}
    }
  ]
}
