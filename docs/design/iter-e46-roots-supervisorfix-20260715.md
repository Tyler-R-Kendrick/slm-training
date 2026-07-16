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
  "steps": 16,
  "gen_steps": 8,
  "context_backend": "scratch",
  "matrix_set": "v5",
  "results": [
    {
      "id": "E46",
      "run_id": "qx_e46_champion",
      "initialization": "scratch",
      "parent_checkpoint": null,
      "description": "V5 champion: lexer+symtable+factorized+structmask+template+MDLM",
      "honest_slot_contract": false,
      "schema_in_context": true,
      "slot_contract_in_context": true,
      "slot_contract_constrained_decode": true,
      "template_fill_decode": true,
      "grammar_ltr_primary": false,
      "grammar_ltr_repair": true,
      "effective_gen_steps": 16,
      "best_of_n": 4,
      "train_dir": "outputs/train_data/v1_curriculum",
      "train_content_fingerprint": "3445bd71a01c3941191c89f31e90ae6df4b885b7244ab51ca10c6c6de6ed4003",
      "checkpoint": "outputs/runs/iter-e46-roots-supervisorfix-20260715/qx_e46_champion/checkpoints/last.pt",
      "pass": false,
      "failures": [
        "held_out:missing_suite",
        "adversarial:missing_suite",
        "ood:missing_suite",
        "rico_held:missing_suite"
      ],
      "suites": {
        "smoke": {
          "parse_rate": 1.0,
          "placeholder_fidelity": 1.0,
          "structural_similarity": 0.6488999999999999,
          "reward_score": 0.969,
          "n": 3,
          "speculative_stats": {
            "generates": 24,
            "denoiser_forwards": 396,
            "forwards_per_generate": 16.5,
            "speculative_batches": 0,
            "speculative_canvases": 0,
            "successor_hits": 0,
            "successor_misses": 0,
            "successor_hit_rate": null,
            "clusters_proposed": 0,
            "clusters_accepted": 0,
            "clusters_rejected": 0,
            "remasked_positions": 7260
          }
        }
      }
    }
  ]
}
