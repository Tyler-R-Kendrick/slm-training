# CAP5 artifact / reproduction index

Machine-readable source: [cap5-artifact-index.json](cap5-artifact-index.json).

| Artifact ID | Kind | Path | Generator | Reproduction command |
| --- | --- | --- | --- | --- |
| cap0-02-arity-analyzer | exact_certificate | `docs/design/cap0-02-arity-analyzer-20260718.json` | scripts/analyze_grammar_arity.py | `python -m scripts.analyze_grammar_arity --fixture bounded-expr --max-ast-nodes 6 --max-live-bindings 2 --dimensions 4 --out outputs/runs/arity/report.json` |
| cap0-03-coding-precision | exact_certificate | `docs/design/cap0-03-coding-precision-20260718.md` | tests/test_dsl/test_arity_coding.py | `pytest tests/test_dsl/test_arity_coding.py -q` |
| cap1-03-task-quotient | exact_certificate | `docs/design/cap1-03-task-quotient-20260718.md` | scripts/analyze_task_quotient.py | `python -m scripts.analyze_task_quotient --help` |
| cap1-04-conditional-rate | exact_certificate | `docs/design/cap1-04-conditional-rate-20260718.md` | scripts/analyze_conditional_rate.py | `python -m scripts.analyze_conditional_rate --help` |
| cap2-bottleneck-results | experiment_matrix | `docs/design/cap2-bottleneck-results.json` | scripts/run_cap2_bottleneck.py | `python -m scripts.run_cap2_bottleneck --out outputs/runs/cap2_bottleneck` |
| cap2-04-state-ablation | experiment_matrix | `docs/design/iter-cap2-04-state-ablation-20260718.json` | scripts/run_cap2_04_state_ablation.py | `python -m scripts.run_cap2_04_state_ablation --out outputs/runs/cap2_state_ablation` |
| cap3-03-ternary-falsification | experiment_matrix | `docs/design/iter-cap3-03-ternary-falsification-20260718.json` | scripts/run_cap3_03_ternary_falsification.py | `python -m scripts.run_cap3_03_ternary_falsification --out outputs/runs/cap3_ternary` |
| cap3-04-sensitivity | experiment_matrix | `docs/design/iter-cap3-04-sensitivity-20260718.json` | scripts/profile_quant_sensitivity.py, scripts/allocate_mixed_precision.py | `python -m scripts.profile_quant_sensitivity --out outputs/runs/quant_sensitivity && python -m scripts.allocate_mixed_precision --report outputs/runs/quant_sensitivity/report.json --out outputs/runs/mixed_precision` |
| cap3-05-equal-byte-ladder | experiment_matrix | `docs/design/iter-cap3-05-equal-byte-ladder-20260718.json` | scripts/run_scaling_ladder.py | `python -m scripts.run_scaling_ladder --out outputs/runs/cap3_ladder` |
| cap4-01-residual-quantization | experiment_matrix | `docs/design/iter-cap4-01-residual-quantization-20260718.json` | scripts/run_residual_trit_fixture.py | `python -m scripts.run_residual_trit_fixture --out outputs/runs/cap4_residual` |
| cap4-02-adaptive-plane-routing | experiment_matrix | `docs/design/iter-cap4-02-adaptive-plane-routing-20260718.json` | scripts/run_adaptive_plane_fixture.py | `python -m scripts.run_adaptive_plane_fixture --out outputs/runs/cap4_adaptive` |
| cap4-03-quantized-energy-inference | experiment_matrix | `docs/design/iter-cap4-03-quantized-energy-inference-20260719.json` | scripts/run_quantized_energy_inference_fixture.py | `python -m scripts.run_quantized_energy_inference_fixture --out outputs/runs/cap4_energy` |
| cap4-04-block-sparsity | experiment_matrix | `docs/design/iter-cap4-04-block-sparsity-20260718.json` | scripts/run_block_sparsity_fixture.py | `python -m scripts.run_block_sparsity_fixture --out outputs/runs/cap4_sparsity` |
| cap4-05-quotient-diffusion-graph | experiment_matrix | `docs/design/iter-cap4-05-quotient-diffusion-graph-20260718.json` | scripts/run_quotient_diffusion_fixture.py | `python -m scripts.run_quotient_diffusion_fixture --out outputs/runs/cap4_diffusion` |
| cap5-repro-summary | reproduction_summary | `outputs/runs/cap_repro/cap_repro_summary.json` | scripts/reproduce_calculated_arity_fixtures.py | `python -m scripts.reproduce_calculated_arity_fixtures --out outputs/runs/cap_repro --verify-expected` |
