# E153 — Grammar trust-model ablation (2026-07-15)

Enabling the existing `grammar_trust_model` policy knob did not change the failure: the one-record replay remained unparsable, timed out at 20 seconds, and recorded 14,322.1 ms denoiser, 4,711.6 ms picker, and 5,160 DFA/probe operations. The next harness change should target repeated denoiser/selection work directly.
