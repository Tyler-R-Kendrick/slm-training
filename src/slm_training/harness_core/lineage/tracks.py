"""Frozen recipes and base candidates for the two equal production tracks."""

from __future__ import annotations

TWOTOWER_BASE_ID = "HuggingFaceTB/SmolLM2-135M"
TWOTOWER_BASE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

# Resolved from the public Hub model records. Never replace these with ``main``.
CAUSAL_BASE_CANDIDATES = {
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
}

TWOTOWER_E53_RECIPE = {
    "model": "twotower",
    "d_model": 192,
    "n_heads": 6,
    "context_layers": 3,
    "denoiser_layers": 6,
    "context_backend": "hf",
    "hf_model_name": TWOTOWER_BASE_ID,
    "hf_model_revision": TWOTOWER_BASE_REVISION,
    "freeze_context": True,
    "output_tokenizer": "lexer",
    "use_symbol_table": True,
    "factorized_embeddings": True,
    "mask_pattern": "mixed",
    "statement_mask_prob": 0.35,
    "remask_span": "statement",
    "schema_in_context": True,
    "slot_contract_in_context": True,
    "slot_contract_constrained_decode": True,
    "honest_slot_contract": True,
    "template_fill_decode": True,
    "gen_steps": 16,
    "fidelity_loss_weight": 1.5,
    "ltr_loss_weight": 2.0,
    "grammar_ltr_repair": True,
    "grammar_ltr_primary": False,
    "mdlm_schedule": True,
    "remask_ratio": 0.12,
    "remask_policy": "combined",
    "remask_use_gate": True,
    "remask_use_entropy": True,
    "core_perturb_frac": 0.25,
    "remask_to_mask": True,
    "slot_aware_trust_gate": True,
    "trust_gate_train": True,
    "use_curriculum": True,
    "mix_curriculum": True,
    "best_of_n": 4,
    "grammar_constrained": True,
}

CAUSAL_LORA_RECIPE = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    "sequence_length": 512,
    "effective_batch_size": 32,
    "learning_rate": 2e-4,
    "scheduler": "cosine",
    "warmup_ratio": 0.03,
    "early_stopping": True,
}

TOKEN_RUNGS = (0.5, 1.0, 3.0)

# A branch loads its parent's weights and therefore cannot silently request a
# different tensor/tokenizer layout. Such work must begin as a new scratch base.
IMMUTABLE_BRANCH_RECIPE_KEYS = {
    "d_model",
    "n_heads",
    "context_layers",
    "denoiser_layers",
    "context_backend",
    "hf_model_name",
    "hf_model_revision",
    "output_tokenizer",
    "use_symbol_table",
    "factorized_embeddings",
    "rank",
    "target_modules",
}
