"""B4 AR→masked-denoiser adaptation (HFDenoiserTower) regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

HERO = (
    'root = Stack([b3], "column")\n'
    'b1 = TextContent(":slot_0")\n'
    'b2 = TextContent(":slot_1")\n'
    "b3 = Card([b1, b2])"
)
CTA = 'root = Stack([b1])\nb1 = Button(":slot_0")'


def _require_tiny_llama() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from transformers import AutoModel

        AutoModel.from_pretrained(TINY_LLAMA)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"HF tiny model unavailable: {exc}")


def test_hf_denoiser_is_bidirectional_and_interface_complete() -> None:
    _require_tiny_llama()
    import torch

    from slm_training.models.hf_denoiser import HFDenoiserTower

    torch.manual_seed(0)
    vocab, d_model, tgt, ctx_len = 23, 16, 6, 3
    tower = HFDenoiserTower(
        vocab_size=vocab, d_model=d_model, max_len=32, hf_model_name=TINY_LLAMA
    )
    tower.eval()
    noisy = torch.randint(1, vocab, (2, tgt))
    ctx = torch.randn(2, ctx_len, d_model)
    with torch.no_grad():
        logits = tower(noisy, ctx, pad_id=0)
        assert logits.shape == (2, tgt, vocab)
        # Bidirectional: changing the LAST target token must move logits at
        # the FIRST position — a causal backbone would leave them unchanged.
        perturbed = noisy.clone()
        perturbed[:, -1] = (perturbed[:, -1] + 1) % (vocab - 1) + 1
        delta = (tower(perturbed, ctx, pad_id=0)[:, 0, :] - logits[:, 0, :]).abs()
        assert float(delta.max()) > 1e-8
        # encode/project split used by the constrained decode fast paths.
        hidden = tower.encode(noisy, ctx, pad_id=0)
        candidates = torch.tensor([1, 2, 3])
        gathered = tower.project(hidden, candidate_ids=candidates)
        assert gathered.shape == (2, tgt, 3)
        full = tower.project(hidden)
        torch.testing.assert_close(full.index_select(-1, candidates), gathered)
    # Attributes the TwoTower integration reads.
    assert tower.lm_head.weight is tower.tok.weight
    assert getattr(tower, "tie_output_embedding", True) is True
    assert tower.max_len == 32
    assert len(tower.layers) > 0
    tower.set_runtime_symbol_features(None)


def test_hf_denoiser_untied_output_head() -> None:
    _require_tiny_llama()
    import torch

    from slm_training.models.hf_denoiser import HFDenoiserTower

    torch.manual_seed(0)
    vocab, d_model, tgt, ctx_len = 23, 16, 6, 3
    tower = HFDenoiserTower(
        vocab_size=vocab,
        d_model=d_model,
        max_len=32,
        hf_model_name=TINY_LLAMA,
        tie_output_embedding=False,
    )
    assert tower.tie_output_embedding is False
    assert tower.lm_head.weight is not tower.tok.weight
    torch.testing.assert_close(tower.lm_head.weight, tower.tok.weight)

    tower.eval()
    noisy = torch.randint(1, vocab, (2, tgt))
    ctx = torch.randn(2, ctx_len, d_model)
    with torch.no_grad():
        logits = tower(noisy, ctx, pad_id=0)
    assert logits.shape == (2, tgt, vocab)
    assert torch.isfinite(logits).all()


def test_twotower_hf_denoiser_trains_and_roundtrips(tmp_path: Path) -> None:
    _require_tiny_llama()
    import torch

    from slm_training.models.hf_denoiser import HFDenoiserTower

    records = [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            context_backend="scratch",
            denoiser_backend="hf",
            hf_model_name=TINY_LLAMA,
            grammar_constrained=False,
            gen_steps=2,
            seed=0,
        ),
        device="cpu",
    )
    assert isinstance(model.denoiser, HFDenoiserTower)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss = model.training_loss(records)
    loss.backward()
    # The adaptation trains the backbone, unlike the frozen context tower.
    grads = [
        p.grad
        for n, p in model.named_parameters()
        if n.startswith("denoiser.backbone.") and p.grad is not None
    ]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    opt.step()

    ckpt = tmp_path / "b4.pt"
    model.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert loaded.config.denoiser_backend == "hf"
    assert isinstance(loaded.denoiser, HFDenoiserTower)
