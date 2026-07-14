"""Export the committed playground checkpoint for lightweight web inference."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.models.twotower import TwoTowerModel


class ContextExport(nn.Module):
    def __init__(self, model: TwoTowerModel) -> None:
        super().__init__()
        self.encoder = model.context.encoder
        self.pad_id = model.context_tokenizer.pad_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids, pad_id=self.pad_id)


class DenoiserExport(nn.Module):
    def __init__(self, model: TwoTowerModel) -> None:
        super().__init__()
        self.denoiser = model.denoiser
        self.pad_id = model.tokenizer.pad_id

    def forward(
        self,
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        ctx_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(
            noisy_ids,
            context,
            pad_id=self.pad_id,
            ctx_pad_mask=ctx_pad_mask,
        )


def export(checkpoint: Path = PLAYGROUND_DEMO_CHECKPOINT) -> tuple[Path, Path]:
    model = TwoTowerModel.from_checkpoint(checkpoint, device="cpu")
    model.eval()
    stem = checkpoint.with_suffix("")
    context_path = stem.with_suffix(".context.onnx")
    denoiser_path = stem.with_suffix(".denoiser.onnx")
    context_ids = torch.tensor([[model.tokenizer.bos_id, model.tokenizer.eos_id]])
    noisy_ids = torch.tensor(
        [[model.tokenizer.bos_id, model.tokenizer.mask_id]], dtype=torch.long
    )
    context = torch.zeros(1, 2, model.config.d_model)
    context_pad = torch.zeros(1, 2, dtype=torch.bool)

    torch.onnx.export(
        ContextExport(model),
        (context_ids,),
        context_path,
        input_names=["input_ids"],
        output_names=["context"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "prompt_length"},
            "context": {0: "batch", 1: "prompt_length"},
        },
        opset_version=17,
    )
    torch.onnx.export(
        DenoiserExport(model),
        (noisy_ids, context, context_pad),
        denoiser_path,
        input_names=["noisy_ids", "context", "ctx_pad_mask"],
        output_names=["logits"],
        dynamic_axes={
            "noisy_ids": {0: "batch", 1: "target_length"},
            "context": {0: "batch", 1: "prompt_length"},
            "ctx_pad_mask": {0: "batch", 1: "prompt_length"},
            "logits": {0: "batch", 1: "target_length"},
        },
        opset_version=17,
    )
    return context_path, denoiser_path


if __name__ == "__main__":
    for artifact in export():
        print(artifact)
