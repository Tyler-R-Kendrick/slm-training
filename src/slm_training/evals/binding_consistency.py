"""C2 (SLM-26): binding-consistency probe for symbol representations.

Measures whether the model's representation of a symbol token tracks its
*referent* (the placeholder surface it denotes this example) rather than its
slot: mean pairwise cosine of denoiser hidden states at symbol positions that
share a surface vs positions with different surfaces. A representation that
binds well has ``same_surface_cos > cross_surface_cos`` (positive margin).

With ``runtime_symbol_features="replace"`` the *input* embedding consistency
is exact by construction (same surface → identical byte-compositional
vector); this probe measures how far that consistency survives the denoiser
stack, which is the part training has to earn. Diagnostic only — no
threshold, no gate, no ship claim.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from slm_training.dsl.schema import ExampleRecord


def binding_consistency_probe(
    model: Any,
    records: list[ExampleRecord],
) -> dict[str, Any]:
    import torch

    from slm_training.models.dsl_tokenizer import (
        SymbolTable,
        is_dsl_native_tokenizer,
    )

    if not is_dsl_native_tokenizer(model.tokenizer):
        raise ValueError("binding probe requires the DSL-native tokenizer")

    by_surface: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for record in records:
            placeholders = list(record.placeholders or [])
            table = SymbolTable.from_placeholders(
                placeholders, max_slots=model.tokenizer.sym_slots
            )
            ids = model.tokenizer.encode(
                record.openui,
                add_special=True,
                use_symbol_table=True,
                placeholders=placeholders,
            )
            id_tensor = torch.tensor([ids], device=model.device_name)
            ctx, ctx_pad = model._encode_context([record.prompt])
            model._set_runtime_symbol_features([table])
            hidden = model.denoiser.encode(
                id_tensor, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            model.denoiser.set_runtime_symbol_features(None)
            sym_ids = {
                model.tokenizer.sym_id(slot): surface
                for slot, surface in enumerate(table.placeholders)
            }
            for position, token_id in enumerate(ids):
                surface = sym_ids.get(int(token_id))
                if surface is not None:
                    by_surface.setdefault(surface, []).append(
                        hidden[0, position].float()
                    )

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        )

    same: list[float] = []
    for vectors in by_surface.values():
        same.extend(_cos(a, b) for a, b in combinations(vectors, 2))
    cross: list[float] = []
    surfaces = list(by_surface)
    for i, j in combinations(range(len(surfaces)), 2):
        for a in by_surface[surfaces[i]]:
            for b in by_surface[surfaces[j]]:
                cross.append(_cos(a, b))
    same_mean = sum(same) / len(same) if same else float("nan")
    cross_mean = sum(cross) / len(cross) if cross else float("nan")
    return {
        "surfaces": len(by_surface),
        "same_surface_pairs": len(same),
        "cross_surface_pairs": len(cross),
        "same_surface_cos": round(same_mean, 4) if same else None,
        "cross_surface_cos": round(cross_mean, 4) if cross else None,
        "binding_margin": (
            round(same_mean - cross_mean, 4) if same and cross else None
        ),
        "mode": str(getattr(model.config, "runtime_symbol_features", "none")),
    }


__all__ = ["binding_consistency_probe"]
