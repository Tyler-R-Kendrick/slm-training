"""Fixed teacher-forced loss suites over the versioned test artifacts.

Five categories, each a deterministic denoising-NLL evaluation:

* ``binding``    (0.30) — placeholder / ``<SYM_i>`` / binder-name positions.
* ``structural`` (0.25) — grammar terminals, component names, list/assignment
  structure positions.
* ``repair``     (0.20) — a single deterministic wrong-token edit is presented
  as a *visible* token; score the NLL of restoring the original at that
  position (E32-style revision capability).
* ``schema_ood`` (0.15) — broad denoising NLL on the ``ood`` suite.
* ``broad``      (0.10) — broad denoising NLL on the base held-out suite.

The suite definition (version, weights, rates, seed, record ids) is frozen to
a JSON artifact so numbers stay comparable across runs; changing any of it is
an explicit objective change and must bump ``LOSS_SUITE_VERSION``.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.denoising_nll import (
    DenoisingNLLConfig,
    _context_text,
    _target_ids,
    evaluate_denoising_nll,
)

LOSS_SUITE_VERSION = "v1"

# Frozen, versioned objective — load from the committed JSON artifact so
# cross-run comparisons cannot silently drift when Python constants change.


def load_suite_spec(version: str = LOSS_SUITE_VERSION) -> dict[str, Any]:
    path = Path(__file__).with_name(f"loss_suite_{version}.json")
    if not path.exists():
        raise FileNotFoundError(f"frozen loss-suite spec missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


_SUITE_SPEC = load_suite_spec(LOSS_SUITE_VERSION)
CATEGORY_WEIGHTS: dict[str, float] = {
    str(k): float(v) for k, v in (_SUITE_SPEC.get("weights") or {}).items()
}
DEFAULT_MASK_RATES_FROM_SPEC: tuple[float, ...] = tuple(
    float(r) for r in (_SUITE_SPEC.get("mask_rates") or [])
)

_STRUCT_CHARS = {"(", ")", "[", "]", ",", "=", "\n"}
_BINDING_CHARS = {":", ".", '"'}


def _token_class(tokenizer: Any, token_id: int) -> str:
    """Classify a token id as binding | structural | other."""
    try:
        from slm_training.models.dsl_tokenizer import (
            TokenKind,
            is_dsl_native_tokenizer,
        )

        if is_dsl_native_tokenizer(tokenizer):
            kind = tokenizer.kind_of(token_id)
            if kind in {TokenKind.SYM, TokenKind.BIND}:
                return "binding"
            if kind in {TokenKind.STRUCT, TokenKind.COMPONENT}:
                return "structural"
            return "other"
    except Exception:  # noqa: BLE001
        pass
    token = tokenizer.id_to_token.get(int(token_id), "")
    if not token:
        return "other"
    if token in _STRUCT_CHARS or (token[:1].isupper() and token.isidentifier()):
        return "structural"
    if token in _BINDING_CHARS or (token[:1].islower() and token.isidentifier()):
        return "binding"
    return "other"


def binding_positions(model: Any, record: ExampleRecord, ids: list[int]) -> list[int]:
    _ = record
    return [
        i
        for i in range(1, len(ids))
        if _token_class(model.tokenizer, ids[i]) == "binding"
    ]


def structural_positions(
    model: Any, record: ExampleRecord, ids: list[int]
) -> list[int]:
    _ = record
    return [
        i
        for i in range(1, len(ids))
        if _token_class(model.tokenizer, ids[i]) == "structural"
    ]


def _repair_rng(record_id: str, edit_index: int, *, suite_version: str, seed: int):
    payload = f"{suite_version}|{record_id}|repair|{edit_index}|{seed}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _wrong_token_id(
    tokenizer: Any, original_id: int, rng: random.Random
) -> int | None:
    """Deterministic locally-plausible wrong token: same class when possible."""
    special = {
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.mask_id,
        tokenizer.unk_id,
    }
    target_class = _token_class(tokenizer, original_id)
    same_class: list[int] = []
    any_class: list[int] = []
    for tid in range(tokenizer.vocab_size):
        if tid == original_id or tid in special:
            continue
        any_class.append(tid)
        if target_class != "other" and _token_class(tokenizer, tid) == target_class:
            same_class.append(tid)
    pool = same_class or any_class
    if not pool:
        return None
    return pool[rng.randrange(len(pool))]


@dataclass(frozen=True)
class RepairNLLConfig:
    suite_version: str = LOSS_SUITE_VERSION
    seed: int = 0
    edits_per_record: int = 3
    batch_size: int = 8


@torch.no_grad()
def evaluate_repair_nll(
    model: Any,
    records: list[ExampleRecord],
    *,
    config: RepairNLLConfig | None = None,
) -> dict[str, Any]:
    """NLL of restoring the original token at one corrupted *visible* position.

    The corrupted canvas has no masks: the denoiser must recognize and revise
    a wrong visible token, which is the primitive behind verifier-localized
    repair (E61/E62 groundwork).
    """
    cfg = config or RepairNLLConfig()
    was_training = bool(getattr(model, "training", False))
    model.eval()
    tokenizer = model.tokenizer
    device = model.device_name
    from slm_training.models.twotower import _pad_batch

    canvases: list[list[int]] = []
    originals: list[tuple[int, int]] = []  # (position, original_id) per canvas
    contexts: list[str] = []
    skipped: list[dict[str, str]] = []
    n_records = 0
    for record in records:
        ids = _target_ids(model, record)
        special = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}
        eligible = [i for i in range(1, len(ids)) if ids[i] not in special]
        if not eligible:
            skipped.append({"id": record.id, "reason": "no_eligible_positions"})
            continue
        ctx_text = _context_text(model, record)
        made_edit = False
        for edit in range(max(1, cfg.edits_per_record)):
            rng = _repair_rng(
                record.id, edit, suite_version=cfg.suite_version, seed=cfg.seed
            )
            pos = eligible[rng.randrange(len(eligible))]
            wrong = _wrong_token_id(tokenizer, int(ids[pos]), rng)
            if wrong is None:
                continue
            canvas = list(ids)
            canvas[pos] = int(wrong)
            canvases.append(canvas)
            originals.append((pos, int(ids[pos])))
            contexts.append(ctx_text)
            made_edit = True
        if made_edit:
            n_records += 1

    nll_sum = 0.0
    restored_top1 = 0
    n_edits = len(canvases)
    batch_size = max(1, int(cfg.batch_size))
    for start in range(0, n_edits, batch_size):
        chunk = canvases[start : start + batch_size]
        chunk_meta = originals[start : start + batch_size]
        ctx, ctx_pad = model._encode_context(
            contexts[start : start + batch_size], cache_keys=None
        )
        noisy = _pad_batch(chunk, tokenizer.pad_id, device=device)
        logits = model.denoiser(
            noisy, ctx, pad_id=tokenizer.pad_id, ctx_pad_mask=ctx_pad
        )
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for row, (pos, original_id) in enumerate(chunk_meta):
            nll_sum += -float(log_probs[row, pos, original_id].item())
            if int(logits[row, pos].argmax().item()) == original_id:
                restored_top1 += 1

    if was_training:
        model.train()
    return {
        "suite_version": cfg.suite_version,
        "seed": cfg.seed,
        "edits_per_record": cfg.edits_per_record,
        "n_records": n_records,
        "n_edits": n_edits,
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "aggregate": {
            "mean_nll": nll_sum / n_edits if n_edits else None,
            "restore_top1": restored_top1 / n_edits if n_edits else None,
        },
    }


def loss_suite_definition(
    *,
    base_suite: str,
    ood_suite: str,
    base_records: list[ExampleRecord] | None,
    ood_records: list[ExampleRecord] | None,
    nll_config: DenoisingNLLConfig,
    repair_config: RepairNLLConfig,
) -> dict[str, Any]:
    """Frozen, versioned description of exactly what was evaluated.

    Starts from the committed ``loss_suite_<version>.json`` artifact and fills
    in the concrete record ids / rates used for this run so results stay
    comparable across machines.
    """
    try:
        definition = dict(load_suite_spec(nll_config.suite_version))
    except FileNotFoundError:
        definition = {
            "loss_suite_version": nll_config.suite_version,
            "weights": dict(CATEGORY_WEIGHTS),
        }
    definition.update(
        {
            "loss_suite_version": nll_config.suite_version,
            "weights": dict(CATEGORY_WEIGHTS),
            "mask_rates": list(nll_config.mask_rates),
            "mask_seed": nll_config.mask_seed,
            "repair_seed": repair_config.seed,
            "repair_edits_per_record": repair_config.edits_per_record,
            "base_suite": base_suite,
            "ood_suite": ood_suite,
            "base_record_ids": (
                sorted(r.id for r in base_records) if base_records else None
            ),
            "ood_record_ids": (
                sorted(r.id for r in ood_records) if ood_records else None
            ),
        }
    )
    return definition


def _load_suite(test_dir: Path, suite: str) -> list[ExampleRecord] | None:
    from slm_training.harnesses.model_build.data import load_suite_records

    try:
        return load_suite_records(Path(test_dir), suite)
    except FileNotFoundError:
        return None


def evaluate_loss_suites(
    model: Any,
    test_dir: Path | str,
    *,
    nll_config: DenoisingNLLConfig | None = None,
    repair_config: RepairNLLConfig | None = None,
    base_suite: str | None = None,
    ood_suite: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run all five categories; missing suites yield ``null`` categories.

    The aggregate renormalizes weights over present categories and reports
    ``complete=False`` when anything was missing — a partial aggregate is
    never silently presented as the full objective.
    """
    try:
        spec = load_suite_spec(LOSS_SUITE_VERSION)
    except FileNotFoundError:
        spec = {}
    rates = DEFAULT_MASK_RATES_FROM_SPEC or (0.15, 0.30, 0.50, 0.70, 0.85)
    nll_cfg = nll_config or DenoisingNLLConfig(
        suite_version=LOSS_SUITE_VERSION,
        mask_rates=rates,
        mask_seed=int(spec.get("mask_seed", 0) or 0),
    )
    repair_cfg = repair_config or RepairNLLConfig(
        suite_version=LOSS_SUITE_VERSION,
        seed=int(spec.get("repair_seed", 0) or 0),
        edits_per_record=int(spec.get("repair_edits_per_record", 3) or 3),
    )
    test_dir = Path(test_dir)
    base_suite = base_suite or str(spec.get("base_suite") or "held_out")
    ood_suite = ood_suite or str(spec.get("ood_suite") or "ood")

    base_records = _load_suite(test_dir, base_suite)
    ood_records = _load_suite(test_dir, ood_suite)
    if limit is not None:
        if base_records is not None:
            base_records = base_records[: max(0, int(limit))]
        if ood_records is not None:
            ood_records = ood_records[: max(0, int(limit))]

    categories: dict[str, dict[str, Any] | None] = {}
    if base_records:
        categories["binding"] = evaluate_denoising_nll(
            model,
            base_records,
            config=nll_cfg,
            position_filter=lambda record, ids: binding_positions(model, record, ids),
        )
        categories["structural"] = evaluate_denoising_nll(
            model,
            base_records,
            config=nll_cfg,
            position_filter=lambda record, ids: structural_positions(
                model, record, ids
            ),
        )
        categories["repair"] = evaluate_repair_nll(
            model, base_records, config=repair_cfg
        )
        categories["broad"] = evaluate_denoising_nll(model, base_records, config=nll_cfg)
    else:
        categories["binding"] = None
        categories["structural"] = None
        categories["repair"] = None
        categories["broad"] = None
    categories["schema_ood"] = (
        evaluate_denoising_nll(model, ood_records, config=nll_cfg)
        if ood_records
        else None
    )

    weighted_sum = 0.0
    weight_used = 0.0
    missing: list[str] = []
    for name, weight in CATEGORY_WEIGHTS.items():
        report = categories.get(name)
        mean = (
            report.get("aggregate", {}).get("mean_nll") if report is not None else None
        )
        if mean is None:
            missing.append(name)
            continue
        weighted_sum += weight * float(mean)
        weight_used += weight

    definition = loss_suite_definition(
        base_suite=base_suite,
        ood_suite=ood_suite,
        base_records=base_records,
        ood_records=ood_records,
        nll_config=nll_cfg,
        repair_config=repair_cfg,
    )
    return {
        "definition": definition,
        "categories": categories,
        "aggregate": {
            "weighted_nll": weighted_sum / weight_used if weight_used > 0 else None,
            "weight_used": weight_used,
            "missing_categories": missing,
            "complete": not missing,
        },
    }


def write_loss_suite_report(path: Path | str, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
