"""Stratified trace selection for self-distillation (P2)."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from slm_training.data.dedup import (
    binding_pattern_cluster,
    prompt_semantic_cluster,
)
from slm_training.data.leakage import fingerprint_openui_structure


@dataclass(frozen=True)
class SelectConfig:
    budget: int = 2000
    per_stratum: int = 2
    require_accepted: bool = True
    exclude_exact_gold: bool = True
    corpus: str = "self_distilled_success"  # or self_distilled_repair / gold_correction
    seed: int = 0
    policy_shas: tuple[str, ...] | None = None
    decode_config_hash: str | None = None


def _trace_text(trace: dict[str, Any]) -> str:
    final = trace.get("final") or {}
    return str(final.get("text") or "")


def _trace_prompt(trace: dict[str, Any]) -> str:
    meta = trace.get("meta") or {}
    return str(meta.get("prompt") or "")


def corpus_label(trace: dict[str, Any]) -> str:
    labels = dict(trace.get("labels") or {})
    meta = dict(trace.get("meta") or {})
    if labels.get("gold_injected") or meta.get("gold_injected"):
        return "gold_correction"
    if labels.get("repair") or meta.get("failure_cone"):
        return "self_distilled_repair"
    if labels.get("accepted"):
        return "self_distilled_success"
    return "rejected"


def stratum_key(trace: dict[str, Any]) -> tuple[str, ...]:
    prompt = _trace_prompt(trace)
    text = _trace_text(trace)
    labels = dict(trace.get("labels") or {})
    length_bin = str(min(8, len(text) // 80))
    pass_bin = "pass" if labels.get("accepted") else "fail"
    repair = "repair" if labels.get("repair") else "clean"
    return (
        prompt_semantic_cluster(prompt),
        fingerprint_openui_structure(text)[:16] if text else "empty",
        binding_pattern_cluster(text) if text else "empty",
        length_bin,
        pass_bin,
        repair,
    )


def filter_traces(
    traces: Iterable[dict[str, Any]],
    config: SelectConfig,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trace in traces:
        labels = dict(trace.get("labels") or {})
        meta = dict(trace.get("meta") or {})
        if config.corpus and corpus_label(trace) != config.corpus:
            continue
        if config.require_accepted and not labels.get("accepted"):
            continue
        if config.exclude_exact_gold and labels.get("exact_gold"):
            continue
        if config.policy_shas:
            sha = str(meta.get("policy_checkpoint_sha") or "")
            if sha not in config.policy_shas:
                continue
        if config.decode_config_hash:
            if str(meta.get("decode_config_hash") or "") != config.decode_config_hash:
                continue
        out.append(trace)
    return out


def select_traces(
    traces: Iterable[dict[str, Any]],
    *,
    config: SelectConfig | None = None,
) -> list[dict[str, Any]]:
    """Coverage over score: few traces from many strata, random within stratum."""
    cfg = config or SelectConfig()
    filtered = filter_traces(traces, cfg)
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for trace in filtered:
        by_stratum[stratum_key(trace)].append(trace)

    rng = random.Random(cfg.seed)
    selected: list[dict[str, Any]] = []
    # Round-robin across strata for coverage.
    strata = sorted(by_stratum.keys(), key=lambda k: hashlib.sha256(str(k).encode()).hexdigest())
    for key in strata:
        members = list(by_stratum[key])
        rng.shuffle(members)
        selected.extend(members[: max(1, cfg.per_stratum)])
        if len(selected) >= cfg.budget:
            break
    rng.shuffle(selected)
    return selected[: cfg.budget]


def iter_selected(store_root: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
    from slm_training.distill.trace_store import TraceStore

    cfg = SelectConfig(**kwargs) if kwargs else SelectConfig()
    store = TraceStore(store_root)
    yield from select_traces(store.iter_traces(), config=cfg)
