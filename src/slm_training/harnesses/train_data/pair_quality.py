"""Deterministic prompt/target integrity and corpus nuisance-cue diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harnesses.staged import Capability
from slm_training.harnesses.synthesis_plan import (
    PairQualityPolicy,
    PromptSurface,
    tiny_corpus_generation_policy,
)

SCHEMA_VERSION = "pair_quality/v1"
MIN_SUPPORT_ROOTS = 20
GROUPED_FOLD_CAP = 8
_NGRAM_N = 4
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SURFACE_LEAK_RES = (
    re.compile(r"Stack\("),
    re.compile(r":slot_"),
    re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*"),
    re.compile(r"root\s*="),
)
_EVIDENCE = frozenset({"verified", "partially_verified", "unverified"})


@lru_cache(maxsize=1)
def _default_prompt_cues() -> tuple[tuple[str, ...], tuple[str, ...]]:
    prompts = tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR).prompts
    return prompts.hard_forbidden_cues, prompts.balanced_generic_cues


def _round(value: float) -> float:
    return round(float(value), 6)


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(_as_text(item) for item in value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _meta(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(record.get("meta"))


def _first(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def _lookup(
    record: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *keys: str,
) -> str:
    meta = _meta(record)
    for source in (provenance, meta, record):
        for key in keys:
            if key in source and source[key] not in (None, ""):
                return str(source[key])
    return ""


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _contains_phrase(text: str, phrase: str) -> bool:
    needle = phrase.strip()
    if not needle:
        return False
    return (
        re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", text, flags=re.I)
        is not None
    )


def _cue_hits(text: str, cues: Sequence[str]) -> tuple[str, ...]:
    return tuple(cue for cue in cues if _contains_phrase(text, cue))


def _surface_leak_count(prompt: str) -> int:
    return sum(len(pattern.findall(prompt)) for pattern in _SURFACE_LEAK_RES)


def _inventory_leak_count(prompt: str, inventory: Sequence[str]) -> int:
    items = tuple(item for item in inventory if str(item).strip())
    if len(items) < 2:
        return 0
    if all(_contains_phrase(prompt, item) for item in items):
        return len(items)
    return 0


def _length_bin(tokens: Sequence[str]) -> str:
    n = len(tokens)
    if n <= 6:
        return "short"
    if n <= 12:
        return "medium"
    return "long"


def _case_style(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "none"
    if text == text.upper():
        return "upper"
    if text == text.lower():
        return "lower"
    if text == text.title():
        return "title"
    return "mixed"


def _punct_style(text: str) -> str:
    marks = sum(text.count(mark) for mark in ".,!?;:")
    if marks == 0:
        return "none"
    if text.count("!") + text.count("?") > 1:
        return "heavy"
    return "light"


def _leading_prefix(tokens: Sequence[str]) -> str:
    return " ".join(tokens[:_NGRAM_N])


def _sort_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    meta = _meta(record)
    return (
        _first(meta.get("root_id"), record.get("root_id")),
        _first(meta.get("template_id"), record.get("template_id")),
        _as_text(record.get("prompt")),
        _first(record.get("source"), meta.get("source_family")),
        _first(meta.get("topology_id"), record.get("topology_id")),
        _first(meta.get("split_family_id")),
    )


def _evidence_class(
    *,
    required: Sequence[str],
    coverage: float,
    forbidden_absent: bool,
    contradictions: int,
) -> str:
    if not required and contradictions == 0 and forbidden_absent:
        return "unverified"
    if coverage >= 1.0 and forbidden_absent and contradictions == 0:
        return "verified"
    return "partially_verified"


@dataclass(frozen=True)
class PairAudit:
    accepted: bool
    required_fact_coverage: float
    forbidden_fact_absent: bool
    contradiction_count: int
    target_surface_leak_count: int
    complete_inventory_leak_count: int
    cue_policy_violations: tuple[str, ...]
    renderer_id: str
    source_id: str
    template_id: str
    semantic_frame_id: str
    root_id: str
    topology_id: str
    split_family_id: str
    invariance_group_id: str | None
    evidence_class: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.evidence_class not in _EVIDENCE:
            raise ValueError(f"unknown evidence_class: {self.evidence_class!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "required_fact_coverage": _round(self.required_fact_coverage),
            "forbidden_fact_absent": self.forbidden_fact_absent,
            "contradiction_count": self.contradiction_count,
            "target_surface_leak_count": self.target_surface_leak_count,
            "complete_inventory_leak_count": self.complete_inventory_leak_count,
            "cue_policy_violations": list(self.cue_policy_violations),
            "renderer_id": self.renderer_id,
            "source_id": self.source_id,
            "template_id": self.template_id,
            "semantic_frame_id": self.semantic_frame_id,
            "root_id": self.root_id,
            "topology_id": self.topology_id,
            "split_family_id": self.split_family_id,
            "invariance_group_id": self.invariance_group_id,
            "evidence_class": self.evidence_class,
            "reasons": list(self.reasons),
        }


def _pair_ids(
    record: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, str | None]:
    invariance = _lookup(record, provenance, "invariance_group_id")
    return {
        "renderer_id": _lookup(record, provenance, "renderer_id", "renderer_family"),
        "source_id": _lookup(
            record,
            provenance,
            "source_id",
            "source_family",
            "provider_id",
            "source",
        ),
        "template_id": _lookup(record, provenance, "template_id"),
        "semantic_frame_id": _lookup(record, provenance, "semantic_frame_id"),
        "root_id": _lookup(record, provenance, "root_id"),
        "topology_id": _lookup(record, provenance, "topology_id"),
        "split_family_id": _lookup(record, provenance, "split_family_id"),
        "invariance_group_id": invariance or None,
    }


def audit_pair(record: Mapping[str, Any], provenance: Mapping[str, Any]) -> PairAudit:
    prompt = _as_text(record.get("prompt"))
    meta = _meta(record)
    required = _as_str_tuple(
        provenance.get(
            "pair_required_facts",
            provenance.get(
                "required_facts",
                meta.get("pair_required_facts", meta.get("required_facts")),
            ),
        )
    )
    target_facts = set(
        _as_str_tuple(
            provenance.get(
                "pair_target_facts",
                provenance.get(
                    "target_facts",
                    meta.get("pair_target_facts", meta.get("target_facts")),
                ),
            )
        )
    )
    forbidden = _as_str_tuple(
        provenance.get(
            "pair_forbidden_facts",
            provenance.get(
                "forbidden_facts",
                meta.get("pair_forbidden_facts", meta.get("forbidden_facts")),
            ),
        )
    )
    inventory = _as_str_tuple(
        provenance.get("component_inventory", meta.get("component_inventory"))
    )
    hard_forbidden, _generic = _default_prompt_cues()

    hits = sum(1 for fact in required if _contains_phrase(prompt, fact))
    coverage = 1.0 if not required else hits / len(required)
    forbidden_in_prompt = tuple(
        fact for fact in forbidden if _contains_phrase(prompt, fact)
    )
    forbidden_in_target = tuple(fact for fact in forbidden if fact in target_facts)
    missing_required = tuple(fact for fact in required if fact not in target_facts)
    both = tuple(fact for fact in required if fact in set(forbidden))
    contradiction_count = len(missing_required) + len(forbidden_in_target) + len(both)
    forbidden_absent = not forbidden_in_prompt and not forbidden_in_target
    leaks = _surface_leak_count(prompt)
    inventory_leaks = _inventory_leak_count(prompt, inventory)
    cue_violations = _cue_hits(prompt, hard_forbidden)
    reasons: list[str] = []
    if contradiction_count:
        reasons.append("prompt_target_contradiction")
    if required and coverage < 1.0:
        reasons.append("required_fact_coverage")
    if not forbidden_absent:
        reasons.append("forbidden_fact_present")
    if leaks:
        reasons.append("target_surface_leak")
    if inventory_leaks:
        reasons.append("complete_inventory_leak")
    if cue_violations:
        reasons.append("hard_forbidden_cue")
    accepted = not reasons
    ids = _pair_ids(record, provenance)
    return PairAudit(
        accepted=accepted,
        required_fact_coverage=_round(coverage),
        forbidden_fact_absent=forbidden_absent,
        contradiction_count=contradiction_count,
        target_surface_leak_count=leaks,
        complete_inventory_leak_count=inventory_leaks,
        cue_policy_violations=cue_violations,
        renderer_id=str(ids["renderer_id"] or ""),
        source_id=str(ids["source_id"] or ""),
        template_id=str(ids["template_id"] or ""),
        semantic_frame_id=str(ids["semantic_frame_id"] or ""),
        root_id=str(ids["root_id"] or ""),
        topology_id=str(ids["topology_id"] or ""),
        split_family_id=str(ids["split_family_id"] or ""),
        invariance_group_id=ids["invariance_group_id"],
        evidence_class=_evidence_class(
            required=required,
            coverage=coverage,
            forbidden_absent=forbidden_absent,
            contradictions=contradiction_count,
        ),
        reasons=tuple(reasons),
    )


def _grouped_folds(root_ids: Sequence[str]) -> list[int]:
    """Hash roots into at most 8 folds. Same root always shares one fold."""
    unique = len(set(root_ids))
    k = min(unique, GROUPED_FOLD_CAP)
    if k <= 0:
        return []
    return [
        int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16) % k
        for root in root_ids
    ]


def _majority_baseline(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def _nb_predict(
    train_x: Sequence[Sequence[str]],
    train_y: Sequence[str],
    test_x: Sequence[Sequence[str]],
    *,
    alpha: float = 1.0,
) -> list[str]:
    if not train_y:
        return [""] * len(test_x)
    class_count = Counter(train_y)
    classes = tuple(sorted(class_count))
    majority = sorted(class_count, key=lambda label: (-class_count[label], label))[0]
    vocab = sorted({feat for feats in train_x for feat in feats})
    if not vocab:
        return [majority] * len(test_x)
    word_count: dict[str, Counter[str]] = {label: Counter() for label in classes}
    totals = {label: 0 for label in classes}
    for feats, label in zip(train_x, train_y):
        word_count[label].update(feats)
        totals[label] += len(feats)
    n = len(train_y)
    v = len(vocab)
    vocab_set = set(vocab)
    predicted: list[str] = []
    for feats in test_x:
        feat_counts = Counter(feat for feat in feats if feat in vocab_set)
        best_label = majority
        best_score = -math.inf
        for label in classes:
            score = math.log(class_count[label] / n)
            denom = totals[label] + alpha * v
            for feat, count in feat_counts.items():
                score += count * math.log((word_count[label][feat] + alpha) / denom)
            if score > best_score or (score == best_score and label < best_label):
                best_label = label
                best_score = score
        predicted.append(best_label)
    return predicted


def _grouped_accuracy(
    features: Sequence[Sequence[str]],
    labels: Sequence[str],
    folds: Sequence[int],
) -> float:
    if not labels:
        return 0.0
    predicted = [""] * len(labels)
    for fold in sorted(set(folds)):
        train_idx = [i for i, item in enumerate(folds) if item != fold]
        test_idx = [i for i, item in enumerate(folds) if item == fold]
        if not test_idx or not train_idx:
            continue
        preds = _nb_predict(
            [features[i] for i in train_idx],
            [labels[i] for i in train_idx],
            [features[i] for i in test_idx],
        )
        for index, pred in zip(test_idx, preds):
            predicted[index] = pred
    return sum(pred == label for pred, label in zip(predicted, labels)) / len(labels)


def _classifier_report(
    *,
    accuracy: float,
    majority: float,
    support: int,
    threshold: float,
    folds: Sequence[int],
    root_ids: Sequence[str],
    report_only: bool = False,
) -> dict[str, Any]:
    lift = _round(accuracy - majority)
    if support < MIN_SUPPORT_ROOTS:
        disposition = "insufficient_support"
    elif report_only:
        disposition = "report_only"
    elif lift > threshold:
        disposition = "fail"
    else:
        disposition = "pass"
    assignments = {root: fold for root, fold in zip(root_ids, folds)}
    return {
        "accuracy": _round(accuracy),
        "majority_baseline": _round(majority),
        "lift": lift,
        "support": support,
        "disposition": disposition,
        "n_folds": len(set(folds)),
        "fold_assignments": dict(sorted(assignments.items())),
        "folds": dict(sorted(assignments.items())),
        "record_folds": list(folds),
        "threshold": _round(threshold),
    }


def _cue_features(
    record: Mapping[str, Any],
    audit: PairAudit,
    *,
    hard_forbidden: Sequence[str],
    generic: Sequence[str],
) -> list[str]:
    prompt = _as_text(record.get("prompt"))
    tokens = _tokens(prompt)
    meta = _meta(record)
    feats = [
        f"pre:{_leading_prefix(tokens)}",
        f"tpl:{audit.template_id}",
        f"ren:{audit.renderer_id}",
        f"srcfam:{_first(meta.get('source_family'), record.get('source'))}",
        f"prv:{_first(meta.get('provider_id'))}",
        f"src:{_first(record.get('source'))}",
        f"len:{_length_bin(tokens)}",
        f"case:{_case_style(prompt)}",
        f"punct:{_punct_style(prompt)}",
    ]
    # Presence and absence tokens so multinomial NB can use one-sided cues.
    for cue in hard_forbidden:
        feats.append(f"hf:{cue}={int(_contains_phrase(prompt, cue))}")
    for cue in generic:
        feats.append(f"gen:{cue}={int(_contains_phrase(prompt, cue))}")
    return feats


def _source_features(record: Mapping[str, Any], audit: PairAudit) -> list[str]:
    meta = _meta(record)
    return [
        f"src:{_first(record.get('source'))}",
        f"srcfam:{_first(meta.get('source_family'), record.get('source'))}",
        f"prv:{_first(meta.get('provider_id'))}",
        f"ren:{audit.renderer_id}",
    ]


def _prompt_features(record: Mapping[str, Any]) -> list[str]:
    return [f"tok:{token}" for token in _tokens(_as_text(record.get("prompt")))]


def _share_report(values: Sequence[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = len(values)
    ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    top, top_n = next(iter(ordered.items()), ("", 0))
    share = 0.0 if total == 0 else top_n / total
    return {
        "top": top,
        "share": _round(share),
        "n": total,
        "counts": {key: int(value) for key, value in ordered.items()},
    }


def _prevalence(
    records: Sequence[Mapping[str, Any]],
    *,
    hard_forbidden: Sequence[str],
    generic: Sequence[str],
) -> dict[str, Any]:
    hard_counts: dict[str, int] = {cue: 0 for cue in hard_forbidden}
    generic_counts: dict[str, int] = {cue: 0 for cue in generic}
    no_generic = 0
    any_hard = 0
    for record in records:
        prompt = _as_text(record.get("prompt"))
        hard_hits = _cue_hits(prompt, hard_forbidden)
        generic_hits = _cue_hits(prompt, generic)
        for cue in hard_hits:
            hard_counts[cue] += 1
        for cue in generic_hits:
            generic_counts[cue] += 1
        if hard_hits:
            any_hard += 1
        if not generic_hits:
            no_generic += 1
    n = len(records)
    return {
        "hard_forbidden": dict(sorted(hard_counts.items())),
        "balanced_generic": dict(sorted(generic_counts.items())),
        "any_hard_forbidden": any_hard,
        "no_generic_fraction": _round(0.0 if n == 0 else no_generic / n),
        "n": n,
    }


def _rare_family_retention(
    audits: Sequence[PairAudit], policy: PairQualityPolicy
) -> dict[str, Any]:
    families = Counter(audit.split_family_id or "" for audit in audits)
    topologies = {audit.topology_id for audit in audits}
    roots = {audit.root_id for audit in audits}
    n = len(audits)
    unique_structure = 0.0 if n == 0 else len(topologies) / n
    exposure = Counter(audit.root_id for audit in audits)
    return {
        "family_counts": dict(sorted(families.items())),
        "n_families": len(families),
        "unique_roots": len(roots),
        "unique_structure_fraction": _round(unique_structure),
        "max_root_exposure": max(exposure.values(), default=0),
        "per_root_exposure_cap": policy.per_root_exposure_cap,
        "retained": True,
    }


def _thresholds(policy: PairQualityPolicy) -> dict[str, Any]:
    payload = policy.to_dict()
    payload["min_support_roots"] = MIN_SUPPORT_ROOTS
    return payload


def _provenance_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    meta = _meta(record)
    return {
        "renderer_id": _first(meta.get("renderer_family"), meta.get("renderer_id")),
        "source_id": _first(
            meta.get("source_family"), meta.get("source_id"), record.get("source")
        ),
        "template_id": _first(meta.get("template_id")),
        "semantic_frame_id": _first(meta.get("semantic_frame_id")),
        "root_id": _first(meta.get("root_id")),
        "topology_id": _first(meta.get("topology_id")),
        "split_family_id": _first(meta.get("split_family_id")),
        "invariance_group_id": _first(meta.get("invariance_group_id")) or None,
        "required_facts": list(
            _as_str_tuple(meta.get("pair_required_facts", meta.get("required_facts")))
        ),
        "target_facts": list(
            _as_str_tuple(meta.get("pair_target_facts", meta.get("target_facts")))
        ),
        "forbidden_facts": list(
            _as_str_tuple(meta.get("pair_forbidden_facts", meta.get("forbidden_facts")))
        ),
        "component_inventory": list(_as_str_tuple(meta.get("component_inventory"))),
        "provider_id": _first(meta.get("provider_id")),
        "source_family": _first(meta.get("source_family")),
        "renderer_family": _first(meta.get("renderer_family")),
    }


@dataclass(frozen=True)
class CorpusAudit:
    pair_audits: tuple[PairAudit, ...]
    cue_prevalence: dict[str, Any]
    prefix_concentration: dict[str, Any]
    template_concentration: dict[str, Any]
    cue_only: dict[str, Any]
    source_only: dict[str, Any]
    full_prompt: dict[str, Any]
    rare_family_retention: dict[str, Any]
    thresholds: dict[str, Any]
    passed: bool
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "pair_audits": [item.to_dict() for item in self.pair_audits],
            "cue_prevalence": self.cue_prevalence,
            "prefix_concentration": self.prefix_concentration,
            "template_concentration": self.template_concentration,
            "cue_only": self.cue_only,
            "source_only": self.source_only,
            "full_prompt": self.full_prompt,
            "rare_family_retention": self.rare_family_retention,
            "thresholds": self.thresholds,
            "passed": self.passed,
            "blocking_reasons": list(self.blocking_reasons),
        }
        return json.loads(canonical_json(payload))

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())


def audit_corpus(
    records: Sequence[Mapping[str, Any]],
    policy: PairQualityPolicy,
    *,
    surface: str = "simplified_nl",
) -> CorpusAudit:
    ordered = tuple(sorted(records, key=_sort_key))
    pair_audits = tuple(
        audit_pair(record, _provenance_from_record(record)) for record in ordered
    )
    hard_forbidden, generic = _default_prompt_cues()
    prefixes = [
        _leading_prefix(_tokens(_as_text(record.get("prompt")))) for record in ordered
    ]
    templates = [audit.template_id for audit in pair_audits]
    prefix_concentration = _share_report(prefixes)
    template_concentration = _share_report(templates)
    root_ids = [
        audit.root_id or f"row-{index}" for index, audit in enumerate(pair_audits)
    ]
    folds = _grouped_folds(root_ids)
    support = len(set(root_ids))
    family_labels = [audit.split_family_id or audit.root_id for audit in pair_audits]
    topo_labels = [audit.topology_id or audit.root_id for audit in pair_audits]
    cue_features = [
        _cue_features(
            record,
            audit,
            hard_forbidden=hard_forbidden,
            generic=generic,
        )
        for record, audit in zip(ordered, pair_audits)
    ]
    source_features = [
        _source_features(record, audit) for record, audit in zip(ordered, pair_audits)
    ]
    prompt_features = [_prompt_features(record) for record in ordered]
    cue_only = _classifier_report(
        accuracy=_grouped_accuracy(cue_features, family_labels, folds),
        majority=_majority_baseline(family_labels),
        support=support,
        threshold=policy.max_cue_only_accuracy_lift,
        folds=folds,
        root_ids=root_ids,
    )
    source_only = _classifier_report(
        accuracy=_grouped_accuracy(source_features, topo_labels, folds),
        majority=_majority_baseline(topo_labels),
        support=support,
        threshold=policy.max_source_only_accuracy_lift,
        folds=folds,
        root_ids=root_ids,
    )
    full_prompt = _classifier_report(
        accuracy=_grouped_accuracy(prompt_features, family_labels, folds),
        majority=_majority_baseline(family_labels),
        support=support,
        threshold=policy.max_cue_only_accuracy_lift,
        folds=folds,
        root_ids=root_ids,
        report_only=True,
    )
    blocking: list[str] = []
    simplified = surface == PromptSurface.SIMPLIFIED_NL.value
    if simplified and any(audit.cue_policy_violations for audit in pair_audits):
        blocking.append("hard_forbidden_cue")
    if simplified and any(audit.target_surface_leak_count for audit in pair_audits):
        blocking.append("target_surface_leak")
    if policy.reject_target_surface_leaks and any(
        audit.complete_inventory_leak_count for audit in pair_audits
    ):
        blocking.append("complete_inventory_leak")
    if any("prompt_target_contradiction" in audit.reasons for audit in pair_audits):
        blocking.append("prompt_target_contradiction")
    if cue_only["disposition"] == "fail":
        blocking.append("cue_only_lift")
    if source_only["disposition"] == "fail":
        blocking.append("source_only_lift")
    if prefix_concentration["share"] > policy.max_normalized_prefix_share:
        blocking.append("prefix_concentration")
    if template_concentration["share"] > policy.max_exact_template_share:
        blocking.append("template_concentration")
    return CorpusAudit(
        pair_audits=pair_audits,
        cue_prevalence=_prevalence(
            ordered, hard_forbidden=hard_forbidden, generic=generic
        ),
        prefix_concentration=prefix_concentration,
        template_concentration=template_concentration,
        cue_only=cue_only,
        source_only=source_only,
        full_prompt=full_prompt,
        rare_family_retention=_rare_family_retention(pair_audits, policy),
        thresholds=_thresholds(policy),
        passed=not blocking,
        blocking_reasons=tuple(blocking),
    )


__all__ = [
    "SCHEMA_VERSION",
    "MIN_SUPPORT_ROOTS",
    "GROUPED_FOLD_CAP",
    "PairAudit",
    "CorpusAudit",
    "audit_pair",
    "audit_corpus",
]
