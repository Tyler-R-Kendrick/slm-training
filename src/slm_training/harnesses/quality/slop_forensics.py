"""LDI3-02 structural-slop forensics for OpenUI generations (SLM-129).

The OpenUI analogue of Auto-Antislop's model-specific phrase profiling. It
profiles verified human/program baselines against on-policy generations at the
surface-token, structural (canonical skeleton / component), and symbol levels,
then ranks over-represented motifs with group-bootstrap statistics and emits
*detector candidates* — never automatic semantic bans, never training labels.

Design: a robust, deterministic forensics **engine** over a clean
:class:`ProgramFeatures` abstraction, plus a defensive :func:`extract_features`
that derives the program-derivable families (canonical skeleton, placeholders,
surface-token n-grams) via the existing DSL canonicalizer — no fragile parse
internals. AST-component, grammar/compiler-trace, and verifier-outcome features
are accepted as optional pre-extracted inputs and populate only when that
evidence is supplied (the issue's "when traces exist" / "where verifier evidence
permits"). Correlation/over-representation is **not** causal preference evidence.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.grammar.backends.ast_utils import collect_placeholders_from_text

__all__ = [
    "ProgramFeatures",
    "CorpusName",
    "MotifFinding",
    "DetectorClass",
    "extract_features",
    "profile_corpora",
    "rank_motifs",
    "classify_finding",
    "forensics_report",
]

# Gold/Silver define the human-quality baseline; parent = on-policy generations.
CorpusName = Literal["gold_silver", "held_out", "parent", "intervention"]
DetectorClass = Literal[
    "diagnostic_only",
    "counterfactual_probe_candidate",
    "constraint_distillation_candidate",
    "semantic_failure_candidate",
    "whitelisted_domain_motif",
]

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]")


@dataclass(frozen=True)
class ProgramFeatures:
    """Per-program motifs by family. Trace/verifier families are optional and
    empty unless that evidence was supplied."""

    program_id: str
    corpus: CorpusName
    prompt_group: str
    surface_ngrams: tuple[str, ...] = ()
    skeleton_hash: str = ""
    placeholders: tuple[str, ...] = ()
    component_edges: tuple[str, ...] = ()  # optional (AST evidence)
    grammar_motifs: tuple[str, ...] = ()  # optional (trace evidence)
    first_failing_gate: str | None = None  # optional (verifier evidence)
    parse_ok: bool = True

    def motifs(self) -> dict[str, tuple[str, ...]]:
        return {
            "surface_ngram": self.surface_ngrams,
            "skeleton": (self.skeleton_hash,) if self.skeleton_hash else (),
            "placeholder": self.placeholders,
            "component_edge": self.component_edges,
            "grammar_motif": self.grammar_motifs,
        }


def _token_ngrams(text: str, max_n: int) -> tuple[str, ...]:
    toks = _TOKEN_RE.findall(text)
    out: list[str] = []
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            out.append("▁".join(toks[i : i + n]))
    return tuple(out)


def extract_features(
    program_id: str,
    corpus: CorpusName,
    source: str,
    *,
    dsl: str | None = None,
    prompt_group: str = "",
    max_ngram: int = 3,
    component_edges: Sequence[str] = (),
    grammar_motifs: Sequence[str] = (),
    first_failing_gate: str | None = None,
) -> ProgramFeatures:
    """Derive robust program features. Canonicalization collapses alpha-equivalent
    programs, so symbol-renamed variants do not create false-distinct skeletons.
    Degrades to token-only (``parse_ok=False``) if canonicalization fails."""
    parse_ok = True
    try:
        skeleton = canonical_fingerprint(source, dsl=dsl)
    except Exception:  # noqa: BLE001 - forensics must not crash on a bad generation
        skeleton = ""
        parse_ok = False
    try:
        placeholders = tuple(sorted(set(collect_placeholders_from_text(source))))
    except Exception:  # noqa: BLE001
        placeholders = ()
    return ProgramFeatures(
        program_id=program_id,
        corpus=corpus,
        prompt_group=prompt_group,
        surface_ngrams=_token_ngrams(source, max_ngram),
        skeleton_hash=skeleton,
        placeholders=placeholders,
        component_edges=tuple(component_edges),
        grammar_motifs=tuple(grammar_motifs),
        first_failing_gate=first_failing_gate,
        parse_ok=parse_ok,
    )


@dataclass
class _FamilyCounts:
    total_programs: int = 0
    motif_program_count: Counter[str] = field(default_factory=Counter)
    motif_groups: dict[str, set[str]] = field(default_factory=dict)


def profile_corpora(
    features: Iterable[ProgramFeatures],
) -> dict[str, dict[CorpusName, _FamilyCounts]]:
    """Count motif *program*-occurrence (not raw token count) per family/corpus,
    tracking the prompt groups each motif appears in for group-bootstrap."""
    profile: dict[str, dict[CorpusName, _FamilyCounts]] = {}
    for feat in features:
        for family, motifs in feat.motifs().items():
            per_corpus = profile.setdefault(family, {})
            counts = per_corpus.setdefault(feat.corpus, _FamilyCounts())
            counts.total_programs += 1
            for motif in set(motifs):  # per-program presence, not frequency
                counts.motif_program_count[motif] += 1
                counts.motif_groups.setdefault(motif, set()).add(feat.prompt_group)
    return profile


@dataclass(frozen=True)
class MotifFinding:
    family: str
    motif: str
    parent_count: int
    parent_total: int
    baseline_count: int
    baseline_total: int
    log_odds: float
    ci_low: float
    ci_high: float
    support: int
    source_concentration: float
    held_stable: bool
    low_support: bool
    detector_class: DetectorClass

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "motif": self.motif,
            "parent_count": self.parent_count,
            "parent_total": self.parent_total,
            "baseline_count": self.baseline_count,
            "baseline_total": self.baseline_total,
            "log_odds": round(self.log_odds, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "support": self.support,
            "source_concentration": round(self.source_concentration, 6),
            "held_stable": self.held_stable,
            "low_support": self.low_support,
            "detector_class": self.detector_class,
        }


def _smoothed_log_odds(ca: int, na: int, cb: int, nb: int, prior: float) -> float:
    pa = (ca + prior) / (na + 2 * prior)
    pb = (cb + prior) / (nb + 2 * prior)
    return math.log(pa / (1 - pa)) - math.log(pb / (1 - pb))


def _group_bootstrap_ci(
    parent: _FamilyCounts,
    baseline: _FamilyCounts,
    motif: str,
    *,
    seed: int,
    iters: int,
    prior: float,
) -> tuple[float, float]:
    """Bootstrap the log-odds by resampling prompt groups (not tokens), so a motif
    concentrated in one prompt family gets a wide, honest interval. Deterministic
    under ``seed``."""
    rng = random.Random(seed)
    p_groups = sorted({g for gs in parent.motif_groups.values() for g in gs}) or [""]
    b_groups = sorted({g for gs in baseline.motif_groups.values() for g in gs}) or [""]
    motif_p_groups = parent.motif_groups.get(motif, set())
    motif_b_groups = baseline.motif_groups.get(motif, set())
    # Approximate per-group rate as presence-in-group; resample groups with replacement.
    samples: list[float] = []
    for _ in range(iters):
        ca = sum(1 for _ in range(len(p_groups)) if rng.choice(p_groups) in motif_p_groups)
        cb = sum(1 for _ in range(len(b_groups)) if rng.choice(b_groups) in motif_b_groups)
        samples.append(
            _smoothed_log_odds(ca, len(p_groups), cb, len(b_groups), prior)
        )
    samples.sort()
    lo = samples[max(0, int(0.025 * iters) - 1)]
    hi = samples[min(iters - 1, int(0.975 * iters))]
    return lo, hi


def rank_motifs(
    profile: Mapping[str, Mapping[CorpusName, _FamilyCounts]],
    *,
    seed: int = 0,
    bootstrap_iters: int = 200,
    prior: float = 0.5,
    min_support: int = 3,
    min_log_odds: float = 0.5,
    whitelist: Sequence[str] = (),
    localizable_families: Sequence[str] = ("grammar_motif", "component_edge"),
    verifier_associated: Mapping[str, str] | None = None,
) -> list[MotifFinding]:
    """Rank parent-over-baseline motifs. Low-support motifs are flagged and cannot
    outrank stable, high-support effects (they sort last regardless of ratio)."""
    whitelist_set = set(whitelist)
    verifier_associated = verifier_associated or {}
    findings: list[MotifFinding] = []
    for family, corpora in profile.items():
        parent = corpora.get("parent")
        baseline = corpora.get("gold_silver")
        if parent is None or baseline is None:
            continue
        held = corpora.get("held_out")
        for motif, pc in parent.motif_program_count.items():
            bc = baseline.motif_program_count.get(motif, 0)
            lo_odds = _smoothed_log_odds(pc, parent.total_programs, bc, baseline.total_programs, prior)
            if lo_odds < min_log_odds:
                continue
            ci_low, ci_high = _group_bootstrap_ci(
                parent, baseline, motif, seed=seed, iters=bootstrap_iters, prior=prior
            )
            groups = parent.motif_groups.get(motif, set())
            concentration = 1.0 / max(1, len(groups))  # 1.0 == a single prompt family
            held_stable = True
            if held is not None:
                hc = held.motif_program_count.get(motif, 0)
                held_odds = _smoothed_log_odds(
                    hc, held.total_programs, bc, baseline.total_programs, prior
                )
                held_stable = abs(held_odds) <= abs(lo_odds) + 1.0
            low_support = pc < min_support
            finding = MotifFinding(
                family=family,
                motif=motif,
                parent_count=pc,
                parent_total=parent.total_programs,
                baseline_count=bc,
                baseline_total=baseline.total_programs,
                log_odds=lo_odds,
                ci_low=ci_low,
                ci_high=ci_high,
                support=pc,
                source_concentration=concentration,
                held_stable=held_stable,
                low_support=low_support,
                detector_class=classify_finding(
                    family,
                    motif,
                    whitelisted=motif in whitelist_set,
                    localizable=family in localizable_families,
                    verifier_gate=verifier_associated.get(motif),
                ),
            )
            findings.append(finding)
    # High-support, held-stable, higher log-odds first; low-support always sorts last.
    findings.sort(
        key=lambda f: (f.low_support, not f.held_stable, -f.log_odds, -f.support, f.motif)
    )
    return findings


def classify_finding(
    family: str,
    motif: str,
    *,
    whitelisted: bool,
    localizable: bool,
    verifier_gate: str | None,
) -> DetectorClass:
    """Assign a detector-candidate class. No class becomes a training label here;
    a semantic-failure candidate still requires same-state replay evidence."""
    if whitelisted:
        return "whitelisted_domain_motif"
    if verifier_gate is not None:
        return "semantic_failure_candidate"
    if localizable:
        return "counterfactual_probe_candidate"
    if family in ("skeleton",):
        return "constraint_distillation_candidate"
    return "diagnostic_only"


def forensics_report(findings: Sequence[MotifFinding], *, top: int = 50) -> dict[str, Any]:
    """A deterministic, JSON-safe report. Over-representation is diagnostic, not a
    causal preference label, and no ban list is emitted."""
    by_class: Counter[str] = Counter(f.detector_class for f in findings)
    return {
        "note": (
            "Over-representation vs Gold/Silver is diagnostic only, not causal "
            "preference evidence. No ban list; no semantic labels; no model update."
        ),
        "finding_count": len(findings),
        "by_detector_class": {k: by_class[k] for k in sorted(by_class)},
        "top_findings": [f.as_dict() for f in findings[:top]],
        "candidate_manifest": [
            f.as_dict()
            for f in findings
            if f.detector_class
            in ("counterfactual_probe_candidate", "semantic_failure_candidate")
            and not f.low_support
        ],
    }
