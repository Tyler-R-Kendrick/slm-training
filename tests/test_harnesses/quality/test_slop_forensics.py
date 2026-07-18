"""Tests for LDI3-02 structural-slop forensics (SLM-129).

Cover the honesty-critical engine: program-occurrence profiling, deterministic
group-bootstrap ranking, low-support suppression, source concentration, detector
classification, deterministic report, and robust extraction (n-grams + graceful
degradation). No model update or ban list is exercised.
"""

from __future__ import annotations

from slm_training.harnesses.quality.slop_forensics import (
    ProgramFeatures,
    classify_finding,
    extract_features,
    forensics_report,
    profile_corpora,
    rank_motifs,
)


def _feat(pid, corpus, group, *, ngrams=(), skeleton="", placeholders=(), grammar=()):
    return ProgramFeatures(
        program_id=pid,
        corpus=corpus,
        prompt_group=group,
        surface_ngrams=tuple(ngrams),
        skeleton_hash=skeleton,
        placeholders=tuple(placeholders),
        grammar_motifs=tuple(grammar),
    )


def _corpus():
    feats = []
    # parent over-represents motif "SLOP" across many prompt groups (high support, stable)
    for i in range(8):
        feats.append(_feat(f"p{i}", "parent", f"g{i % 4}", ngrams=("SLOP", "ok")))
    # baseline rarely has it
    for i in range(8):
        feats.append(_feat(f"b{i}", "gold_silver", f"g{i % 4}", ngrams=("ok",) if i else ("SLOP", "ok")))
    for i in range(8):
        feats.append(_feat(f"h{i}", "held_out", f"g{i % 4}", ngrams=("SLOP", "ok")))
    return feats


def test_profile_counts_program_occurrence_not_frequency() -> None:
    # one program repeating a motif counts once for that program
    feats = [_feat("p0", "parent", "g0", ngrams=("X", "X", "X"))]
    profile = profile_corpora(feats)
    assert profile["surface_ngram"]["parent"].motif_program_count["X"] == 1


def test_rank_flags_over_represented_and_sign_is_positive() -> None:
    findings = rank_motifs(profile_corpora(_corpus()), seed=1, bootstrap_iters=80)
    slop = [f for f in findings if f.motif == "SLOP"]
    assert slop and slop[0].log_odds > 0
    assert slop[0].parent_count > slop[0].baseline_count


def test_group_bootstrap_is_deterministic_under_seed() -> None:
    p = profile_corpora(_corpus())
    a = rank_motifs(p, seed=7, bootstrap_iters=120)
    b = rank_motifs(p, seed=7, bootstrap_iters=120)
    assert [f.as_dict() for f in a] == [f.as_dict() for f in b]
    c = rank_motifs(p, seed=8, bootstrap_iters=120)
    # different seed generally shifts the bootstrap CI (not the point estimate)
    assert any(f.ci_low != g.ci_low for f, g in zip(a, c)) or len(a) <= 1


def test_low_support_cannot_outrank_stable_high_support() -> None:
    feats = _corpus()
    # a low-support extreme motif present in exactly one parent program, never baseline
    feats.append(_feat("rare", "parent", "gz", ngrams=("RARE",)))
    findings = rank_motifs(profile_corpora(feats), seed=1, bootstrap_iters=80, min_support=3)
    order = [f.motif for f in findings]
    rare = next(f for f in findings if f.motif == "RARE")
    assert rare.low_support is True
    # SLOP (high support) must sort ahead of the low-support RARE
    assert order.index("SLOP") < order.index("RARE")


def test_source_concentration_reported() -> None:
    # motif in a single prompt group -> concentration 1.0
    feats = [
        _feat("p0", "parent", "only", ngrams=("Z",)),
        _feat("p1", "parent", "only", ngrams=("Z",)),
        _feat("p2", "parent", "only", ngrams=("Z",)),
        _feat("b0", "gold_silver", "g0", ngrams=("ok",)),
    ]
    findings = rank_motifs(profile_corpora(feats), seed=1, bootstrap_iters=50, min_log_odds=0.0)
    z = next(f for f in findings if f.motif == "Z")
    assert z.source_concentration == 1.0


def test_detector_classification_rules() -> None:
    assert classify_finding("surface_ngram", "m", whitelisted=True, localizable=False, verifier_gate=None) == "whitelisted_domain_motif"
    assert classify_finding("surface_ngram", "m", whitelisted=False, localizable=False, verifier_gate="G3") == "semantic_failure_candidate"
    assert classify_finding("grammar_motif", "m", whitelisted=False, localizable=True, verifier_gate=None) == "counterfactual_probe_candidate"
    assert classify_finding("skeleton", "m", whitelisted=False, localizable=False, verifier_gate=None) == "constraint_distillation_candidate"
    assert classify_finding("surface_ngram", "m", whitelisted=False, localizable=False, verifier_gate=None) == "diagnostic_only"


def test_report_is_deterministic_and_emits_no_ban_list() -> None:
    findings = rank_motifs(profile_corpora(_corpus()), seed=1, bootstrap_iters=60)
    r1 = forensics_report(findings)
    r2 = forensics_report(findings)
    assert r1 == r2
    assert "ban" in r1["note"].lower() or "no ban list" in r1["note"].lower()
    assert "diagnostic" in r1["note"].lower()
    assert "candidate_manifest" in r1  # candidates, not applied bans


def test_verifier_association_promotes_semantic_candidate() -> None:
    findings = rank_motifs(
        profile_corpora(_corpus()), seed=1, bootstrap_iters=60,
        verifier_associated={"SLOP": "G4"},
    )
    slop = next(f for f in findings if f.motif == "SLOP")
    assert slop.detector_class == "semantic_failure_candidate"


def test_extract_features_ngrams_and_graceful_degradation() -> None:
    feat = extract_features("id", "parent", "a b a", max_ngram=2)
    assert "a" in feat.surface_ngrams and "a▁b" in feat.surface_ngrams
    # A clearly invalid program cannot canonicalize -> parse_ok False, no crash.
    bad = extract_features("id2", "parent", "((((not openui at all", max_ngram=1)
    assert bad.parse_ok is False and bad.skeleton_hash == ""
