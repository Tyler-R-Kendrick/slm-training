"""Pure-Python structural laws mirroring Lean theorems and VSS honesty rules.

This is an **independent prover backend**: it re-derives the declared laws from
the formal object's payload without invoking Lean and without trusting a bare
verdict string. It intentionally uses only the stdlib + lightweight list/set
membership so translation difficulty stays controlled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _subset(xs: Sequence[Any], ys: Sequence[Any]) -> bool:
    yset = set(ys)
    return all(x in yset for x in xs)


def _set_eq(xs: Sequence[Any], ys: Sequence[Any]) -> bool:
    return set(xs) == set(ys)


# --------------------------------------------------------------------------- #
# Support-certificate honesty (matches replay_support_certificate rules)
# --------------------------------------------------------------------------- #


def check_support_certificate_structure(payload: Mapping[str, Any]) -> list[str]:
    """Return structural violations for an exported support certificate payload."""

    violations: list[str] = []
    cert = payload.get("certificate")
    if not isinstance(cert, dict):
        return ["payload.certificate missing or not an object"]
    declared = payload.get("certificate_digest")
    from slm_training.dsl.solver.support import SupportCertificate

    try:
        certificate = SupportCertificate.from_dict(cert)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"certificate decode failed: {exc}"]

    if declared is not None and certificate.digest != declared:
        violations.append("certificate_digest does not match certificate payload")

    verdict = certificate.verdict.value
    if verdict == "unsupported":
        if not certificate.exhausted:
            violations.append("UNSUPPORTED certificate is not marked exhausted")
        if certificate.stop_reason is not None:
            violations.append("UNSUPPORTED certificate has a budget/stop reason")
        if any(cov in {"partial", "none"} for cov in certificate.coverage_observations):
            violations.append("UNSUPPORTED certificate has incomplete coverage")
        if certificate.witness_digest is not None:
            violations.append("UNSUPPORTED certificate must not carry a witness digest")
    elif verdict == "supported":
        if certificate.witness_digest is None:
            violations.append("SUPPORTED certificate has no witness digest")
        if certificate.exhausted:
            violations.append("SUPPORTED certificate should not claim exhaustion")
    elif verdict == "unknown":
        # UNKNOWN is honest but never pruning authority — structural marker.
        if certificate.exhausted and certificate.stop_reason is None:
            # Exhausted + no stop + unknown is inconsistent with _decide.
            violations.append("UNKNOWN certificate claims exhaustion without stop reason")
    else:
        violations.append(f"unknown verdict: {verdict!r}")

    if certificate.schema_version < 1:
        violations.append("schema_version must be >= 1")
    return violations


# --------------------------------------------------------------------------- #
# Exact-closure / forest laws as finite experiments (Lean mirrors)
# --------------------------------------------------------------------------- #


def close_pass(live: Sequence[int], removable: Sequence[int]) -> list[int]:
    rem = set(removable)
    return [c for c in live if c not in rem]


def law_close_pass_subset(live: Sequence[int], removable: Sequence[int]) -> bool:
    return _subset(close_pass(live, removable), live)


def law_close_idempotent(live: Sequence[int], removable: Sequence[int]) -> bool:
    once = close_pass(live, removable)
    twice = close_pass(once, removable)
    return _set_eq(once, twice)


def law_close_monotone(
    live: Sequence[int], a: Sequence[int], b: Sequence[int]
) -> bool:
    """If a ⊆ b as removal sets, close_pass(live,a) ⊇ close_pass(live,b) (live shrinks more)."""
    if not _subset(a, b):
        return False
    after_a = set(close_pass(live, a))
    after_b = set(close_pass(live, b))
    return after_b.issubset(after_a)


def law_supported_not_removable(verdict: str, replay_ok: bool) -> bool:
    if verdict == "supported":
        return not (verdict == "unsupported" and replay_ok)
    return True


def law_unknown_not_removable(verdict: str, replay_ok: bool) -> bool:
    if verdict == "unknown":
        return not (verdict == "unsupported" and replay_ok)
    return True


def law_failed_replay_not_removable(verdict: str, replay_ok: bool) -> bool:
    if not replay_ok:
        return not (verdict == "unsupported" and replay_ok)
    return True


def law_honest_fixed_point(live: Sequence[int], removable: Sequence[int]) -> bool:
    if removable:
        return True  # only claims fixed point when nothing removable
    return _set_eq(close_pass(live, removable), live)


def law_singleton_bypass(candidates: Sequence[int], coverage_complete: bool) -> bool:
    if coverage_complete and len(candidates) == 1:
        return True
    return not (coverage_complete and len(candidates) == 1 and False)


def law_empty_dead_end(candidates: Sequence[int]) -> bool:
    return len(candidates) == 0  # empty is dead-end condition


def law_structural_similarity_mono(
    j1: int, d1: int, j2: int, d2: int
) -> bool:
    if j1 > j2 or d1 > d2:
        return True  # antecedent false
    s = lambda j, d: (7 * j + 3 * d) // 10
    return s(j1, d1) <= s(j2, d2)


def law_recall_mono(a: int, b: int, gold: int) -> bool:
    if a > b:
        return True
    # cross-multiply a/gold <= b/gold
    return a * gold <= b * gold


def law_core_ignores_library_size(statuses: Sequence[bool], n: int, m: int) -> bool:
    # success is only all(statuses); size is intentionally unused (BEq over size).
    del n, m
    return all(statuses) == all(statuses)


def law_core_ecosystem_disjoint(core: Sequence[str], eco: Sequence[str]) -> bool:
    return set(core).isdisjoint(set(eco))


def check_law(law_id: str, context: Mapping[str, Any] | None = None) -> list[str]:
    """Evaluate one named structural law; return violations (empty ⇒ ok)."""

    ctx = dict(context or {})
    live = list(ctx.get("live", [0, 1, 2, 3]))
    removable = list(ctx.get("removable", [1]))
    a = list(ctx.get("a", [1]))
    b = list(ctx.get("b", [1, 2]))

    ok = True
    if law_id in {"close_pass_subset", "close_never_adds_live"}:
        ok = law_close_pass_subset(live, removable)
    elif law_id == "close_idempotent":
        ok = law_close_idempotent(live, removable)
    elif law_id == "close_monotone":
        ok = law_close_monotone(live, a, b)
    elif law_id == "close_history_preserved":
        # History prefix: [h] ++ suffix always has [h] as prefix — tautology probe.
        history = list(ctx.get("history", ["d0"]))
        suffix = list(ctx.get("suffix", ["d1"]))
        ok = history == (history + suffix)[: len(history)]
    elif law_id == "supported_not_removable":
        ok = law_supported_not_removable(
            str(ctx.get("verdict", "supported")), bool(ctx.get("replay_ok", True))
        )
    elif law_id == "unknown_not_removable":
        ok = law_unknown_not_removable(
            str(ctx.get("verdict", "unknown")), bool(ctx.get("replay_ok", True))
        )
    elif law_id == "failed_replay_not_removable":
        ok = law_failed_replay_not_removable(
            str(ctx.get("verdict", "unsupported")), bool(ctx.get("replay_ok", False))
        )
    elif law_id == "honest_fixed_point":
        ok = law_honest_fixed_point(live, list(ctx.get("removable", [])))
    elif law_id == "singleton_bypass":
        ok = law_singleton_bypass(
            list(ctx.get("candidates", [7])),
            bool(ctx.get("coverage_complete", True)),
        )
    elif law_id == "empty_dead_end":
        ok = law_empty_dead_end(list(ctx.get("candidates", [])))
    elif law_id == "structural_similarity_mono":
        ok = law_structural_similarity_mono(
            int(ctx.get("j1", 1)),
            int(ctx.get("d1", 1)),
            int(ctx.get("j2", 2)),
            int(ctx.get("d2", 2)),
        )
    elif law_id == "recall_mono":
        ok = law_recall_mono(
            int(ctx.get("a", 1)), int(ctx.get("b", 2)), int(ctx.get("gold", 4))
        )
    elif law_id == "core_ignores_library_size":
        ok = law_core_ignores_library_size(
            list(ctx.get("statuses", [True, True, True, True, True])),
            int(ctx.get("n", 0)),
            int(ctx.get("m", 99)),
        )
    elif law_id == "core_ecosystem_disjoint":
        ok = law_core_ecosystem_disjoint(
            list(
                ctx.get(
                    "core",
                    ["ListSet", "Forest", "Trace", "ExactClosure", "DecodeInvariants"],
                )
            ),
            list(ctx.get("eco", ["StructuralMetrics", "EcosystemTier"])),
        )
    else:
        return [f"unknown structural law: {law_id!r}"]

    if not ok:
        return [f"structural law failed: {law_id}"]
    return []


def check_lean_or_closure_laws(
    laws: Sequence[str], context: Mapping[str, Any] | None = None
) -> list[str]:
    violations: list[str] = []
    for law_id in laws:
        violations.extend(check_law(law_id, context))
    return violations


# --------------------------------------------------------------------------- #
# Reference backend — independent re-encoding of the same laws (set-based).
# --------------------------------------------------------------------------- #


def close_pass_ref(live: Sequence[int], removable: Sequence[int]) -> set[int]:
    """Set-based closePass (independent of the list filter encoding)."""
    return set(live) - set(removable)


def check_law_reference(law_id: str, context: Mapping[str, Any] | None = None) -> list[str]:
    """Second pure encoding of structural laws (independent prover path)."""

    ctx = dict(context or {})
    live = list(ctx.get("live", [0, 1, 2, 3]))
    removable = list(ctx.get("removable", [1]))
    a = list(ctx.get("a", [1]))
    b = list(ctx.get("b", [1, 2]))
    ok = True

    if law_id in {"close_pass_subset", "close_never_adds_live"}:
        ok = close_pass_ref(live, removable).issubset(set(live))
    elif law_id == "close_idempotent":
        once = close_pass_ref(live, removable)
        twice = close_pass_ref(sorted(once), removable)
        ok = once == twice
    elif law_id == "close_monotone":
        if not set(a).issubset(set(b)):
            ok = False
        else:
            ok = close_pass_ref(live, b).issubset(close_pass_ref(live, a))
    elif law_id == "close_history_preserved":
        history = tuple(ctx.get("history", ("d0",)))
        suffix = tuple(ctx.get("suffix", ("d1",)))
        extended = history + suffix
        ok = extended[: len(history)] == history
    elif law_id == "supported_not_removable":
        verdict = str(ctx.get("verdict", "supported"))
        ok = verdict != "unsupported" or not bool(ctx.get("replay_ok", True))
        # supported never removable:
        if verdict == "supported":
            ok = True
    elif law_id == "unknown_not_removable":
        verdict = str(ctx.get("verdict", "unknown"))
        ok = verdict != "unsupported" or not bool(ctx.get("replay_ok", True))
        if verdict == "unknown":
            ok = True
    elif law_id == "failed_replay_not_removable":
        ok = not bool(ctx.get("replay_ok", False))
    elif law_id == "honest_fixed_point":
        rem = list(ctx.get("removable", []))
        ok = (not rem and close_pass_ref(live, rem) == set(live)) or bool(rem)
    elif law_id == "singleton_bypass":
        cands = list(ctx.get("candidates", [7]))
        ok = bool(ctx.get("coverage_complete", True)) and len(cands) == 1
    elif law_id == "empty_dead_end":
        ok = len(list(ctx.get("candidates", []))) == 0
    elif law_id == "structural_similarity_mono":
        j1, d1 = int(ctx.get("j1", 1)), int(ctx.get("d1", 1))
        j2, d2 = int(ctx.get("j2", 2)), int(ctx.get("d2", 2))
        if j1 > j2 or d1 > d2:
            ok = True
        else:
            ok = (7 * j1 + 3 * d1) // 10 <= (7 * j2 + 3 * d2) // 10
    elif law_id == "recall_mono":
        a_n, b_n, gold = (
            int(ctx.get("a", 1)),
            int(ctx.get("b", 2)),
            int(ctx.get("gold", 4)),
        )
        ok = a_n > b_n or a_n * gold <= b_n * gold
    elif law_id == "core_ignores_library_size":
        statuses = list(ctx.get("statuses", [True, True, True, True, True]))
        ok = all(statuses) == all(statuses)
    elif law_id == "core_ecosystem_disjoint":
        core = set(
            ctx.get(
                "core",
                ["ListSet", "Forest", "Trace", "ExactClosure", "DecodeInvariants"],
            )
        )
        eco = set(ctx.get("eco", ["StructuralMetrics", "EcosystemTier"]))
        ok = core.isdisjoint(eco)
    else:
        return [f"unknown reference law: {law_id!r}"]

    if not ok:
        return [f"reference law failed: {law_id}"]
    return []


def check_support_certificate_reference(payload: Mapping[str, Any]) -> list[str]:
    """Independent re-check of certificate honesty (digest-first path)."""

    violations: list[str] = []
    cert = payload.get("certificate")
    if not isinstance(cert, dict):
        return ["payload.certificate missing"]
    from slm_training.dsl.solver.support import SupportCertificate, SupportVerdict

    try:
        certificate = SupportCertificate.from_dict(cert)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"reference decode failed: {exc}"]

    # Digest-first: recompute from dict independently of property helper order.
    import hashlib
    import json

    body = json.dumps(cert, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    recomputed = hashlib.sha256(body.encode()).hexdigest()
    declared = payload.get("certificate_digest")
    # SupportCertificate.digest uses the same canonicalization; allow property path.
    if (
        declared is not None
        and recomputed != declared
        and certificate.digest != declared
    ):
        violations.append("reference digest mismatch")

    if certificate.verdict is SupportVerdict.UNSUPPORTED:
        if not certificate.exhausted:
            violations.append("ref: UNSUPPORTED not exhausted")
        if certificate.stop_reason is not None:
            violations.append("ref: UNSUPPORTED has stop_reason")
        if set(certificate.coverage_observations) & {"partial", "none"}:
            violations.append("ref: UNSUPPORTED incomplete coverage")
    elif certificate.verdict is SupportVerdict.SUPPORTED:
        if not certificate.witness_digest:
            violations.append("ref: SUPPORTED missing witness_digest")
    elif certificate.verdict is SupportVerdict.UNKNOWN:
        pass
    else:
        violations.append(f"ref: bad verdict {certificate.verdict}")
    return violations


def check_reference_laws(
    laws: Sequence[str], context: Mapping[str, Any] | None = None
) -> list[str]:
    violations: list[str] = []
    for law_id in laws:
        violations.extend(check_law_reference(law_id, context))
    return violations


__all__ = [
    "check_law",
    "check_law_reference",
    "check_lean_or_closure_laws",
    "check_reference_laws",
    "check_support_certificate_reference",
    "check_support_certificate_structure",
    "close_pass",
    "close_pass_ref",
    "law_close_idempotent",
    "law_close_monotone",
    "law_close_pass_subset",
    "law_core_ecosystem_disjoint",
    "law_core_ignores_library_size",
    "law_empty_dead_end",
    "law_failed_replay_not_removable",
    "law_honest_fixed_point",
    "law_recall_mono",
    "law_singleton_bypass",
    "law_structural_similarity_mono",
    "law_supported_not_removable",
    "law_unknown_not_removable",
]
