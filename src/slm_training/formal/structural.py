"""Pure-Python structural laws mirroring Lean theorems and VSS honesty rules.

This is an **independent prover backend**: it re-derives the declared laws from
the formal object's payload without invoking Lean and without trusting a bare
verdict string. It intentionally uses only the stdlib + lightweight list/set
membership so translation difficulty stays controlled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def removable(verdict: str, replay_ok: bool) -> bool:
    """A candidate is removable only under a replay-confirmed UNSUPPORTED.

    Mirrors ExactClosure's honesty rule: UNKNOWN and replay-invalid results
    never license removal (``docs/design/leverproof-integration.md``). This
    is the shared oracle the three ``*_not_removable`` laws below
    characterize — no live closure-engine decision is wired into this pure
    checker (that integration is tracked as open work, not claimed here);
    what is verified is that the oracle itself agrees with the Lean-mirrored
    intent at every point of its input domain, so a later edit that loosens
    or inverts the definition is caught.
    """
    return verdict == "unsupported" and bool(replay_ok)


def law_supported_not_removable(verdict: str, replay_ok: bool) -> bool:
    return not (verdict == "supported" and removable(verdict, replay_ok))


def law_unknown_not_removable(verdict: str, replay_ok: bool) -> bool:
    return not (verdict == "unknown" and removable(verdict, replay_ok))


def law_failed_replay_not_removable(verdict: str, replay_ok: bool) -> bool:
    return not (not replay_ok and removable(verdict, replay_ok))


def law_honest_fixed_point(live: Sequence[int], removable_ids: Sequence[int]) -> bool:
    if removable_ids:
        return True  # only claims fixed point when nothing removable
    return _set_eq(close_pass(live, removable_ids), live)


def is_singleton_bypass_state(
    candidates: Sequence[int], coverage_complete: bool
) -> bool:
    """KERN-02: bare ``coverage_complete`` has zero singleton authority.

    Forced emission requires ``CompleteDomainEvidenceV1.authorizes_forced_emission``.
    This classifier stays for telemetry-shaped law rows and always returns
    False when only a Boolean coverage flag is supplied.
    """
    del candidates, coverage_complete
    return False


def law_singleton_bypass(candidates: Sequence[int], coverage_complete: bool) -> bool:
    return is_singleton_bypass_state(candidates, coverage_complete)


def law_proved_complete_singleton(
    state_id: str,
    candidates: Sequence[int],
    legal_members: Sequence[int],
    expected_state_id: str,
) -> bool:
    """True when proved complete-domain evidence authorizes forced emission."""
    from slm_training.formal.complete_domain import build_complete_domain

    evidence = build_complete_domain(state_id, candidates, legal_members)
    return evidence.authorizes_forced_emission(expected_state_id=expected_state_id)


def is_empty_dead_end(candidates: Sequence[int]) -> bool:
    """True exactly when the candidate list is the empty dead-end state."""
    return len(candidates) == 0


def law_empty_dead_end(candidates: Sequence[int]) -> bool:
    return is_empty_dead_end(candidates)


def law_structural_similarity_mono(
    j1: int, d1: int, j2: int, d2: int
) -> bool:
    if j1 > j2 or d1 > d2:
        return True  # antecedent false

    def score(j: int, d: int) -> int:
        return (7 * j + 3 * d) // 10

    return score(j1, d1) <= score(j2, d2)


def law_recall_mono(a: int, b: int, gold: int) -> bool:
    if a > b:
        return True
    # cross-multiply a/gold <= b/gold
    return a * gold <= b * gold


def _core_success(statuses: Sequence[bool], n: int, m: int) -> bool:
    # Success is BEq over statuses alone; (n, m) are intentionally unused —
    # that is exactly the invariance the law below verifies by varying them.
    del n, m
    return all(statuses)


def law_core_ignores_library_size(
    statuses: Sequence[bool], n1: int, m1: int, n2: int, m2: int
) -> bool:
    """Core success must not depend on which (library-size) pair is passed."""
    return _core_success(statuses, n1, m1) == _core_success(statuses, n2, m2)


def law_core_ecosystem_disjoint(core: Sequence[str], eco: Sequence[str]) -> bool:
    return set(core).isdisjoint(set(eco))


@dataclass(frozen=True)
class _LawCase:
    """One enumerated instance of a law: inputs plus the expected verdict.

    Universal-implication laws (``expect=True`` in every case) fail on any
    counterexample the case list covers; classifier laws carry mixed
    expectations so both false positives and false negatives are caught.
    """

    context: dict[str, Any]
    expect: bool = True


# Some exported law lists (objects.py) name the same case table under a
# second, Lean-theorem-matching id. Resolve to the canonical table key before
# any case lookup so both backends enumerate the identical case set.
_LAW_ID_ALIASES: dict[str, str] = {
    "close_never_adds_live": "close_pass_subset",
}


def _canonical_law_id(law_id: str) -> str:
    return _LAW_ID_ALIASES.get(law_id, law_id)


# Shared, bounded enumeration used by both independent backends (check_law
# and check_law_reference) so a divergence between the two encodings has a
# real chance to surface, and a single hardcoded default can no longer stand
# in for verification across the law's actual domain.
_LAW_CASES: dict[str, tuple[_LawCase, ...]] = {
    "close_pass_subset": (
        _LawCase({"live": [0, 1, 2, 3], "removable": [1]}),
        _LawCase({"live": [0, 1, 2, 3], "removable": []}),
        _LawCase({"live": [0, 1, 2, 3], "removable": [0, 1, 2, 3]}),
        _LawCase({"live": [0, 1, 2, 3], "removable": [9]}),
        _LawCase({"live": [], "removable": []}),
    ),
    "close_idempotent": (
        _LawCase({"live": [0, 1, 2, 3], "removable": [1]}),
        _LawCase({"live": [0, 1, 2, 3], "removable": []}),
        _LawCase({"live": [0, 1, 2, 3], "removable": [0, 1, 2, 3]}),
        _LawCase({"live": [], "removable": [1]}),
    ),
    "close_monotone": (
        _LawCase({"live": [0, 1, 2, 3], "a": [], "b": [1]}),
        _LawCase({"live": [0, 1, 2, 3], "a": [1], "b": [1, 2]}),
        _LawCase({"live": [0, 1, 2, 3], "a": [1, 2], "b": [1, 2]}),
        _LawCase({"live": [0, 1, 2, 3], "a": [], "b": []}),
    ),
    "close_history_preserved": (
        _LawCase({"history": [], "suffix": []}),
        _LawCase({"history": [], "suffix": ["d1"]}),
        _LawCase({"history": ["d0"], "suffix": []}),
        _LawCase({"history": ["d0", "d1"], "suffix": ["d2", "d3"]}),
    ),
    "supported_not_removable": tuple(
        _LawCase({"verdict": v, "replay_ok": r})
        for v in ("supported", "unsupported", "unknown")
        for r in (True, False)
    ),
    "unknown_not_removable": tuple(
        _LawCase({"verdict": v, "replay_ok": r})
        for v in ("supported", "unsupported", "unknown")
        for r in (True, False)
    ),
    "failed_replay_not_removable": tuple(
        _LawCase({"verdict": v, "replay_ok": r})
        for v in ("supported", "unsupported", "unknown")
        for r in (True, False)
    ),
    "honest_fixed_point": (
        _LawCase({"live": [0, 1, 2], "removable": []}),
        _LawCase({"live": [], "removable": []}),
        _LawCase({"live": [0, 1, 2], "removable": [1]}),
    ),
    "singleton_bypass": (
        _LawCase({"candidates": [], "coverage_complete": True}, expect=False),
        _LawCase({"candidates": [7], "coverage_complete": True}, expect=False),
        _LawCase({"candidates": [7], "coverage_complete": False}, expect=False),
        _LawCase({"candidates": [1, 2], "coverage_complete": True}, expect=False),
        _LawCase({"candidates": [1, 2, 3], "coverage_complete": False}, expect=False),
    ),
    "proved_complete_singleton": (
        _LawCase(
            {
                "state_id": "s1",
                "candidates": [7],
                "legal_members": [7],
                "expected_state_id": "s1",
            },
            expect=True,
        ),
        _LawCase(
            {
                "state_id": "s1",
                "candidates": [7],
                "legal_members": [7],
                "expected_state_id": "stale",
            },
            expect=False,
        ),
        _LawCase(
            {
                "state_id": "s1",
                "candidates": [7],
                "legal_members": [7, 8],
                "expected_state_id": "s1",
            },
            expect=False,
        ),
        _LawCase(
            {
                "state_id": "s1",
                "candidates": [7, 8],
                "legal_members": [7],
                "expected_state_id": "s1",
            },
            expect=False,
        ),
        _LawCase(
            {
                "state_id": "s1",
                "candidates": [7, 7],
                "legal_members": [7],
                "expected_state_id": "s1",
            },
            expect=False,
        ),
    ),
    "empty_dead_end": (
        _LawCase({"candidates": []}, expect=True),
        _LawCase({"candidates": [1]}, expect=False),
        _LawCase({"candidates": [1, 2]}, expect=False),
    ),
    "structural_similarity_mono": (
        _LawCase({"j1": 1, "d1": 1, "j2": 2, "d2": 2}),
        _LawCase({"j1": 0, "d1": 0, "j2": 0, "d2": 0}),
        _LawCase({"j1": 3, "d1": 1, "j2": 3, "d2": 3}),
        _LawCase({"j1": 5, "d1": 5, "j2": 1, "d2": 9}),  # antecedent false
    ),
    "recall_mono": (
        _LawCase({"a": 1, "b": 2, "gold": 4}),
        _LawCase({"a": 0, "b": 0, "gold": 4}),
        _LawCase({"a": 4, "b": 4, "gold": 4}),
        _LawCase({"a": 3, "b": 1, "gold": 4}),  # antecedent false
    ),
    "core_ignores_library_size": (
        _LawCase({"statuses": [True, True, True], "n1": 0, "m1": 0, "n2": 5, "m2": 9}),
        _LawCase(
            {"statuses": [True, False, True], "n1": 1, "m1": 1, "n2": 20, "m2": 20}
        ),
        _LawCase({"statuses": [], "n1": 0, "m1": 0, "n2": 3, "m2": 3}),
    ),
    "core_ecosystem_disjoint": (
        _LawCase(
            {
                "core": [
                    "ListSet",
                    "Forest",
                    "Trace",
                    "ExactClosure",
                    "DecodeInvariants",
                ],
                "eco": ["StructuralMetrics", "EcosystemTier"],
            }
        ),
        _LawCase({"core": [], "eco": ["StructuralMetrics"]}),
        _LawCase({"core": ["Forest"], "eco": []}),
    ),
}


def _eval_law(law_id: str, ctx: Mapping[str, Any]) -> bool:
    """Evaluate the structural-backend law function for one case's context."""
    if law_id in {"close_pass_subset", "close_never_adds_live"}:
        return law_close_pass_subset(list(ctx["live"]), list(ctx["removable"]))
    if law_id == "close_idempotent":
        return law_close_idempotent(list(ctx["live"]), list(ctx["removable"]))
    if law_id == "close_monotone":
        return law_close_monotone(list(ctx["live"]), list(ctx["a"]), list(ctx["b"]))
    if law_id == "close_history_preserved":
        history = list(ctx["history"])
        suffix = list(ctx["suffix"])
        return history == (history + suffix)[: len(history)]
    if law_id == "supported_not_removable":
        return law_supported_not_removable(str(ctx["verdict"]), bool(ctx["replay_ok"]))
    if law_id == "unknown_not_removable":
        return law_unknown_not_removable(str(ctx["verdict"]), bool(ctx["replay_ok"]))
    if law_id == "failed_replay_not_removable":
        return law_failed_replay_not_removable(
            str(ctx["verdict"]), bool(ctx["replay_ok"])
        )
    if law_id == "honest_fixed_point":
        return law_honest_fixed_point(list(ctx["live"]), list(ctx["removable"]))
    if law_id == "singleton_bypass":
        return law_singleton_bypass(
            list(ctx["candidates"]), bool(ctx["coverage_complete"])
        )
    if law_id == "proved_complete_singleton":
        return law_proved_complete_singleton(
            str(ctx["state_id"]),
            list(ctx["candidates"]),
            list(ctx["legal_members"]),
            str(ctx["expected_state_id"]),
        )
    if law_id == "empty_dead_end":
        return law_empty_dead_end(list(ctx["candidates"]))
    if law_id == "structural_similarity_mono":
        return law_structural_similarity_mono(
            int(ctx["j1"]), int(ctx["d1"]), int(ctx["j2"]), int(ctx["d2"])
        )
    if law_id == "recall_mono":
        return law_recall_mono(int(ctx["a"]), int(ctx["b"]), int(ctx["gold"]))
    if law_id == "core_ignores_library_size":
        return law_core_ignores_library_size(
            list(ctx["statuses"]),
            int(ctx["n1"]),
            int(ctx["m1"]),
            int(ctx["n2"]),
            int(ctx["m2"]),
        )
    if law_id == "core_ecosystem_disjoint":
        return law_core_ecosystem_disjoint(list(ctx["core"]), list(ctx["eco"]))
    raise KeyError(law_id)


def check_law(law_id: str, context: Mapping[str, Any] | None = None) -> list[str]:
    """Evaluate one named structural law; return violations (empty ⇒ ok).

    With an explicit ``context`` this checks exactly that one instance
    (single-shot, for a caller that supplies real state). With no context —
    the path every exporter in this codebase actually uses — it enumerates
    the law's bounded case table (:data:`_LAW_CASES`) instead of one
    hardcoded default, so the check can find a real counterexample rather
    than passing vacuously on a single anecdotal input.
    """
    if context is not None:
        try:
            ok = _eval_law(law_id, dict(context))
        except (KeyError, TypeError, ValueError) as exc:
            return [f"structural law {law_id!r} raised on supplied context: {exc}"]
        return [] if ok else [f"structural law failed: {law_id}"]

    cases = _LAW_CASES.get(_canonical_law_id(law_id))
    if cases is None:
        return [f"unknown structural law: {law_id!r}"]
    violations: list[str] = []
    for index, case in enumerate(cases):
        try:
            ok = _eval_law(law_id, case.context)
        except (KeyError, TypeError, ValueError) as exc:
            violations.append(f"structural law {law_id!r} case {index} raised: {exc}")
            continue
        if ok != case.expect:
            violations.append(
                f"structural law failed: {law_id} case {index} "
                f"(got {ok}, expected {case.expect}, context={case.context})"
            )
    return violations


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


def removable_ref(verdict: str, replay_ok: bool) -> bool:
    """Independent (dict-lookup) re-encoding of the removable oracle.

    Deliberately not a call to :func:`removable` — a shared helper would let
    a single bug in the shared definition pass both backends. This is the
    same rule (``UNSUPPORTED`` + replay-confirmed only) expressed a different
    way, so the two backends can disagree if either encoding drifts.
    """
    return {"unsupported": bool(replay_ok)}.get(verdict, False)


def _core_success_ref(statuses: Sequence[bool], n: int, m: int) -> bool:
    del n, m
    return all(statuses)


def _eval_law_reference(law_id: str, ctx: Mapping[str, Any]) -> bool:
    """Evaluate the reference-backend (set-based) law encoding for one case."""
    canonical = _canonical_law_id(law_id)
    if canonical == "close_pass_subset":
        live = list(ctx["live"])
        removable = list(ctx["removable"])
        return close_pass_ref(live, removable).issubset(set(live))
    if canonical == "close_idempotent":
        live = list(ctx["live"])
        removable = list(ctx["removable"])
        once = close_pass_ref(live, removable)
        twice = close_pass_ref(sorted(once), removable)
        return once == twice
    if canonical == "close_monotone":
        live = list(ctx["live"])
        a = list(ctx["a"])
        b = list(ctx["b"])
        if not set(a).issubset(set(b)):
            return False
        return close_pass_ref(live, b).issubset(close_pass_ref(live, a))
    if canonical == "close_history_preserved":
        history = tuple(ctx["history"])
        suffix = tuple(ctx["suffix"])
        extended = history + suffix
        return extended[: len(history)] == history
    if canonical == "supported_not_removable":
        verdict = str(ctx["verdict"])
        replay_ok = bool(ctx["replay_ok"])
        return not (verdict == "supported" and removable_ref(verdict, replay_ok))
    if canonical == "unknown_not_removable":
        verdict = str(ctx["verdict"])
        replay_ok = bool(ctx["replay_ok"])
        return not (verdict == "unknown" and removable_ref(verdict, replay_ok))
    if canonical == "failed_replay_not_removable":
        verdict = str(ctx["verdict"])
        replay_ok = bool(ctx["replay_ok"])
        return not (not replay_ok and removable_ref(verdict, replay_ok))
    if canonical == "honest_fixed_point":
        live = list(ctx["live"])
        rem = list(ctx["removable"])
        if rem:
            return True
        return close_pass_ref(live, rem) == set(live)
    if canonical == "singleton_bypass":
        return law_singleton_bypass(
            list(ctx["candidates"]), bool(ctx["coverage_complete"])
        )
    if canonical == "proved_complete_singleton":
        return law_proved_complete_singleton(
            str(ctx["state_id"]),
            list(ctx["candidates"]),
            list(ctx["legal_members"]),
            str(ctx["expected_state_id"]),
        )
    if canonical == "empty_dead_end":
        return len(list(ctx["candidates"])) == 0
    if canonical == "structural_similarity_mono":
        j1, d1 = int(ctx["j1"]), int(ctx["d1"])
        j2, d2 = int(ctx["j2"]), int(ctx["d2"])
        if j1 > j2 or d1 > d2:
            return True
        return (7 * j1 + 3 * d1) // 10 <= (7 * j2 + 3 * d2) // 10
    if canonical == "recall_mono":
        a_n, b_n, gold = int(ctx["a"]), int(ctx["b"]), int(ctx["gold"])
        return a_n > b_n or a_n * gold <= b_n * gold
    if canonical == "core_ignores_library_size":
        statuses = list(ctx["statuses"])
        n1, m1 = int(ctx["n1"]), int(ctx["m1"])
        n2, m2 = int(ctx["n2"]), int(ctx["m2"])
        return _core_success_ref(statuses, n1, m1) == _core_success_ref(statuses, n2, m2)
    if canonical == "core_ecosystem_disjoint":
        core = set(ctx["core"])
        eco = set(ctx["eco"])
        return core.isdisjoint(eco)
    raise KeyError(law_id)


def check_law_reference(law_id: str, context: Mapping[str, Any] | None = None) -> list[str]:
    """Second, independent pure encoding of structural laws (set-based path).

    Mirrors :func:`check_law`'s enumeration behavior: with no explicit
    context (every real exporter's path) it checks the same bounded case
    table (:data:`_LAW_CASES`) as the structural backend through a wholly
    separate evaluation function, so an encoding bug in one backend has a
    real chance of being caught by the other rather than both silently
    agreeing on a single hardcoded default.
    """
    if context is not None:
        try:
            ok = _eval_law_reference(law_id, dict(context))
        except KeyError:
            return [f"unknown reference law: {law_id!r}"]
        except (TypeError, ValueError) as exc:
            return [f"reference law {law_id!r} raised on supplied context: {exc}"]
        return [] if ok else [f"reference law failed: {law_id}"]

    cases = _LAW_CASES.get(_canonical_law_id(law_id))
    if cases is None:
        return [f"unknown reference law: {law_id!r}"]
    violations: list[str] = []
    for index, case in enumerate(cases):
        try:
            ok = _eval_law_reference(law_id, case.context)
        except (KeyError, TypeError, ValueError) as exc:
            violations.append(f"reference law {law_id!r} case {index} raised: {exc}")
            continue
        if ok != case.expect:
            violations.append(
                f"reference law failed: {law_id} case {index} "
                f"(got {ok}, expected {case.expect}, context={case.context})"
            )
    return violations


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
    "law_proved_complete_singleton",
    "law_structural_similarity_mono",
    "law_supported_not_removable",
    "law_unknown_not_removable",
]
