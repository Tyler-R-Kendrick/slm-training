"""Tests for deterministic bounded domain-obstruction cores (PGS-C02 / SLM-500, PGS-G02 / SLM-508)."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import replace

import pytest
from pydantic import ValidationError

from slm_training.dsl.solver.goal_support import (
    DOMAIN_OBSTRUCTION_CLAIM,
    GoalFailureAtomV1,
    GoalTerminalEvidenceV1,
    GoalUnknownAtomV1,
    exact_mandatory_failure_atom_ids,
    obstruction_core_algorithm_table,
    obstruction_core_emission_allowed,
)
from slm_training.dsl.solver.goal_support_obstruction import (
    DomainObstructionCoreV1,
    TerminalFailureSetV1,
    compute_domain_obstruction_core,
    validate_domain_obstruction_core,
)
from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleId, SolverBounds, SupportVerdict
from slm_training.dsl.solver.support import SupportCertificate, SupportQuery
from slm_training.formal.goal_support_mapping import (
    TerminalFailureSetV1 as FormalTerminalFailureSetV1,
    hits_all as ref_hits_all,
    subset_minimal_exact as ref_subset_minimal_exact,
)
from tests.test_dsl.test_solver.test_goal_support import _profile

_SEED = 508
_MAX_GENERATED_FAILURE_FAMILIES = 48
_MAX_ATOM_UNIVERSE = 6
_MAX_TERMINALS_PER_FAMILY = 5


def _bounds() -> SolverBounds:
    return SolverBounds(
        max_tokens=10_000,
        max_nodes=10_000,
        max_depth=32,
        max_backtracks=10_000,
        max_verifier_calls=10_000,
    )


def _state(**overrides: object) -> FiniteDomainState:
    defaults: dict[str, object] = {
        "problem_id": "fixture/obstruction",
        "pack_id": "fixture",
        "constraint_version": "v1",
        "bounds": _bounds(),
        "holes": (),
    }
    defaults.update(overrides)
    return FiniteDomainState(**defaults)


def _failure_atom(atom_id: str, **overrides: object) -> GoalFailureAtomV1:
    defaults: dict[str, object] = {
        "atom_id": atom_id,
        "reason_code": "fixture_fail",
        "source_kind": "constraint",
        "completeness_class": "EXACT",
    }
    defaults.update(overrides)
    return GoalFailureAtomV1(**defaults)


def _terminal(
    digest_seed: str,
    *,
    overall_status: str = "REJECT",
    failure_atoms: tuple[GoalFailureAtomV1, ...] = (),
    unknown_atoms: tuple[GoalUnknownAtomV1, ...] = (),
) -> GoalTerminalEvidenceV1:
    digest = hashlib.sha256(digest_seed.encode()).hexdigest()
    profile = _profile()
    payload = GoalTerminalEvidenceV1(
        profile_digest=profile.digest,
        program_digest=digest,
        canonical_program_digest=digest,
        program_spec_digest="3" * 64,
        semantic_plan_digest="4" * 64,
        mandatory_failure_atoms=failure_atoms,
        mandatory_unknown_atoms=unknown_atoms,
        structural_status=overall_status,
        overall_status=overall_status,
    ).to_dict(include_digest=True)
    return GoalTerminalEvidenceV1.from_dict(payload)


def _certificate(
    *,
    verdict: SupportVerdict = SupportVerdict.UNSUPPORTED,
    exhausted: bool = True,
    coverage: tuple[str, ...] = ("complete",),
    stop_reason: str | None = None,
) -> SupportCertificate:
    profile = _profile()
    state = _state()
    hole = HoleId(namespace="fixture", path=(0, "ROOT"), kind="next")
    candidate = DomainValue.create("letter", {"letter": "a"})
    query = SupportQuery(
        state_fingerprint=state.fingerprint,
        hole_id=hole,
        candidate=candidate,
    )
    return SupportCertificate(
        schema_version=1,
        query=query,
        verdict=verdict,
        problem_id=state.problem_id,
        pack_id=state.pack_id,
        constraint_version=state.constraint_version,
        bounds=state.bounds,
        search_order="dfs",
        explored_state_fingerprints=(),
        coverage_observations=coverage,
        verifier_profile=f"openui/goal-support/v1/{profile.digest}",
        exhausted=exhausted,
        stop_reason=stop_reason,
    )


def _failure_sets(spec: dict[str, tuple[str, ...]]) -> tuple[TerminalFailureSetV1, ...]:
    return tuple(
        TerminalFailureSetV1(
            terminal_evidence_digest=hashlib.sha256(name.encode()).hexdigest(),
            exact_mandatory_failure_atom_ids=atoms,
        )
        for name, atoms in sorted(spec.items())
    )


def _brute_force_min_hitting(
    failure_sets: tuple[TerminalFailureSetV1, ...],
    atom_universe: tuple[str, ...],
) -> tuple[str, ...]:
    for size in range(1, len(atom_universe) + 1):
        best: tuple[str, ...] | None = None
        for combo in itertools.combinations(atom_universe, size):
            candidate = tuple(sorted(combo))
            core_set = set(candidate)
            if all(
                core_set & set(item.exact_mandatory_failure_atom_ids)
                for item in failure_sets
            ):
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            return best
    raise AssertionError("no hitting set")


def _ref_formal_failures(
    failure_sets: tuple[TerminalFailureSetV1, ...],
) -> tuple[FormalTerminalFailureSetV1, ...]:
    """Map production failure sets to int-keyed formal reference sets."""
    atom_to_int = {
        atom: index
        for index, atom in enumerate(
            sorted({a for item in failure_sets for a in item.exact_mandatory_failure_atom_ids})
        )
    }
    return tuple(
        FormalTerminalFailureSetV1(
            terminal=index,
            atoms=tuple(atom_to_int[atom] for atom in item.exact_mandatory_failure_atom_ids),
        )
        for index, item in enumerate(failure_sets)
    )


def _ref_core_ints(core: tuple[str, ...], failure_sets: tuple[TerminalFailureSetV1, ...]) -> tuple[int, ...]:
    atom_to_int = {
        atom: index
        for index, atom in enumerate(
            sorted({a for item in failure_sets for a in item.exact_mandatory_failure_atom_ids})
        )
    }
    return tuple(atom_to_int[atom] for atom in core)


def _ref_subset_minimal_remove_any_breaks(
    core: tuple[str, ...],
    failure_sets: tuple[TerminalFailureSetV1, ...],
) -> bool:
    formal_failures = _ref_formal_failures(failure_sets)
    formal_core = _ref_core_ints(core, failure_sets)
    if not ref_hits_all(formal_core, formal_failures):
        return False
    for atom in formal_core:
        trial = tuple(item for item in formal_core if item != atom)
        if trial and ref_hits_all(trial, formal_failures):
            return False
    return True


def _rng() -> random.Random:
    return random.Random(_SEED)


def _generated_failure_family(rng: random.Random) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    universe_size = rng.randint(2, _MAX_ATOM_UNIVERSE)
    universe = tuple(f"a{i}" for i in range(universe_size))
    terminal_count = rng.randint(1, _MAX_TERMINALS_PER_FAMILY)
    specs: dict[str, tuple[str, ...]] = {}
    for index in range(terminal_count):
        pick_count = rng.randint(1, min(3, universe_size))
        atoms = tuple(sorted(rng.sample(list(universe), pick_count)))
        specs[f"t{index}"] = atoms
    return specs, universe


def _core_from_records(
    records: tuple[GoalTerminalEvidenceV1, ...],
    *,
    mode: str | None = None,
    replay_ok: bool = True,
) -> DomainObstructionCoreV1 | None:
    profile = _profile()
    state = _state()
    cert = _certificate()
    return compute_domain_obstruction_core(
        certificate=cert,
        terminal_records=records,
        profile=profile,
        state=state,
        replay_ok=replay_ok,
        mode=mode,
    )


def test_minimum_cardinality_matches_brute_force_exhaustive() -> None:
    universe = ("a1", "a2", "a3", "a4")
    specs = {
        "t0": ("a1", "a2"),
        "t1": ("a2", "a3"),
        "t2": ("a3", "a4"),
    }
    failure_sets = _failure_sets(specs)
    expected = _brute_force_min_hitting(failure_sets, universe)
    records = tuple(
        _terminal(name, failure_atoms=tuple(_failure_atom(a) for a in atoms))
        for name, atoms in specs.items()
    )
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    assert tuple(core.core_atoms) == expected
    assert core.stats.subsets_examined > 0


def test_lexical_tie_break_on_equal_cardinality() -> None:
    specs = {
        "t0": ("a", "b"),
        "t1": ("a", "c"),
        "t2": ("b", "c"),
    }
    records = tuple(
        _terminal(name, failure_atoms=tuple(_failure_atom(x) for x in atoms))
        for name, atoms in specs.items()
    )
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    assert tuple(core.core_atoms) == ("a", "b")


def test_subset_minimal_remove_any_breaks() -> None:
    specs = {
        "t0": ("x", "y"),
        "t1": ("y", "z"),
        "t2": ("x", "z"),
    }
    records = tuple(
        _terminal(name, failure_atoms=tuple(_failure_atom(a) for a in atoms))
        for name, atoms in specs.items()
    )
    core = _core_from_records(records, mode="subset_minimal_exact")
    assert core is not None
    assert len(core.core_atoms) >= 2
    ok, violations = validate_domain_obstruction_core(
        core,
        certificate=_certificate(),
        terminal_records=records,
        profile=_profile(),
        state=_state(),
        replay_ok=True,
    )
    assert ok, violations
    for atom in core.core_atoms:
        tampered_atoms = tuple(item for item in core.core_atoms if item != atom)
        payload = core.to_dict()
        payload["core_atoms"] = tampered_atoms
        payload.pop("core_digest")
        bad = DomainObstructionCoreV1(**payload)
        ok2, violations2 = validate_domain_obstruction_core(
            bad,
            certificate=_certificate(),
            terminal_records=records,
            profile=_profile(),
            state=_state(),
            replay_ok=True,
        )
        assert not ok2
        assert any(
            "removable" in item or "does not hit" in item or "empty core_atoms" in item
            for item in violations2
        )


def test_sound_overapprox_hits_without_minimality_claim() -> None:
    specs = {f"t{i}": (f"a{i}", f"a{i+10}") for i in range(20)}
    records = tuple(
        _terminal(name, failure_atoms=tuple(_failure_atom(a) for a in atoms))
        for name, atoms in specs.items()
    )
    core = _core_from_records(records, mode="sound_overapprox")
    assert core is not None
    assert core.mode == "sound_overapprox"
    ok, violations = validate_domain_obstruction_core(
        core,
        certificate=_certificate(),
        terminal_records=records,
        profile=_profile(),
        state=_state(),
        replay_ok=True,
    )
    assert ok, violations


def test_permutation_invariance_canonicalizes_identically() -> None:
    atoms_a = (_failure_atom("a1"), _failure_atom("a2"))
    atoms_b = (_failure_atom("a2"), _failure_atom("a3"))
    r1 = (_terminal("t0", failure_atoms=atoms_a), _terminal("t1", failure_atoms=atoms_b))
    r2 = (_terminal("t1", failure_atoms=atoms_b), _terminal("t0", failure_atoms=atoms_a))
    c1 = _core_from_records(r1, mode="minimum_cardinality_exact")
    c2 = _core_from_records(r2, mode="minimum_cardinality_exact")
    assert c1 is not None and c2 is not None
    assert c1.compute_digest() == c2.compute_digest()
    assert tuple(c1.core_atoms) == tuple(c2.core_atoms)


def test_zero_rejected_terminal_rejects_emission() -> None:
    records = (_terminal("accept", overall_status="ACCEPT"),)
    assert not obstruction_core_emission_allowed(
        certificate=_certificate(),
        terminal_records=records,
        profile=_profile(),
        replay_ok=True,
    )
    assert _core_from_records(records) is None


def test_mandatory_unknown_rejects_emission() -> None:
    records = (
        _terminal(
            "reject",
            failure_atoms=(_failure_atom("a1"),),
            unknown_atoms=(
                GoalUnknownAtomV1(
                    atom_id="u1",
                    reason_code="skipped",
                    source_kind="constraint",
                ),
            ),
        ),
    )
    assert not obstruction_core_emission_allowed(
        certificate=_certificate(),
        terminal_records=records,
        profile=_profile(),
        replay_ok=True,
    )
    assert exact_mandatory_failure_atom_ids(records[0]) == ()
    assert _core_from_records(records) is None


def test_heuristic_only_failure_rejects_emission() -> None:
    with pytest.raises(ValidationError, match="mandatory_failure_atoms cannot claim EXACT"):
        _terminal(
            "reject",
            failure_atoms=(
                _failure_atom(
                    "h1",
                    completeness_class="HEURISTIC",
                    source_kind="gate",
                    source_identity="G11",
                ),
            ),
        )


def test_stale_identity_fails_validation() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    stale_state = _state(problem_id="changed")
    ok, violations = validate_domain_obstruction_core(
        core,
        certificate=_certificate(),
        terminal_records=records,
        profile=_profile(),
        state=stale_state,
        replay_ok=True,
    )
    assert not ok
    assert any("state_fingerprint" in item for item in violations)


def test_candidate_addition_stale_bounds_digest() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    cert = _certificate()
    stale_cert = SupportCertificate.from_dict(cert.to_dict())
    stale_cert = SupportCertificate(
        schema_version=stale_cert.schema_version,
        query=stale_cert.query,
        verdict=stale_cert.verdict,
        problem_id=stale_cert.problem_id,
        pack_id=stale_cert.pack_id,
        constraint_version=stale_cert.constraint_version,
        bounds=SolverBounds(
            max_tokens=999,
            max_nodes=stale_cert.bounds.max_nodes,
            max_depth=stale_cert.bounds.max_depth,
            max_backtracks=stale_cert.bounds.max_backtracks,
            max_verifier_calls=stale_cert.bounds.max_verifier_calls,
        ),
        search_order=stale_cert.search_order,
        explored_state_fingerprints=stale_cert.explored_state_fingerprints,
        coverage_observations=stale_cert.coverage_observations,
        verifier_profile=stale_cert.verifier_profile,
        exhausted=stale_cert.exhausted,
        stop_reason=stale_cert.stop_reason,
    )
    ok, violations = validate_domain_obstruction_core(
        core,
        certificate=stale_cert,
        terminal_records=records,
        profile=_profile(),
        state=_state(),
        replay_ok=True,
    )
    assert not ok
    assert any("bounds_digest" in item for item in violations)


def test_tampered_core_digest_rejected() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    payload = core.to_dict()
    payload["core_digest"] = "f" * 64
    with pytest.raises(ValueError, match="core_digest does not match"):
        DomainObstructionCoreV1.from_dict(payload)


def test_empty_core_atoms_rejected_on_validation() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    payload = core.to_dict()
    payload["core_atoms"] = []
    payload.pop("core_digest")
    bad = DomainObstructionCoreV1(**payload)
    ok, violations = validate_domain_obstruction_core(
        bad,
        certificate=_certificate(),
        terminal_records=records,
        profile=_profile(),
        state=_state(),
        replay_ok=True,
    )
    assert not ok
    assert any("empty core_atoms" in item for item in violations)


def test_dangling_atom_rejected() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    payload = core.to_dict()
    payload["core_atoms"] = ["a1", "missing"]
    payload.pop("core_digest")
    bad = DomainObstructionCoreV1(**payload)
    ok, violations = validate_domain_obstruction_core(
        bad,
        certificate=_certificate(),
        terminal_records=records,
        profile=_profile(),
        state=_state(),
        replay_ok=True,
    )
    assert not ok
    assert any("dangling" in item for item in violations)


def test_supported_verdict_never_emits_core() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    cert = _certificate(verdict=SupportVerdict.SUPPORTED)
    core = compute_domain_obstruction_core(
        certificate=cert,
        terminal_records=records,
        profile=_profile(),
        state=_state(),
        replay_ok=True,
    )
    assert core is None


def test_unknown_verdict_never_emits_core() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    cert = _certificate(
        verdict=SupportVerdict.UNKNOWN,
        exhausted=False,
        coverage=("partial",),
    )
    core = compute_domain_obstruction_core(
        certificate=cert,
        terminal_records=records,
        profile=_profile(),
        state=_state(),
        replay_ok=True,
    )
    assert core is None


def test_partial_coverage_never_emits_core() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    cert = _certificate(coverage=("partial",))
    assert not obstruction_core_emission_allowed(
        certificate=cert,
        terminal_records=records,
        profile=_profile(),
        replay_ok=True,
    )


def test_cross_profile_terminal_or_certificate_cannot_emit_core() -> None:
    record = _terminal("t0", failure_atoms=(_failure_atom("a1"),))
    stale_record = record.model_copy(
        update={"profile_digest": "b" * 64, "evidence_digest": ""}
    )
    profile = _profile()
    state = _state()
    certificate = _certificate()

    assert compute_domain_obstruction_core(
        certificate=certificate,
        terminal_records=(stale_record,),
        profile=profile,
        state=state,
        replay_ok=True,
    ) is None
    assert compute_domain_obstruction_core(
        certificate=replace(certificate, verifier_profile="stale/profile"),
        terminal_records=(record,),
        profile=profile,
        state=state,
        replay_ok=True,
    ) is None


def test_duplicate_core_atoms_rejected_at_parse() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    payload = core.to_dict()
    payload["core_atoms"] = ["a1", "a1"]
    payload.pop("core_digest")
    with pytest.raises(ValidationError, match="core_atoms must be unique"):
        DomainObstructionCoreV1(**payload)


def test_obstruction_core_algorithm_table_documents_thresholds() -> None:
    table = obstruction_core_algorithm_table()
    assert table["claim"] == DOMAIN_OBSTRUCTION_CLAIM
    assert "minimum_cardinality_exact" in table["modes"]
    assert table["min_cardinality_exact_atom_threshold"] >= 1


def test_round_trip_core_json() -> None:
    records = (
        _terminal("t0", failure_atoms=(_failure_atom("a1"), _failure_atom("a2"))),
        _terminal("t1", failure_atoms=(_failure_atom("a2"),)),
    )
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    restored = DomainObstructionCoreV1.from_dict(core.to_dict())
    assert restored == core


# --------------------------------------------------------------------------- #
# PGS-G02 / SLM-508 — independent reference models + generated properties
# --------------------------------------------------------------------------- #


def test_generated_minimum_cardinality_matches_brute_force_reference() -> None:
    rng = _rng()
    checked = 0
    for _ in range(_MAX_GENERATED_FAILURE_FAMILIES):
        specs, universe = _generated_failure_family(rng)
        failure_sets = _failure_sets(specs)
        try:
            expected = _brute_force_min_hitting(failure_sets, universe)
        except AssertionError:
            continue
        records = tuple(
            _terminal(name, failure_atoms=tuple(_failure_atom(a) for a in atoms))
            for name, atoms in specs.items()
        )
        core = _core_from_records(records, mode="minimum_cardinality_exact")
        assert core is not None, {"specs": specs, "universe": universe}
        assert tuple(core.core_atoms) == expected, {
            "specs": specs,
            "expected": expected,
            "got": core.core_atoms,
        }
        formal_failures = _ref_formal_failures(failure_sets)
        formal_core = _ref_core_ints(tuple(core.core_atoms), failure_sets)
        assert ref_hits_all(formal_core, formal_failures)
        checked += 1
    assert checked >= 8, "too few generated hitting-set families exercised"


def test_generated_subset_minimal_loses_hitting_after_any_removal() -> None:
    specs = {f"t{i}": tuple(f"a{j}" for j in range(i, i + 2)) for i in range(4)}
    records = tuple(
        _terminal(name, failure_atoms=tuple(_failure_atom(a) for a in atoms))
        for name, atoms in specs.items()
    )
    core = _core_from_records(records, mode="subset_minimal_exact")
    assert core is not None
    failure_sets = _failure_sets(specs)
    assert _ref_subset_minimal_remove_any_breaks(tuple(core.core_atoms), failure_sets)
    formal_core = _ref_core_ints(tuple(core.core_atoms), failure_sets)
    assert ref_subset_minimal_exact(formal_core, _ref_formal_failures(failure_sets))


def test_empty_mandatory_failure_set_forbids_exact_core() -> None:
    records = (
        _terminal(
            "reject_no_atoms",
            failure_atoms=(),
            unknown_atoms=(),
        ),
    )
    assert not obstruction_core_emission_allowed(
        certificate=_certificate(),
        terminal_records=records,
        replay_ok=True,
    )
    assert _core_from_records(records) is None


def test_duplicate_terminal_records_reject_core_construction() -> None:
    atom = _failure_atom("dup")
    record = _terminal("dup", failure_atoms=(atom,))
    records = (record, record)
    with pytest.raises(ValidationError, match="terminal_failure_sets digests must be unique"):
        _core_from_records(records, mode="minimum_cardinality_exact")


def test_terminal_order_permutation_preserves_core_digest() -> None:
    specs = {"t0": ("a1", "a2"), "t1": ("a2", "a3"), "t2": ("a1", "a3")}
    records_a = tuple(
        _terminal(name, failure_atoms=tuple(_failure_atom(a) for a in atoms))
        for name, atoms in specs.items()
    )
    records_b = tuple(reversed(records_a))
    core_a = _core_from_records(records_a, mode="minimum_cardinality_exact")
    core_b = _core_from_records(records_b, mode="minimum_cardinality_exact")
    assert core_a is not None and core_b is not None
    assert core_a.compute_digest() == core_b.compute_digest()
    assert tuple(core_a.core_atoms) == tuple(core_b.core_atoms)


def test_candidate_addition_invalidates_old_obstruction_core_identity() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    core_before = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core_before is not None
    stale_state = _state(problem_id="enriched-domain")
    ok, violations = validate_domain_obstruction_core(
        core_before,
        certificate=_certificate(),
        terminal_records=records,
        profile=_profile(),
        state=stale_state,
        replay_ok=True,
    )
    assert not ok
    assert any("state_fingerprint" in item for item in violations)


def test_profile_digest_mutation_stales_obstruction_core() -> None:
    records = (_terminal("t0", failure_atoms=(_failure_atom("a1"),)),)
    core = _core_from_records(records, mode="minimum_cardinality_exact")
    assert core is not None
    stale_profile = _profile(profile_id="mutated-profile")
    ok, violations = validate_domain_obstruction_core(
        core,
        certificate=_certificate(),
        terminal_records=records,
        profile=stale_profile,
        state=_state(),
        replay_ok=True,
    )
    assert not ok
    assert any("profile_digest" in item for item in violations)


def test_sound_overapprox_never_claims_minimality_mode() -> None:
    specs = {"t0": ("a", "b"), "t1": ("b", "c"), "t2": ("d", "e")}
    records = tuple(
        _terminal(name, failure_atoms=tuple(_failure_atom(a) for a in atoms))
        for name, atoms in specs.items()
    )
    core = _core_from_records(records, mode="sound_overapprox")
    assert core is not None
    assert core.mode == "sound_overapprox"
    assert core.mode != "subset_minimal_exact"
    assert core.mode != "minimum_cardinality_exact"
    failure_sets = _failure_sets(specs)
    formal_core = _ref_core_ints(tuple(core.core_atoms), failure_sets)
    assert ref_hits_all(formal_core, _ref_formal_failures(failure_sets))


def test_obstruction_reference_models_are_independent_of_validate() -> None:
    """Reference hitting/subset-minimal checks do not call validate_domain_obstruction_core."""
    specs = {"t0": ("x", "y"), "t1": ("y", "z")}
    failure_sets = _failure_sets(specs)
    expected = _brute_force_min_hitting(failure_sets, ("x", "y", "z"))
    formal = _ref_formal_failures(failure_sets)
    formal_core = _ref_core_ints(expected, failure_sets)
    assert ref_hits_all(formal_core, formal)
    assert ref_subset_minimal_exact(formal_core, formal)
