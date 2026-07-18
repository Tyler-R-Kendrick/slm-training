"""Tri-state support oracle and replayable-certificate regressions (VSS0-04).

The honesty invariants exercised here are owned by
``docs/design/verified-scope-solver.md``: ``UNKNOWN`` never becomes
``UNSUPPORTED``; ``UNSUPPORTED`` requires full exhaustion with complete coverage;
certificates carry only replay-safe evidence (a witness *digest*, never raw text).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from slm_training.dsl.solver import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
    SupportVerdict,
)
from slm_training.dsl.solver.support import (
    REFERENCE_SEARCH_ORDER,
    SUPPORT_CERTIFICATE_SCHEMA_VERSION,
    EnumerativeSupportOracle,
    Expansion,
    ExpansionEdge,
    SearchCounters,
    SupportCertificate,
    SupportQuery,
    VerifierOutcome,
    get_support_backend,
    register_support_backend,
    replay_support_certificate,
    support_backends,
)

BIG = SolverBounds(9999, 9999, 9999, 9999, 9999)
HID = HoleId("completion_forest", (0, "h"), "next_semantic_decision")
META = (("coverage", "complete"), ("support_verdict", "unknown"))


def path_value(kind: str, token_ids: list[int]) -> DomainValue:
    return DomainValue.create("completion_path", {"kind": kind, "token_ids": list(token_ids)})


def terminal_value(source: str) -> DomainValue:
    return DomainValue.create("terminal", {"source": source})


def projection(
    problem: str,
    values: tuple[DomainValue, ...],
    *,
    bounds: SolverBounds = BIG,
    constraint_version: str = "v1",
) -> FiniteDomainState:
    return FiniteDomainState(
        problem, "openui", constraint_version, bounds, (HoleDomain(HID, values, META),)
    )


def query_for(state: FiniteDomainState, candidate: DomainValue) -> SupportQuery:
    return SupportQuery(state.fingerprint, HID, candidate)


def source_decoder(terminal: DomainValue) -> str | None:
    payload = terminal.payload
    return payload.get("source") if isinstance(payload, dict) else None


def accept_only(*accepted: str, profile: str = "fixture-oracle", capability: bool = True):
    accepted_set = set(accepted)

    def verifier(terminal: str) -> VerifierOutcome:
        return VerifierOutcome(terminal in accepted_set, profile, ("oracle",), capability)

    return verifier


def dict_expander(mapping: dict[str, Expansion]):
    def expander(state: FiniteDomainState) -> Expansion:
        return mapping.get(state.fingerprint, Expansion("complete"))

    return expander


def oracle_for(mapping, *, verifier, decoder=source_decoder, max_steps=None):
    return EnumerativeSupportOracle(
        expander=dict_expander(mapping),
        decoder=decoder,
        verifier=verifier,
        max_steps=max_steps,
    )


def single_leaf_scenario(*, coverage="complete", source="root = Ok", bounds=BIG):
    """base -> root(candidate) -> one edge -> one terminal leaf with ``source``."""
    candidate = path_value("A", [1])
    base = projection("base", (candidate, path_value("B", [2])), bounds=bounds)
    root = base.with_decision(HID, candidate)
    leaf = projection("leaf", (path_value("z", [0]),), bounds=bounds)
    mapping = {
        root.fingerprint: Expansion(coverage, (ExpansionEdge(path_value("go", [3]), leaf),)),
        leaf.fingerprint: Expansion("complete", (), terminal_value(source)),
    }
    return base, candidate, mapping


def leaves_scenario(bounds, n_leaves, *, terminal=False, source="root = Bad"):
    candidate = path_value("A", [1])
    base = projection("base", (candidate,), bounds=bounds)
    root = base.with_decision(HID, candidate)
    leaves = [
        projection(f"leaf{index}", (path_value("z", [0]),), bounds=bounds)
        for index in range(n_leaves)
    ]
    edges = tuple(
        ExpansionEdge(path_value(f"e{index}", [index + 1]), leaves[index])
        for index in range(n_leaves)
    )
    mapping = {root.fingerprint: Expansion("complete", edges)}
    for leaf in leaves:
        mapping[leaf.fingerprint] = Expansion(
            "complete", (), terminal_value(source) if terminal else None
        )
    return base, candidate, mapping


# --------------------------------------------------------------------------
# Tri-state verdicts
# --------------------------------------------------------------------------


def test_supported_with_witness_records_digest_not_raw_text() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Ok")
    result = oracle_for(mapping, verifier=accept_only("root = Ok")).check(
        base, query_for(base, candidate)
    )
    cert = result.certificate
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.witness == "root = Ok"
    assert cert.witness_digest == hashlib.sha256(b"root = Ok").hexdigest()
    assert cert.witness_source == "enumerative"
    assert cert.verifier_profile == "fixture-oracle"
    assert cert.exhausted is False and cert.stop_reason is None
    # The raw witness text is never embedded in the certificate.
    assert "root = Ok" not in json.dumps(cert.to_dict())


def test_unsupported_only_after_complete_exhaustion() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Bad")
    result = oracle_for(mapping, verifier=accept_only("root = Ok")).check(
        base, query_for(base, candidate)
    )
    cert = result.certificate
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert cert.exhausted is True
    assert cert.coverage_observations == ("complete",)
    assert cert.stop_reason is None
    assert cert.witness_digest is None
    assert cert.failure_counts == (("verifier_rejected", 1),)


def test_unknown_from_partial_coverage_never_unsupported() -> None:
    base, candidate, mapping = single_leaf_scenario(coverage="partial", source="root = Bad")
    result = oracle_for(mapping, verifier=accept_only("root = Ok")).check(
        base, query_for(base, candidate)
    )
    cert = result.certificate
    assert result.verdict is SupportVerdict.UNKNOWN
    assert "partial" in cert.coverage_observations
    assert cert.stop_reason == "incomplete_coverage"
    assert cert.witness_digest is None


def _nodes_case():
    base, candidate, mapping = leaves_scenario(dataclasses.replace(BIG, max_nodes=2), 3)
    return base, candidate, oracle_for(mapping, verifier=accept_only()), "max_nodes"


def _backtracks_case():
    base, candidate, mapping = leaves_scenario(dataclasses.replace(BIG, max_backtracks=1), 3)
    return base, candidate, oracle_for(mapping, verifier=accept_only()), "max_backtracks"


def _verifier_case():
    base, candidate, mapping = leaves_scenario(
        dataclasses.replace(BIG, max_verifier_calls=1), 3, terminal=True
    )
    return base, candidate, oracle_for(mapping, verifier=accept_only()), "max_verifier_calls"


def _tokens_case():
    bounds = dataclasses.replace(BIG, max_tokens=3)
    candidate = path_value("A", [1])
    base = projection("base", (candidate,), bounds=bounds)
    root = base.with_decision(HID, candidate)
    leaf = projection("leaf", (path_value("z", [0]),), bounds=bounds)
    mapping = {
        root.fingerprint: Expansion(
            "complete", (ExpansionEdge(path_value("big", [1, 2, 3, 4, 5]), leaf),)
        )
    }
    return base, candidate, oracle_for(mapping, verifier=accept_only()), "max_tokens"


def _depth_case():
    bounds = dataclasses.replace(BIG, max_depth=1)
    candidate = path_value("A", [1])
    base = projection("base", (candidate,), bounds=bounds)
    root = base.with_decision(HID, candidate)
    first = projection("n1", (path_value("z", [0]),), bounds=bounds)
    second = projection("n2", (path_value("z", [0]),), bounds=bounds)
    mapping = {
        root.fingerprint: Expansion("complete", (ExpansionEdge(path_value("d", [1]), first),)),
        first.fingerprint: Expansion("complete", (ExpansionEdge(path_value("d", [1]), second),)),
    }
    return base, candidate, oracle_for(mapping, verifier=accept_only()), "max_depth"


@pytest.mark.parametrize(
    "builder",
    [_nodes_case, _tokens_case, _depth_case, _backtracks_case, _verifier_case],
)
def test_each_finite_budget_yields_unknown_never_unsupported(builder) -> None:
    base, candidate, oracle, stop_reason = builder()
    result = oracle.check(base, query_for(base, candidate))
    assert result.verdict is SupportVerdict.UNKNOWN
    assert result.certificate.stop_reason == stop_reason
    assert result.certificate.exhausted is False


def test_step_budget_yields_unknown() -> None:
    base, candidate, mapping = leaves_scenario(BIG, 5)
    result = oracle_for(mapping, verifier=accept_only(), max_steps=2).check(
        base, query_for(base, candidate)
    )
    assert result.verdict is SupportVerdict.UNKNOWN
    assert result.certificate.stop_reason == "step_budget"


def test_supported_witness_found_before_later_partial_branch() -> None:
    candidate = path_value("A", [1])
    base = projection("base", (candidate,))
    root = base.with_decision(HID, candidate)
    child_a = projection("childA", (path_value("z", [0]),))
    child_b = projection("childB", (path_value("z", [0]),))
    mapping = {
        root.fingerprint: Expansion(
            "complete",
            (
                ExpansionEdge(path_value("a", [1]), child_a),  # token-id ascending: first
                ExpansionEdge(path_value("b", [2]), child_b),  # partial branch, unreached
            ),
        ),
        child_a.fingerprint: Expansion("complete", (), terminal_value("root = Ok")),
        child_b.fingerprint: Expansion("partial", (), terminal_value("root = Ok")),
    }
    result = oracle_for(mapping, verifier=accept_only("root = Ok")).check(
        base, query_for(base, candidate)
    )
    assert result.verdict is SupportVerdict.SUPPORTED
    assert child_b.fingerprint not in result.certificate.explored_state_fingerprints
    assert "partial" not in result.certificate.coverage_observations


# --------------------------------------------------------------------------
# Search bookkeeping
# --------------------------------------------------------------------------


def test_duplicate_states_are_suppressed_by_fingerprint() -> None:
    candidate = path_value("A", [1])
    base = projection("base", (candidate,))
    root = base.with_decision(HID, candidate)
    shared = projection("shared", (path_value("z", [0]),))
    mapping = {
        root.fingerprint: Expansion(
            "complete",
            (
                ExpansionEdge(path_value("e1", [1]), shared),
                ExpansionEdge(path_value("e2", [2]), shared),
            ),
        ),
        shared.fingerprint: Expansion("complete"),
    }
    result = oracle_for(mapping, verifier=accept_only()).check(
        base, query_for(base, candidate)
    )
    fingerprints = result.certificate.explored_state_fingerprints
    assert fingerprints.count(shared.fingerprint) == 1
    assert result.counters.nodes_expanded == 2  # root + shared once
    assert result.verdict is SupportVerdict.UNSUPPORTED


def test_deterministic_explored_and_certificate_ordering() -> None:
    candidate = path_value("A", [1])
    base = projection("base", (candidate,))
    root = base.with_decision(HID, candidate)
    child_a = projection("childA", (path_value("z", [0]),))
    child_b = projection("childB", (path_value("z", [0]),))
    forward = {
        root.fingerprint: Expansion(
            "complete",
            (
                ExpansionEdge(path_value("a", [1]), child_a),
                ExpansionEdge(path_value("b", [2]), child_b),
            ),
        ),
        child_a.fingerprint: Expansion("complete", (), terminal_value("bad-a")),
        child_b.fingerprint: Expansion("complete", (), terminal_value("bad-b")),
    }
    reverse = dict(forward)
    reverse[root.fingerprint] = Expansion(
        "complete",
        (
            ExpansionEdge(path_value("b", [2]), child_b),
            ExpansionEdge(path_value("a", [1]), child_a),
        ),
    )
    first = oracle_for(forward, verifier=accept_only()).check(base, query_for(base, candidate))
    second = oracle_for(reverse, verifier=accept_only()).check(base, query_for(base, candidate))
    assert (
        first.certificate.explored_state_fingerprints
        == second.certificate.explored_state_fingerprints
    )
    assert first.certificate.to_dict() == second.certificate.to_dict()
    order = first.certificate.explored_state_fingerprints
    # token-id ascending: child_a (token 1) is explored before child_b (token 2).
    assert order.index(child_a.fingerprint) < order.index(child_b.fingerprint)


def test_verifier_rejection_is_recorded_not_silently_ignored() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Bad")
    result = oracle_for(mapping, verifier=accept_only("root = Ok")).check(
        base, query_for(base, candidate)
    )
    assert dict(result.certificate.failure_counts)["verifier_rejected"] == 1


def test_missing_capability_yields_unknown_never_unsupported() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Bad")
    result = oracle_for(
        mapping, verifier=accept_only("root = Ok", capability=False)
    ).check(base, query_for(base, candidate))
    cert = result.certificate
    assert result.verdict is SupportVerdict.UNKNOWN
    assert dict(cert.failure_counts).get("missing_capability") == 1
    assert cert.stop_reason == "incomplete_coverage"


def test_undecodable_terminal_yields_unknown_never_unsupported() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Ok")
    result = EnumerativeSupportOracle(
        expander=dict_expander(mapping),
        decoder=lambda terminal: None,
        verifier=accept_only("root = Ok"),
    ).check(base, query_for(base, candidate))
    cert = result.certificate
    assert result.verdict is SupportVerdict.UNKNOWN
    assert dict(cert.failure_counts).get("undecodable_terminal") == 1


def test_candidate_absent_from_domain_is_unknown_not_unsupported() -> None:
    candidate = path_value("A", [1])
    base = projection("base", (path_value("B", [2]),))  # candidate not in the domain
    result = oracle_for({}, verifier=accept_only()).check(
        base, SupportQuery(base.fingerprint, HID, candidate)
    )
    assert result.verdict is SupportVerdict.UNKNOWN
    assert result.certificate.stop_reason == "candidate_absent_from_domain"


# --------------------------------------------------------------------------
# Serialization + replay
# --------------------------------------------------------------------------


def test_certificate_query_and_counters_json_round_trip() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Ok")
    result = oracle_for(mapping, verifier=accept_only("root = Ok")).check(
        base, query_for(base, candidate)
    )
    cert = result.certificate
    restored = SupportCertificate.from_dict(json.loads(json.dumps(cert.to_dict())))
    assert restored == cert
    assert restored.search_order == REFERENCE_SEARCH_ORDER
    assert restored.schema_version == SUPPORT_CERTIFICATE_SCHEMA_VERSION
    assert SearchCounters.from_dict(result.counters.to_dict()) == result.counters
    assert SupportQuery.from_dict(cert.query.to_dict()) == cert.query


def test_replay_accepts_honest_supported_and_unsupported() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Ok")
    verifier = accept_only("root = Ok")
    supported = oracle_for(mapping, verifier=verifier).check(base, query_for(base, candidate))
    ok = replay_support_certificate(
        supported.certificate,
        base,
        expander=dict_expander(mapping),
        decoder=source_decoder,
        verifier=verifier,
    )
    assert ok.ok and ok.violations == ()

    base2, candidate2, mapping2 = single_leaf_scenario(source="root = Bad")
    unsupported = oracle_for(mapping2, verifier=accept_only("root = Ok")).check(
        base2, query_for(base2, candidate2)
    )
    ok2 = replay_support_certificate(
        unsupported.certificate,
        base2,
        expander=dict_expander(mapping2),
        decoder=source_decoder,
        verifier=accept_only("root = Ok"),
    )
    assert ok2.ok and ok2.violations == ()


def test_replay_rejects_stale_constraint_version() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Bad")
    verifier = accept_only("root = Ok")
    result = oracle_for(mapping, verifier=verifier).check(base, query_for(base, candidate))
    stale = dataclasses.replace(base, constraint_version="v2")
    replay = replay_support_certificate(
        result.certificate,
        stale,
        expander=dict_expander(mapping),
        decoder=source_decoder,
        verifier=verifier,
    )
    assert not replay.ok
    assert "stale_constraint_version" in replay.violations


def test_replay_rejects_tampered_witness_digest() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Ok")
    verifier = accept_only("root = Ok")
    result = oracle_for(mapping, verifier=verifier).check(base, query_for(base, candidate))
    tampered = dataclasses.replace(result.certificate, witness_digest="0" * 64)
    replay = replay_support_certificate(
        tampered,
        base,
        expander=dict_expander(mapping),
        decoder=source_decoder,
        verifier=verifier,
    )
    assert not replay.ok
    assert "witness_digest_mismatch" in replay.violations


def test_replay_rejects_unsupported_when_not_exhausted() -> None:
    base, candidate, mapping = single_leaf_scenario(source="root = Bad")
    verifier = accept_only("root = Ok")
    result = oracle_for(mapping, verifier=verifier).check(base, query_for(base, candidate))
    assert result.verdict is SupportVerdict.UNSUPPORTED
    forged = dataclasses.replace(result.certificate, exhausted=False)
    replay = replay_support_certificate(
        forged,
        base,
        expander=dict_expander(mapping),
        decoder=source_decoder,
        verifier=verifier,
    )
    assert not replay.ok
    assert "unsupported_not_exhausted" in replay.violations


def test_replay_rejects_unsupported_forged_from_a_budget_stop() -> None:
    base, candidate, mapping = leaves_scenario(dataclasses.replace(BIG, max_nodes=2), 3)
    verifier = accept_only()
    result = oracle_for(mapping, verifier=verifier).check(base, query_for(base, candidate))
    assert result.verdict is SupportVerdict.UNKNOWN
    forged = dataclasses.replace(
        result.certificate,
        verdict=SupportVerdict.UNSUPPORTED,
        exhausted=True,
        coverage_observations=("complete",),
    )
    replay = replay_support_certificate(
        forged,
        base,
        expander=dict_expander(mapping),
        decoder=source_decoder,
        verifier=verifier,
    )
    assert not replay.ok
    assert "unsupported_budget_stop" in replay.violations
    assert "unsupported_not_reproduced" in replay.violations


def test_replay_accepts_unknown_as_honest() -> None:
    base, candidate, mapping = single_leaf_scenario(coverage="partial", source="root = Bad")
    verifier = accept_only("root = Ok")
    result = oracle_for(mapping, verifier=verifier).check(base, query_for(base, candidate))
    assert result.verdict is SupportVerdict.UNKNOWN
    replay = replay_support_certificate(
        result.certificate,
        base,
        expander=dict_expander(mapping),
        decoder=source_decoder,
        verifier=verifier,
    )
    assert replay.ok and replay.violations == ()


# --------------------------------------------------------------------------
# Backend registry
# --------------------------------------------------------------------------


def test_support_backend_registry_ships_only_enumerative() -> None:
    assert support_backends() == ["enumerative"]
    assert get_support_backend("enumerative") is EnumerativeSupportOracle
    with pytest.raises(KeyError, match="unknown support backend"):
        get_support_backend("smt")


def test_register_support_backend_seam(monkeypatch) -> None:
    from slm_training.dsl.solver import support as support_module

    monkeypatch.setattr(support_module, "_BACKENDS", dict(support_module._BACKENDS))

    def sentinel(**kwargs):
        return None

    register_support_backend("custom-test", sentinel)
    assert get_support_backend("custom-test") is sentinel
    assert "custom-test" in support_backends()


# --------------------------------------------------------------------------
# Torch-free guarantee
# --------------------------------------------------------------------------


def test_support_module_imports_without_torch() -> None:
    root = Path(__file__).parents[2]
    source = (root / "src/slm_training/dsl/solver/support.py").read_text()
    assert "import torch" not in source
    code = """
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError(f'unexpected torch import: {fullname}')
        return None
sys.meta_path.insert(0, BlockTorch())
import slm_training.dsl.solver.support
assert 'torch' not in sys.modules
"""
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=env)
