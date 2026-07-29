from scripts.run_perf_matrix import (
    _choice_brute_allowed_ids,
    _choice_brute_distance,
    _choice_fixture_states,
    _completion_domain_payload,
    _completion_fixture_gates,
    _guardrails,
    _legacy_choice_allowed_ids,
    _payload_digest,
    _prior_choice_diagnostic,
    _repo_relative_artifact_paths,
    _sample_summary,
    experiments,
)


def test_invalid_p0_cannot_promote_candidate() -> None:
    result = _guardrails(
        {"parse_rate": 0.0, "placeholder_fidelity": 0.0, "latency_ms_mean": 100.0},
        {"parse_rate": 1.0, "placeholder_fidelity": 1.0, "latency_ms_mean": 10.0},
    )
    assert result["pass"] is False
    assert "invalid P0" in result["note"]


def test_valid_p0_applies_quality_floor() -> None:
    result = _guardrails(
        {"parse_rate": 1.0, "placeholder_fidelity": 1.0, "latency_ms_mean": 100.0},
        {"parse_rate": 0.96, "placeholder_fidelity": 0.96, "latency_ms_mean": 50.0},
    )
    assert result["pass"] is True
    assert result["speedup_vs_p0"] == 2.0


def test_c5_c10_are_registered_without_running() -> None:
    rows = {row.eid: row for row in experiments()}
    assert set(rows) >= {"C5", "C6", "C7", "C8", "C9", "C10"}
    assert rows["C6"].grammar_active_symbol_bitsets is True
    assert rows["C8"].grammar_completion_bounds is True
    assert rows["C9"].compiler_prefill_max_states == 1
    assert rows["C10"].compiler_prefill_max_states == 0


def test_completion_timing_summary_is_robust_and_deterministic() -> None:
    summary = _sample_summary([9, 1, 5, 100, 7])
    assert summary == {
        "samples_ns": [9, 1, 5, 100, 7],
        "median_ns": 7.0,
        "min_ns": 1,
        "max_ns": 100,
        "mad_ns": 2.0,
    }


def test_completion_result_paths_are_repo_relative(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = {
        "runDir": str(tmp_path / "outputs" / "run"),
        "remote": "hf://buckets/example",
    }
    assert _repo_relative_artifact_paths(payload) == {
        "runDir": "outputs/run",
        "remote": "hf://buckets/example",
    }


def test_completion_domain_digest_covers_ordered_witnesses() -> None:
    from slm_training.dsl.grammar_capabilities import (
        CompletionDomainCandidateV1,
        CompletionDomainV1,
    )

    first = CompletionDomainV1(
        status="complete",
        candidates=(
            CompletionDomainCandidateV1((3,), "bind", (3, 4, 2)),
            CompletionDomainCandidateV1((5,), "component", (5, 6, 2)),
        ),
        scope_fingerprint="scope",
        terminals=("NAME",),
    )
    second = CompletionDomainV1(
        status="complete",
        candidates=tuple(reversed(first.candidates)),
        scope_fingerprint="scope",
        terminals=("NAME",),
    )
    assert _payload_digest(_completion_domain_payload(first)) != _payload_digest(
        _completion_domain_payload(second)
    )


def test_choice_exact_distance_and_allowed_sets_match_independent_oracle() -> None:
    import time

    deadline = time.monotonic() + 30.0
    fixtures = _choice_fixture_states()
    for _, state, remaining in fixtures[:2]:
        assert state._completion_distance(remaining) == _choice_brute_distance(
            state, remaining, deadline=deadline
        )
        assert state.allowed_ids(remaining) == _choice_brute_allowed_ids(
            state, remaining, deadline=deadline
        )


def test_choice_legacy_greedy_false_prune_is_explicit() -> None:
    _, state, _ = _choice_fixture_states()[0]
    remaining = 4
    root_marker = state.tokenizer.token_to_id["r="]
    assert root_marker in state.allowed_ids(remaining)
    assert root_marker not in _legacy_choice_allowed_ids(state, remaining)


def test_prior_choice_timing_is_non_promotable_incomplete_metadata() -> None:
    prior = _prior_choice_diagnostic()
    assert prior["promotable"] is False
    assert prior["status"] == "incomplete_diagnostic"
    assert prior["v2_median_ns"] == 17_021_824
    assert prior["legacy_median_ns"] == 182_931_458
    assert "raw_samples_not_retained" in prior["reason"]


def test_solver_and_compiler_fixture_gates_require_parity_speed_and_singleton() -> None:
    solver = {"equivalent": True}
    compiler = {
        "equivalent": True,
        "compiler_ms_speedup_v1_over_v2": 5.0,
        "singleton": {"equivalent": True, "zero_forward": True},
    }
    assert all(_completion_fixture_gates(solver, compiler).values())

    compiler["compiler_ms_speedup_v1_over_v2"] = 4.999
    compiler["singleton"]["zero_forward"] = False
    gates = _completion_fixture_gates(solver, compiler)
    assert gates["compiler_ms_speedup_ge_5x"] is False
    assert gates["compiler_singleton_zero_forward"] is False
