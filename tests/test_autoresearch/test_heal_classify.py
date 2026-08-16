"""Blocker classification: environment vs code precision (skeptic O5)."""

from __future__ import annotations

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch.heal.classify import (
    HARD_HARNESS_MARKERS,
    classify_blocker,
    missing_js_module,
)


class TestEnvironmentVsCode:
    def test_agentv_unavailable_is_environment(self) -> None:
        assert (
            classify_blocker(
                "repair_harness",
                "AgentV SDK is unavailable; run npm ci at the repo root",
            )
            == "environment"
        )

    def test_err_module_not_found_is_environment(self) -> None:
        assert (
            classify_blocker(
                "repair_harness",
                "eval crashed: ERR_MODULE_NOT_FOUND Cannot find package "
                "'typebox' imported from @agentv/core",
            )
            == "environment"
        )

    def test_repo_internal_import_is_code(self) -> None:
        # `npm ci` cannot fix a repo-internal python import — routing this to
        # the environment playbook would create an install-verify-fail loop.
        assert (
            classify_blocker(
                "repair_harness", "No module named slm_training.harnesses.foo"
            )
            == "code"
        )

    def test_scripts_import_is_code(self) -> None:
        assert (
            classify_blocker(
                "repair_harness", "import error in scripts.evaluate_model"
            )
            == "code"
        )

    def test_unattributed_module_error_fails_toward_code(self) -> None:
        assert (
            classify_blocker("repair_harness", "module not found") == "code"
        )
        assert (
            classify_blocker("repair_harness", "no module named mystery")
            == "code"
        )

    def test_generic_harness_reason_is_code(self) -> None:
        assert (
            classify_blocker("repair_harness", "harness crashed for reasons")
            == "code"
        )


class TestFormalSplit:
    def test_theorem_backed_miss_is_contradiction(self) -> None:
        assert (
            classify_blocker(
                "repair_formal", "theorem_backed_band_miss: parse_rate_pm"
            )
            == "formal_contradiction"
        )

    def test_checker_unavailable_is_infra(self) -> None:
        assert (
            classify_blocker(
                "repair_formal",
                "in-repo LeverProof checker is unavailable; run make build",
            )
            == "formal_infra"
        )

    def test_ambiguous_formal_stays_contradiction(self) -> None:
        # Absence of a visible contradiction is not evidence of transience
        # (skeptic O9): unclassifiable formal reasons stay evidence.
        assert (
            classify_blocker("repair_formal", "formal status unknown")
            == "formal_contradiction"
        )

    def test_stop_campaign_is_always_contradiction(self) -> None:
        assert (
            classify_blocker("stop_campaign", "anything at all")
            == "formal_contradiction"
        )


class TestOtherKinds:
    @pytest.mark.parametrize(
        ("kind", "expected"),
        case_values(__file__, "test_kind_routing"),
    )
    def test_kind_routing(self, kind: str, expected: str) -> None:
        assert classify_blocker(kind, "reason") == expected


class TestMarkers:
    def test_marker_list_covers_documented_crashes(self) -> None:
        # The emission-time hard markers and this classifier share one list;
        # the documented true-crash signatures must all be present.
        for marker in (
            "agentv sdk is unavailable",
            "npm ci",
            "module not found",
            "import error",
            "no module named",
        ):
            assert marker in HARD_HARNESS_MARKERS

    def test_missing_js_module_parses_node_error(self) -> None:
        assert (
            missing_js_module(
                "Cannot find package 'typebox' imported from x.mjs"
            )
            == "typebox"
        )
        assert missing_js_module("Cannot find module '@agentv/core'") == (
            "@agentv/core"
        )
        assert missing_js_module("no js error here") is None
