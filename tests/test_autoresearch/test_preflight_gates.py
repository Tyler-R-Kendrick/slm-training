"""WP-3 preflight gate seam: loader contract + prior-attempt skeptic.

Unit tests only — no training or eval runs. Covers:

- discovery via ``pkgutil.iter_modules`` + module-level ``CHECK`` objects;
- fail-soft law (import crash / run crash / malformed verdict -> ``warn``);
- deterministic ordering and ``has_block``;
- ``prior_attempts`` verdict matrix over the guarded evidence-store client and
  the committed-ledger fallback;
- fingerprint parity with the driver's ``_knobs_fingerprint``;
- the driver-side ``_preflight_screening_slug`` gate (skip on block, fail-soft
  override when every open arm blocks).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from slm_training.autoresearch import preflight
from slm_training.autoresearch.preflight import (
    PreflightVerdict,
    has_block,
    run_preflight,
)
from slm_training.autoresearch.preflight import prior_attempts


# ---------------------------------------------------------------------------
# Loader contract
# ---------------------------------------------------------------------------


@pytest.fixture()
def plugin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Extend the package search path with a throwaway plugin directory."""
    extra = tmp_path / "plugins"
    extra.mkdir()
    monkeypatch.setattr(
        preflight, "__path__", [*list(preflight.__path__), str(extra)]
    )
    yield extra
    for name in list(sys.modules):
        if name.startswith("slm_training.autoresearch.preflight.zz_"):
            del sys.modules[name]


def _write_plugin(directory: Path, name: str, body: str) -> None:
    (directory / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_run_preflight_discovers_prior_attempts_check() -> None:
    verdicts = run_preflight({"slug": "nonexistent-arm-zzz"})
    ids = [item.check_id for item in verdicts]
    assert "prior_attempts" in ids
    assert ids == sorted(ids)
    assert all(isinstance(item, PreflightVerdict) for item in verdicts)


def test_discovered_plugin_runs_and_orders_by_check_id(plugin_dir: Path) -> None:
    _write_plugin(
        plugin_dir,
        "zz_ok",
        """
        from slm_training.autoresearch.preflight import PreflightVerdict

        class _Check:
            check_id = "zz_ok"

            def run(self, candidate: dict) -> PreflightVerdict:
                return PreflightVerdict(
                    check_id=self.check_id,
                    verdict="pass",
                    reasons=["saw " + str(candidate.get("slug"))],
                )

        CHECK = _Check()
        """,
    )
    verdicts = run_preflight({"slug": "abc"})
    ids = [item.check_id for item in verdicts]
    assert "zz_ok" in ids
    assert ids == sorted(ids)
    picked = next(item for item in verdicts if item.check_id == "zz_ok")
    assert picked.verdict == "pass"
    assert picked.reasons == ["saw abc"]


def test_import_crash_becomes_warn_not_exception(plugin_dir: Path) -> None:
    _write_plugin(plugin_dir, "zz_broken_import", "raise RuntimeError('boom-import')")
    verdicts = run_preflight({})
    picked = next(v for v in verdicts if v.check_id == "zz_broken_import")
    assert picked.verdict == "warn"
    assert "boom-import" in " ".join(picked.reasons)


def test_run_crash_becomes_warn_with_traceback_summary(plugin_dir: Path) -> None:
    _write_plugin(
        plugin_dir,
        "zz_broken_run",
        """
        class _Check:
            check_id = "zz_broken_run"

            def run(self, candidate: dict):
                raise ValueError("boom-run")

        CHECK = _Check()
        """,
    )
    verdicts = run_preflight({})
    picked = next(v for v in verdicts if v.check_id == "zz_broken_run")
    assert picked.verdict == "warn"
    joined = " ".join(picked.reasons)
    assert "ValueError" in joined and "boom-run" in joined


def test_dict_verdict_is_validated_and_module_without_check_skipped(
    plugin_dir: Path,
) -> None:
    _write_plugin(
        plugin_dir,
        "zz_dict_verdict",
        """
        class _Check:
            check_id = "zz_dict_verdict"

            def run(self, candidate: dict):
                return {
                    "check_id": "zz_dict_verdict",
                    "verdict": "block",
                    "reasons": ["typed via dict"],
                }

        CHECK = _Check()
        """,
    )
    _write_plugin(plugin_dir, "zz_no_check", "VALUE = 1\n")
    verdicts = run_preflight({})
    ids = [item.check_id for item in verdicts]
    assert "zz_no_check" not in ids
    picked = next(v for v in verdicts if v.check_id == "zz_dict_verdict")
    assert picked.verdict == "block"
    assert has_block(verdicts)


def test_has_block_false_for_pass_and_warn_only() -> None:
    verdicts = [
        PreflightVerdict(check_id="a", verdict="pass", reasons=[]),
        PreflightVerdict(check_id="b", verdict="warn", reasons=["x"]),
    ]
    assert not has_block(verdicts)


# ---------------------------------------------------------------------------
# prior_attempts check
# ---------------------------------------------------------------------------


def _install_fake_store(
    monkeypatch: pytest.MonkeyPatch, records: list[dict]
) -> None:
    pkg = ModuleType("slm_training.evidence_store")
    client = ModuleType("slm_training.evidence_store.client")

    def find_prior_attempts(
        hypothesis_text=None, lever_keys=None, config_fingerprint=None
    ):
        return records

    client.find_prior_attempts = find_prior_attempts  # type: ignore[attr-defined]
    pkg.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slm_training.evidence_store", pkg)
    monkeypatch.setitem(sys.modules, "slm_training.evidence_store.client", client)


_FP = "a" * 16
_ENDPOINT = "smoke.structural_similarity"


def _candidate(**overrides) -> dict:
    base = {
        "hypothesis_text": "arm improves structure",
        "lever_keys": ["compiler_decode_mode"],
        "config_fingerprint": _FP,
        "n_seeds": 1,
        "steps": 40,
        "minimum_effect": 0.01,
        "endpoint_metric": _ENDPOINT,
        "slug": "some-arm",
    }
    base.update(overrides)
    return base


def test_blocks_powered_refutation_by_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_store(
        monkeypatch,
        [
            {
                "config_fingerprint": _FP,
                "endpoint_metric": _ENDPOINT,
                "outcome": "confirm_failed",
                "n_seeds": 9,
                "p_value": None,
            }
        ],
    )
    verdict = prior_attempts.CHECK.run(_candidate())
    assert verdict.verdict == "block"
    assert verdict.data["powered_refutations"] == 1
    assert verdict.data["source"] == "evidence_store"


def test_blocks_powered_refutation_by_p_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_store(
        monkeypatch,
        [
            {
                "config_fingerprint": _FP,
                "endpoint_metric": _ENDPOINT,
                "outcome": "ship_rejected",
                "n_seeds": 2,
                "p_value": 0.41,
            }
        ],
    )
    verdict = prior_attempts.CHECK.run(_candidate())
    assert verdict.verdict == "block"


def test_warns_on_underpowered_prior_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_store(
        monkeypatch,
        [
            {
                "config_fingerprint": _FP,
                "endpoint_metric": _ENDPOINT,
                "outcome": "confirm_failed",
                "n_seeds": 2,
                "p_value": None,
            }
        ],
    )
    verdict = prior_attempts.CHECK.run(_candidate())
    assert verdict.verdict == "warn"
    assert verdict.data["matched"] == 1


def test_endpoint_mismatch_never_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_store(
        monkeypatch,
        [
            {
                "config_fingerprint": _FP,
                "endpoint_metric": "held_out.structural_similarity",
                "outcome": "confirm_failed",
                "n_seeds": 12,
                "p_value": 0.2,
            }
        ],
    )
    verdict = prior_attempts.CHECK.run(_candidate())
    assert verdict.verdict == "warn"  # attempted elsewhere, never a block


def test_passes_when_no_matching_records(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_store(
        monkeypatch,
        [
            {
                "config_fingerprint": "b" * 16,
                "endpoint_metric": _ENDPOINT,
                "outcome": "confirm_failed",
                "n_seeds": 9,
                "p_value": None,
            }
        ],
    )
    verdict = prior_attempts.CHECK.run(_candidate())
    assert verdict.verdict == "pass"
    assert verdict.data["matched"] == 0


def test_passes_without_any_config_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_store(monkeypatch, [])
    verdict = prior_attempts.CHECK.run(
        {"hypothesis_text": "x", "endpoint_metric": _ENDPOINT}
    )
    assert verdict.verdict == "pass"
    assert "no config identity" in verdict.reasons[0]


def test_ledger_fallback_blocks_on_slug_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the guarded evidence-store import to fail regardless of whether a
    # concurrent WP has landed the real package.
    monkeypatch.setitem(sys.modules, "slm_training.evidence_store", None)
    ledger = tmp_path / "evidence_ledger.v1.json"
    ledger.write_text(
        json.dumps(
            {
                "schema": "autotrain_evidence_ledger/v1",
                "arms": {
                    "dead-arm": {
                        "n_obs": 12,
                        "n_complete": 9,
                        "n_positive": 0,
                        "n_null": 9,
                    },
                    "live-arm": {
                        "n_obs": 3,
                        "n_complete": 2,
                        "n_positive": 1,
                        "n_null": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prior_attempts, "_LEDGER_PATH", ledger)
    blocked = prior_attempts.CHECK.run(_candidate(slug="dead-arm"))
    assert blocked.verdict == "block"
    assert blocked.data["source"] == "evidence_ledger_fallback"
    alive = prior_attempts.CHECK.run(_candidate(slug="live-arm"))
    assert alive.verdict in {"warn", "pass"}
    assert alive.verdict != "block"


def test_ledger_fallback_missing_file_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "slm_training.evidence_store", None)
    monkeypatch.setattr(prior_attempts, "_LEDGER_PATH", tmp_path / "missing.json")
    verdict = prior_attempts.CHECK.run(_candidate())
    assert verdict.verdict == "pass"


def test_arm_fingerprint_converges_evidence_store_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fingerprint-convergence fix: a real evidence-store climb-ledger-arm
    record (fingerprinted over a slug-keyed descriptor — a domain that never
    equals a live candidate's lever-derived fingerprint) must still be
    discoverable and matched by ``prior_attempts``, without falling back to
    the raw-ledger path. Exercises the real ``evidence_store`` package, not
    the ``_install_fake_store`` stand-in.
    """
    from slm_training.evidence_store import client as real_client
    from slm_training.evidence_store.local_index import write_local_index
    from slm_training.evidence_store.records import (
        EvidenceRecordV1,
        compute_arm_fingerprint,
    )

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    slug = "dead-arm-convergence"
    record = EvidenceRecordV1(
        experiment_id=f"autotrain-climb-arm-{slug}",
        campaign_id="autotrain-climb",
        hypothesis_text=f"Screening arm '{slug}' improves {_ENDPOINT} over control",
        lever_keys=[slug],
        config_fingerprint=compute_arm_fingerprint(slug),
        endpoint_metric=_ENDPOINT,
        n_seeds=9,
        outcome="confirm_failed",
    )
    local_index = tmp_path / "local_index.jsonl"
    write_local_index([record], local_index)
    monkeypatch.setattr(real_client, "DEFAULT_LOCAL_INDEX_PATH", local_index)

    # A live candidate's own fingerprint is lever-derived — deliberately a
    # different value from the arm fingerprint, mirroring the real driver.
    candidate = _candidate(slug=slug, config_fingerprint="c" * 16)
    verdict = prior_attempts.CHECK.run(candidate)
    assert verdict.verdict == "block"
    assert verdict.data["source"] == "evidence_store"
    assert verdict.data["powered_refutations"] == 1


def test_check_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(candidate):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(prior_attempts, "_find_prior_attempts", _explode)
    verdict = prior_attempts.CHECK.run(_candidate())
    assert verdict.verdict == "warn"
    assert "store exploded" in " ".join(verdict.reasons)


def test_fingerprint_fallback_matches_driver_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without the evidence-store package, the local fallback must mirror the
    # driver's _knobs_fingerprint exactly.
    monkeypatch.setitem(sys.modules, "slm_training.evidence_store", None)
    levers = {"b_key": 2, "a_key": "x", "steps": 40}
    expected = hashlib.sha256(
        json.dumps(
            {"a_key": "x", "b_key": 2},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    assert prior_attempts.config_fingerprint(levers) == expected
    # steps is cycle jitter, not identity.
    assert prior_attempts.config_fingerprint(
        {**levers, "steps": 999}
    ) == expected


def test_fingerprint_prefers_evidence_store_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg = ModuleType("slm_training.evidence_store")
    records = ModuleType("slm_training.evidence_store.records")

    def compute_config_fingerprint(config):
        return "store-fp-" + "/".join(sorted(config))

    records.compute_config_fingerprint = (  # type: ignore[attr-defined]
        compute_config_fingerprint
    )
    pkg.records = records  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slm_training.evidence_store", pkg)
    monkeypatch.setitem(
        sys.modules, "slm_training.evidence_store.records", records
    )
    assert (
        prior_attempts.config_fingerprint({"b": 1, "a": 2}) == "store-fp-a/b"
    )


def test_fingerprint_excludes_steps_before_evidence_store_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the evidence-store helper must see the steps-excluded
    stable mapping, not the raw config — otherwise the fingerprint changes
    every cycle and prior-attempt matching silently never fires."""
    pkg = ModuleType("slm_training.evidence_store")
    records = ModuleType("slm_training.evidence_store.records")
    records.compute_config_fingerprint = (  # type: ignore[attr-defined]
        lambda config: "store-fp-" + "/".join(sorted(config))
    )
    pkg.records = records  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slm_training.evidence_store", pkg)
    monkeypatch.setitem(sys.modules, "slm_training.evidence_store.records", records)
    assert prior_attempts.config_fingerprint(
        {"a": 1, "steps": 10}
    ) == prior_attempts.config_fingerprint({"a": 1, "steps": 20})


# ---------------------------------------------------------------------------
# Driver-side gate (_preflight_screening_slug)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_autotrain_continuous.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_autotrain_continuous_preflight_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def only_prior_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the WP-3 gate from sibling plugins' verdict policies.

    ``power_check`` / ``concluded_family`` are concurrent work with their own
    block semantics; the driver-gate tests assert this seam's mechanics only.
    """
    monkeypatch.setattr(
        preflight, "_discover_module_names", lambda: ["prior_attempts"]
    )


def _gate(driver, slug: str, *, reselect):
    return driver._preflight_screening_slug(
        slug,
        steps=40,
        endpoint_metric=_ENDPOINT,
        minimum_effect=0.01,
        skip=set(),
        reselect=reselect,
    )


def test_driver_gate_passes_through_unblocked_slug(
    driver, only_prior_attempts, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_store(monkeypatch, [])
    bank_slug = driver._SCREENING_ARM_BANK[0][0]
    chosen, payload = _gate(
        driver, bank_slug, reselect=lambda skip: pytest.fail("must not reselect")
    )
    assert chosen == bank_slug
    assert payload is not None
    assert payload["selected_slug"] == bank_slug
    assert payload["blocked_slugs"] == []
    assert "prior_attempts" in {
        v["check_id"] for v in payload["verdicts"][bank_slug]
    }


def test_driver_gate_passes_cumulative_arm_seeds_not_literal_one(
    driver, only_prior_attempts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeds-policy reconciliation: the candidate's ``n_seeds`` reflects the
    arm's cumulative ledger ``n_complete`` (projected +1 for this cycle), not
    a literal ``1`` — a literal 1 is undecidable by construction and would
    make ``power_decidability`` block every screening cycle forever.
    """
    _install_fake_store(monkeypatch, [])
    bank_slug = driver._SCREENING_ARM_BANK[0][0]

    from slm_training.autoresearch import evidence_ledger as ev

    monkeypatch.setattr(
        ev,
        "load_ledger",
        lambda *a, **k: {"arms": {bank_slug: {"n_complete": 6}}},
    )
    captured: list[dict] = []
    real_run_preflight = preflight.run_preflight

    def _spy(candidate):
        captured.append(dict(candidate))
        return real_run_preflight(candidate)

    monkeypatch.setattr(preflight, "run_preflight", _spy)
    _gate(driver, bank_slug, reselect=lambda skip: pytest.fail("must not reselect"))
    assert captured
    assert captured[0]["n_seeds"] == 7  # cumulative 6 + this cycle


def test_driver_gate_unseen_arm_uses_first_cycle_seeds(
    driver, only_prior_attempts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An arm absent from the ledger (never screened) starts at n_seeds=1."""
    _install_fake_store(monkeypatch, [])
    bank_slug = driver._SCREENING_ARM_BANK[0][0]

    from slm_training.autoresearch import evidence_ledger as ev

    monkeypatch.setattr(ev, "load_ledger", lambda *a, **k: {"arms": {}})
    captured: list[dict] = []
    real_run_preflight = preflight.run_preflight

    def _spy(candidate):
        captured.append(dict(candidate))
        return real_run_preflight(candidate)

    monkeypatch.setattr(preflight, "run_preflight", _spy)
    _gate(driver, bank_slug, reselect=lambda skip: pytest.fail("must not reselect"))
    assert captured
    assert captured[0]["n_seeds"] == 1


def test_driver_gate_ledger_load_failure_degrades_to_one(
    driver, only_prior_attempts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ledger-load error must never break the gate — degrade to n_seeds=1."""
    _install_fake_store(monkeypatch, [])
    bank_slug = driver._SCREENING_ARM_BANK[0][0]

    from slm_training.autoresearch import evidence_ledger as ev

    def _explode(*a, **k):
        raise RuntimeError("ledger unreadable")

    monkeypatch.setattr(ev, "load_ledger", _explode)
    captured: list[dict] = []
    real_run_preflight = preflight.run_preflight

    def _spy(candidate):
        captured.append(dict(candidate))
        return real_run_preflight(candidate)

    monkeypatch.setattr(preflight, "run_preflight", _spy)
    chosen, payload = _gate(
        driver, bank_slug, reselect=lambda skip: pytest.fail("must not reselect")
    )
    assert chosen == bank_slug
    assert captured
    assert captured[0]["n_seeds"] == 1


def test_driver_gate_skips_blocked_slug_and_reselects(
    driver, only_prior_attempts, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_slug = driver._SCREENING_ARM_BANK[0][0]
    second_slug = driver._SCREENING_ARM_BANK[1][0]
    first_extras = dict(driver._SCREENING_ARM_BANK[0][2])
    first_fp = driver._knobs_fingerprint(
        driver._apply_arm_extras(40, first_extras)
    )
    _install_fake_store(
        monkeypatch,
        [
            {
                "config_fingerprint": first_fp,
                "endpoint_metric": _ENDPOINT,
                "outcome": "confirm_failed",
                "n_seeds": 9,
                "p_value": None,
            }
        ],
    )
    chosen, payload = _gate(
        driver, first_slug, reselect=lambda skip: second_slug
    )
    assert chosen == second_slug
    assert payload["blocked_slugs"] == [first_slug]
    assert "override" not in payload


def test_driver_gate_fails_soft_when_all_open_arms_block(
    driver, only_prior_attempts, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_slug = driver._SCREENING_ARM_BANK[0][0]

    class _BlockEverything:
        check_id = "prior_attempts"

        def run(self, candidate: dict) -> PreflightVerdict:
            return PreflightVerdict(
                check_id=self.check_id,
                verdict="block",
                reasons=["always blocked (test)"],
            )

    monkeypatch.setattr(prior_attempts, "CHECK", _BlockEverything())
    chosen, payload = _gate(driver, first_slug, reselect=lambda skip: None)
    assert chosen == first_slug
    assert payload["override"] == "all_open_arms_blocked_ran_original_pick"
    assert first_slug in payload["blocked_slugs"]


def test_driver_gate_fails_soft_when_reselect_raises(
    driver, only_prior_attempts, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_slug = driver._SCREENING_ARM_BANK[0][0]

    class _BlockEverything:
        check_id = "prior_attempts"

        def run(self, candidate: dict) -> PreflightVerdict:
            return PreflightVerdict(
                check_id=self.check_id, verdict="block", reasons=["blocked"]
            )

    def _explode(skip):
        raise RuntimeError("reselect exploded")

    monkeypatch.setattr(prior_attempts, "CHECK", _BlockEverything())
    chosen, payload = _gate(driver, first_slug, reselect=_explode)
    assert chosen == first_slug
    assert payload is None  # fail-soft path: run the original pick unfiltered
