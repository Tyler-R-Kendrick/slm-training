"""OpenFeature representation of typed experiments: flagd export + provider."""

from __future__ import annotations

import json

import pytest
from openfeature import api
from openfeature.evaluation_context import EvaluationContext
from openfeature.flag_evaluation import ErrorCode, Reason

from slm_training.autoresearch.openfeature import (
    EXPERIMENT_CONTEXT_ATTRIBUTE,
    FLAGD_SCHEMA,
    ExperimentFlagProvider,
    export_flagd_flags,
)
from slm_training.autoresearch.schemas import ExperimentKnobs, ExperimentSpec

from tests.test_autoresearch.test_harness import experiment, hypothesis_matrix


def _experiments() -> tuple[ExperimentSpec, ...]:
    return (
        experiment(
            experiment_id="exp-a",
            knobs=ExperimentKnobs(
                steps=300,
                lr=3e-4,
                schema_in_context=True,
                compiler_decode_mode="restricted",
                mixture_weights={"grammar": 2.0, "layout": 1.0},
            ),
        ),
        experiment(
            experiment_id="exp-b",
            hypothesis="A distinct hypothesis about fewer supervised steps.",
            knobs=ExperimentKnobs(steps=100),
        ),
    )


def test_flagd_export_shape() -> None:
    document = export_flagd_flags(_experiments(), flag_set_id="test-campaign")
    assert document["$schema"] == FLAGD_SCHEMA
    assert document["metadata"] == {"flagSetId": "test-campaign"}
    flags = document["flags"]
    assert set(flags) == {
        "steps",
        "lr",
        "schema_in_context",
        "compiler_decode_mode",
        "mixture_weights",
    }
    steps = flags["steps"]
    assert steps["state"] == "ENABLED"
    assert steps["variants"] == {"exp-a": 300, "exp-b": 100}
    assert steps["defaultVariant"] is None
    assert steps["targeting"] == {
        "if": [
            {"in": [{"var": EXPERIMENT_CONTEXT_ATTRIBUTE}, ["exp-a", "exp-b"]]},
            {"var": EXPERIMENT_CONTEXT_ATTRIBUTE},
            None,
        ]
    }
    # Knobs set by only one experiment carry only that variant; the other
    # experiment falls through targeting to the code default.
    assert flags["schema_in_context"]["variants"] == {"exp-a": True}
    assert json.dumps(document)  # JSON-serializable end to end


def test_flagd_export_rejects_duplicates_and_empty() -> None:
    spec = _experiments()[0]
    with pytest.raises(ValueError, match="duplicate experiment_id"):
        export_flagd_flags([spec, spec], flag_set_id="test-campaign")
    with pytest.raises(ValueError, match="at least one experiment"):
        export_flagd_flags([], flag_set_id="test-campaign")


def test_flagd_export_from_hypothesis_matrix() -> None:
    matrix = hypothesis_matrix()
    document = export_flagd_flags(
        (item.experiment for item in matrix.hypotheses),
        flag_set_id=matrix.campaign_id,
    )
    assert set(document["flags"]) == {"steps"}
    assert len(document["flags"]["steps"]["variants"]) == len(matrix.hypotheses)


def test_export_openfeature_cli(tmp_path, capsys) -> None:
    from scripts.autoresearch import main

    matrix = hypothesis_matrix()
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(matrix.model_dump_json(), encoding="utf-8")
    output = tmp_path / "flags.flagd.json"
    assert (
        main(
            [
                "--root",
                str(tmp_path / "campaigns"),
                "export-openfeature",
                "--campaign-id",
                matrix.campaign_id,
                "--matrix",
                str(matrix_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["$schema"] == FLAGD_SCHEMA
    assert document["metadata"]["flagSetId"] == matrix.campaign_id
    printed = json.loads(capsys.readouterr().out)
    assert printed == document
    artifacts = list(
        (tmp_path / "campaigns" / matrix.campaign_id / "artifacts" / "openfeature")
        .glob("*.json")
    )
    assert len(artifacts) == 1


@pytest.fixture()
def client():
    api.set_provider(ExperimentFlagProvider(_experiments()), "openui-experiments")
    yield api.get_client("openui-experiments")
    api.clear_providers()


def _context(experiment_id: str = "exp-a") -> EvaluationContext:
    return EvaluationContext(
        attributes={EXPERIMENT_CONTEXT_ATTRIBUTE: experiment_id}
    )


def test_provider_targeting_match(client) -> None:
    details = client.get_integer_details("steps", 200, _context())
    assert (details.value, details.variant) == (300, "exp-a")
    assert details.reason == Reason.TARGETING_MATCH
    assert client.get_boolean_value("schema_in_context", False, _context()) is True
    assert client.get_float_value("lr", 1e-3, _context()) == pytest.approx(3e-4)
    assert (
        client.get_string_value("compiler_decode_mode", "off", _context())
        == "restricted"
    )
    assert client.get_object_value("mixture_weights", {}, _context()) == {
        "grammar": 2.0,
        "layout": 1.0,
    }


def test_provider_unset_knob_falls_back_to_code_default(client) -> None:
    details = client.get_boolean_details(
        "schema_in_context", False, _context("exp-b")
    )
    assert details.value is False
    assert details.reason == Reason.DEFAULT
    assert details.variant is None


def test_provider_targeting_key_selects_experiment(client) -> None:
    details = client.get_integer_details(
        "steps", 200, EvaluationContext(targeting_key="exp-b")
    )
    assert (details.value, details.variant) == (100, "exp-b")


def test_provider_fails_closed(client) -> None:
    unknown = client.get_boolean_details("not_a_knob", True, _context())
    assert (unknown.value, unknown.error_code) == (True, ErrorCode.FLAG_NOT_FOUND)

    mismatch = client.get_boolean_details("steps", False, _context())
    assert mismatch.error_code == ErrorCode.TYPE_MISMATCH

    missing = client.get_integer_details("steps", 200, EvaluationContext())
    assert missing.error_code == ErrorCode.TARGETING_KEY_MISSING

    stranger = client.get_integer_details("steps", 200, _context("exp-zzz"))
    assert (stranger.value, stranger.error_code) == (
        200,
        ErrorCode.INVALID_CONTEXT,
    )


def test_provider_from_matrix(client) -> None:
    provider = ExperimentFlagProvider.from_matrix(hypothesis_matrix())
    details = provider.resolve_integer_details("steps", 200, _context("hyp-3"))
    assert (details.value, details.variant) == (103, "hyp-3")
