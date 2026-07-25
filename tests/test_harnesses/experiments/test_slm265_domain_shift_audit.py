from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm265_domain_shift_audit import build_coverage_gap_manifest


def _record(record_id: str, prompt: str, openui: str) -> ExampleRecord:
    return ExampleRecord(id=record_id, prompt=prompt, openui=openui, placeholders=[":x"])


def test_manifest_is_deterministic_outcome_blind_and_exhaustive() -> None:
    train = [_record("train", "make a card", 'root = Card(":x")')]
    evals = {"rico_held": [
        _record("near", "make a card", 'root = Card(":x")'),
        _record("far", "make a button", 'root = Button(":x")'),
        _record("mid", "make a stack", 'root = Stack([Card(":x")], "column")'),
    ]}
    first = build_coverage_gap_manifest(train, evals)
    assert first == build_coverage_gap_manifest(train, evals)
    assert {row["stratum"] for row in first["rows"]} == {"near", "mid", "far"}
    assert all("outcome" not in row for row in first["rows"])
    assert first["coverage_gaps"] == [{"suite": "rico_held", "n": 3, "far_n": 1, "leaked_n": 1}]
