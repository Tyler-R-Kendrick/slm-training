from slm_training.harness_core.lineage.records import FileArtifact


def test_file_artifact_round_trips_deterministically():
    artifact = FileArtifact(name="context_tokenizer.json", size_bytes=12, sha256="a" * 64)
    assert FileArtifact.from_dict(artifact.to_dict()) == artifact
