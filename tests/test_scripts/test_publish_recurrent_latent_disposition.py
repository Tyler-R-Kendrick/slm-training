from __future__ import annotations

from scripts.publish_recurrent_latent_disposition import main


def test_committed_recurrent_disposition_is_current() -> None:
    assert main(["--check"]) == 0
