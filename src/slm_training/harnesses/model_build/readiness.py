"""Model-owned preparation probes; never train, decode, or acquire HF weights."""

from dataclasses import asdict
from pathlib import Path


class ReadinessCapabilityUnavailable(ValueError):
    """A declared local preparation capability is unavailable, not bad data."""


def configured_ranker(config: dict, root: Path) -> Path | None:
    """Resolve and load exactly the ranker used by TwoTower's runtime."""
    from slm_training.dsl.grammar.fastpath.speculative_rank import COMMITTED_NGRAM_TABLE, load_ranker

    mode = str(config.get("speculative_rank") or "off").lower()
    if mode == "off":
        return None
    if mode != "ngram":
        raise ValueError("unsupported speculative ranker in readiness configuration")
    path = Path(config.get("speculative_rank_table") or COMMITTED_NGRAM_TABLE)
    path = path if path.is_absolute() else root / path
    # Reject aliases before reading model-controlled files.
    from slm_training.data.readiness_contract import scoped_path
    path = scoped_path(root, path.relative_to(root).as_posix())
    load_ranker(path, margin=config.get("speculative_rank_margin", 0.0))
    return path


def prepare_records(path: Path, config: dict) -> dict:
    """Actual trainer contract preparation without a neural forward."""
    import torch
    from slm_training.data.readiness_contract import evidence_digest
    from slm_training.harnesses.model_build.data import _load_symbol_only_records
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    cfg = TwoTowerConfig(**config)
    if cfg.context_backend != "scratch" or cfg.denoiser_backend != "scratch":
        raise ReadinessCapabilityUnavailable("capability_unavailable:local readiness requires scratch backends")
    records = _load_symbol_only_records(path)
    with torch.random.fork_rng(devices=[]):
        model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    return {"owner": "TwoTowerModel.from_records", "device": "cpu",
            "records_prepared": len(records), "forwards": 0,
            "resolved_config_digest": evidence_digest(asdict(model.config)),
            "artifact_identity": model.artifact_identity()}
