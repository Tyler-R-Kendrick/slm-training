"""Cross-structure semantic dedup (SemDeDup-style) with a lexical fallback.

Unlike :func:`slm_training.data.dedup.apply_fuzzy_dedup` — which only compares
records inside the same structural-fingerprint bucket — this pass compares
every record against every kept leader, so paraphrased prompts that produce
slightly different layouts still collapse to one representative.

Engine selection mirrors the lang-core Node-bridge pattern: when the optional
``sentence-transformers`` extra is installed the payloads are embedded and
compared by cosine; otherwise a deterministic pure-Python hashed char-n-gram
TF-IDF vectorizer is used. ``SLM_SEMANTIC_DEDUP_ENGINE`` pins the engine
(``auto`` | ``embeddings`` | ``lexical``) so published builds stay comparable.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np

from slm_training.data.dedup import _keep_key, _record_family
from slm_training.data.leakage import norm_text, normalize_openui_structure
from slm_training.dsl.schema import ExampleRecord

# Families that are near-duplicates by design; none are currently exempt.
DELIBERATE_VARIANT_FAMILIES: frozenset[str] = frozenset()

# Engine-specific cosine thresholds: embeddings place paraphrases much closer
# together than surface n-grams do, so the lexical fallback runs stricter.
EMBEDDING_THRESHOLD = 0.92
LEXICAL_THRESHOLD = 0.95

_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_LEXICAL_DIM = 4096
_NGRAM_SIZES = (3, 4, 5)
_WORD_RE = re.compile(r"[a-z0-9]+")


def semantic_payload(record: ExampleRecord) -> str:
    """Prompt ⊕ namespace/binder-normalized structure — the dedup surface."""
    return norm_text(record.prompt) + "\n" + normalize_openui_structure(record.openui)


def _engine_preference() -> str:
    value = (os.getenv("SLM_SEMANTIC_DEDUP_ENGINE") or "auto").strip().lower()
    if value not in {"auto", "embeddings", "lexical"}:
        raise ValueError(
            f"invalid SLM_SEMANTIC_DEDUP_ENGINE {value!r}; "
            "expected auto | embeddings | lexical"
        )
    return value


@lru_cache(maxsize=1)
def _load_embedder() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model_name = os.getenv("SLM_EMBEDDING_MODEL") or _DEFAULT_EMBEDDING_MODEL
    return SentenceTransformer(model_name)


def similarity_engine() -> str:
    """Report which engine :func:`record_vectors` will use right now."""
    preference = _engine_preference()
    if preference == "lexical":
        return "lexical-tfidf"
    embedder = _load_embedder()
    if embedder is None:
        if preference == "embeddings":
            raise RuntimeError(
                "SLM_SEMANTIC_DEDUP_ENGINE=embeddings but sentence-transformers "
                "is not installed; pip install '.[embeddings]'"
            )
        return "lexical-tfidf"
    return "embeddings"


def _char_ngrams(text: str) -> list[str]:
    tokens = _WORD_RE.findall(text.lower())
    joined = " ".join(tokens)
    grams: list[str] = []
    for n in _NGRAM_SIZES:
        grams.extend(joined[i : i + n] for i in range(max(0, len(joined) - n + 1)))
    return grams


def _bucket(gram: str) -> int:
    digest = hashlib.sha256(gram.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _LEXICAL_DIM


def _lexical_vectors(payloads: list[str]) -> np.ndarray:
    """Hashed char-n-gram TF-IDF, l2-normalized. Deterministic per corpus."""
    counts = np.zeros((len(payloads), _LEXICAL_DIM), dtype=np.float64)
    for row, payload in enumerate(payloads):
        for gram in _char_ngrams(payload):
            counts[row, _bucket(gram)] += 1.0
    document_frequency = (counts > 0).sum(axis=0)
    idf = np.zeros(_LEXICAL_DIM, dtype=np.float64)
    present = document_frequency > 0
    idf[present] = 1.0 + np.log(
        (1.0 + len(payloads)) / (1.0 + document_frequency[present])
    )
    weighted = np.log1p(counts) * idf
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return weighted / norms


def record_vectors(records: list[ExampleRecord]) -> tuple[np.ndarray, str]:
    """L2-normalized payload vectors + the engine that produced them."""
    payloads = [semantic_payload(record) for record in records]
    engine = similarity_engine()
    if engine == "embeddings":
        embedder = _load_embedder()
        assert embedder is not None
        vectors = np.asarray(
            embedder.encode(payloads, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float64,
        )
        return vectors, engine
    return _lexical_vectors(payloads), engine


def default_threshold(engine: str) -> float:
    return EMBEDDING_THRESHOLD if engine == "embeddings" else LEXICAL_THRESHOLD


def apply_semantic_dedup(
    records: list[ExampleRecord],
    *,
    threshold: float | None = None,
    exempt_families: frozenset[str] = DELIBERATE_VARIANT_FAMILIES,
) -> tuple[list[ExampleRecord], list[dict[str, str]]]:
    """Collapse cross-structure semantic near-duplicates to one leader each.

    Leader clustering in FAMILY_PRIORITY-then-id order (same keep policy as the
    MinHash pass): a record whose cosine similarity to any kept leader meets the
    threshold is dropped with ``reason="semantic_cosine"``. Records from
    ``exempt_families`` are kept unconditionally and never serve as leaders.
    """
    exempt_ids = {r.id for r in records if _record_family(r) in exempt_families}
    pool = [r for r in records if r.id not in exempt_ids]
    if len(pool) < 2:
        return list(records), []
    vectors, engine = record_vectors(pool)
    cutoff = float(threshold) if threshold is not None else default_threshold(engine)
    if cutoff <= 0.0 or math.isnan(cutoff):
        return list(records), []

    order = sorted(range(len(pool)), key=lambda i: _keep_key(pool[i]))
    leader_rows: list[int] = []
    leader_matrix: np.ndarray | None = None
    kept_ids: set[str] = set(exempt_ids)
    dropped: list[dict[str, str]] = []
    for index in order:
        record = pool[index]
        vector = vectors[index]
        if leader_matrix is not None and len(leader_rows):
            similarities = leader_matrix @ vector
            best = int(np.argmax(similarities))
            if float(similarities[best]) >= cutoff:
                leader = pool[leader_rows[best]]
                dropped.append(
                    {
                        "id": record.id,
                        "duplicate_of": leader.id,
                        "similarity": round(float(similarities[best]), 4),
                        "engine": engine,
                        "reason": "semantic_cosine",
                    }
                )
                continue
        leader_rows.append(index)
        leader_matrix = vectors[leader_rows]
        kept_ids.add(record.id)
    kept = [record for record in records if record.id in kept_ids]
    return kept, dropped


__all__ = [
    "DELIBERATE_VARIANT_FAMILIES",
    "EMBEDDING_THRESHOLD",
    "LEXICAL_THRESHOLD",
    "apply_semantic_dedup",
    "default_threshold",
    "record_vectors",
    "semantic_payload",
    "similarity_engine",
]
