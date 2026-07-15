"""OpenUI language-contract identity.

Pins and hashes the exact language surface this repo targets so every dataset can
be stamped with a stable ``contract_id``. A change to the OpenUI package versions,
the grammar, the prop-order table, or the output tokenizers yields a new
``contract_id`` — i.e. a new dataset version. Because a component's positional
argument order is derived from its schema, silently changing the component library
can change the language the model accepts; binding datasets to ``contract_id``
makes that break loud instead of silent.

Scope note: the installed OpenUI Lang is the **0.2.x subset** (``@openuidev/lang-core``
tops out at 0.2.9). Full Lang v0.5 (state / queries / mutations / actions / tools)
has no published package yet; when it ships, extending the grammar / codec /
tokenizer is a contract *version bump* here, not a redesign.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# Language spec the repo currently targets (see module docstring).
LANG_SPEC = "openui-lang-0.2.x"

# src/slm_training/dsl/language_contract.py -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRIDGE_PACKAGE_JSON = _REPO_ROOT / "src" / "apps" / "openui_bridge" / "package.json"
_GRAMMAR_FILES = (
    _REPO_ROOT / "src" / "slm_training" / "dsl" / "grammars" / "openui.lark",
    _REPO_ROOT
    / "src"
    / "slm_training"
    / "dsl"
    / "grammars"
    / "openui_prop_order.json",
)
# The official OpenUI packages whose versions define the language surface.
_OPENUI_PACKAGES = (
    "@openuidev/lang-core",
    "@openuidev/react-lang",
    "@openuidev/react-ui",
    "@openuidev/react-headless",
)


def _sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_openui_versions(package_json: Path) -> dict[str, str]:
    data = json.loads(package_json.read_text(encoding="utf-8"))
    deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
    return {name: str(deps[name]) for name in _OPENUI_PACKAGES if name in deps}


@dataclass(frozen=True)
class LanguageContract:
    """Immutable identity of the OpenUI language surface a dataset targets."""

    lang_spec: str
    openui_versions: tuple[tuple[str, str], ...]
    grammar_sha256: str
    tokenizer_version: int
    dsl_tokenizer_version: int

    @property
    def contract_id(self) -> str:
        """Stable 16-hex identity of this contract."""
        payload = json.dumps(
            {
                "lang_spec": self.lang_spec,
                "openui_versions": list(self.openui_versions),
                "grammar_sha256": self.grammar_sha256,
                "tokenizer_version": self.tokenizer_version,
                "dsl_tokenizer_version": self.dsl_tokenizer_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "lang_spec": self.lang_spec,
            "openui_versions": dict(self.openui_versions),
            "grammar_sha256": self.grammar_sha256,
            "tokenizer_version": self.tokenizer_version,
            "dsl_tokenizer_version": self.dsl_tokenizer_version,
            "contract_id": self.contract_id,
        }


@lru_cache(maxsize=1)
def current_contract() -> LanguageContract:
    """Build the contract from the repo's pinned OpenUI surface (offline, deterministic)."""
    # Lazy imports keep this lightweight module free of the tokenizers' heavy deps
    # and any import cycle with ``slm_training.models``.
    from slm_training.models.dsl_tokenizer import DSL_TOKENIZER_VERSION
    from slm_training.models.tokenizer import TOKENIZER_VERSION

    versions = _read_openui_versions(_BRIDGE_PACKAGE_JSON)
    return LanguageContract(
        lang_spec=LANG_SPEC,
        openui_versions=tuple(sorted(versions.items())),
        grammar_sha256=_sha256_files(_GRAMMAR_FILES),
        tokenizer_version=int(TOKENIZER_VERSION),
        dsl_tokenizer_version=int(DSL_TOKENIZER_VERSION),
    )


def contract_id() -> str:
    """Stable 16-hex identity of the current language contract."""
    return current_contract().contract_id
