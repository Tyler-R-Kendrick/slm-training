"""Certified, reusable static authority for OpenUI completion.

The artifact contains only request-independent facts:

* the compiled LALR control table;
* the frozen tokenizer-id to grammar-terminal projection.

Scope, symbol visibility, literal bodies, macro expansions, and semantic state
remain request-local.  The loader accepts an artifact only after hashes and an
independent reconstruction from the live grammar/tokenizer agree exactly.
"""

from __future__ import annotations

import hashlib
import json
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.dsl.grammar.fastpath.token_map import dsl_direct_terminal_map

ARTIFACT_SCHEMA = "openui_completion_artifact/v1"
CHECKER_SCHEMA = "openui_completion_checker/v1"
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).parents[3] / "resources/decode/openui_completion_v1.safetensors"
)
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACT_PATH.with_suffix(".manifest.json")

_ROLE_NONE = 0
_ROLE_PUNCT = 1
_ROLE_BOOL = 2
_ROLE_NULL = 3
_ROLE_KIND = 4
_ROLE_STRING_LITERAL = 5
_ROLE_LITERAL_STRING_OPEN = 6
_ROLE_LITERAL_NUMBER_OPEN = 7
_ROLE_LITERAL_END = 8
_ROLE_SKIP = 9


@dataclass(frozen=True)
class CompletionArtifact:
    """A checked artifact and the exact direct-feed map it certifies."""

    manifest: dict[str, Any]
    direct_map: dict[str, Any]

    @property
    def digest(self) -> str:
        return str(self.manifest["artifact_sha256"])


def completion_artifact_checkpoint_identity(
    *,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, str]:
    """Return the hash-bound identity embedded in new native checkpoints."""

    raw = artifact_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256_bytes(raw)
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported completion artifact schema")
    if manifest.get("artifact_sha256") != digest:
        raise ValueError("completion artifact digest mismatch at checkpoint save")
    return {
        "schema": str(manifest["schema"]),
        "checker_schema": str(manifest["checker_schema"]),
        "sha256": digest,
        "grammar_sha256": str(manifest["grammar_sha256"]),
        "tokenizer_authority_sha256": str(manifest["tokenizer_authority_sha256"]),
    }


def require_checkpoint_completion_artifact(payload: dict[str, Any]) -> None:
    """Reject a checkpoint whose declared static authority no longer matches.

    Legacy checkpoints without a declaration remain loadable and receive the
    ordinary runtime checker/fallback behavior; only a present claim is strict.
    """

    declared = payload.get("completion_artifact")
    if declared is None:
        return
    if not isinstance(declared, dict):
        raise ValueError("invalid checkpoint completion_artifact identity")
    current = completion_artifact_checkpoint_identity()
    if declared != current:
        raise ValueError(
            "checkpoint completion artifact does not match the installed "
            "grammar/tokenizer authority"
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tokenizer_authority(tokenizer: Any) -> dict[str, Any]:
    """Canonical request-independent tokenizer authority."""

    return {
        "version": int(getattr(tokenizer, "version", 0)),
        "sym_slots": int(getattr(tokenizer, "sym_slots", 0)),
        "bind_slots": int(getattr(tokenizer, "bind_slots", 0)),
        "state_slots": int(getattr(tokenizer, "state_slots", 0)),
        "macro_slots": int(getattr(tokenizer, "macro_slots", 0)),
        "abstract_plan_slots": int(getattr(tokenizer, "abstract_plan_slots", 0)),
        "bind_encoding": str(getattr(tokenizer, "bind_encoding", "")),
        "token_to_id": sorted(
            (str(token), int(token_id))
            for token, token_id in tokenizer.token_to_id.items()
        ),
        "id_to_kind": sorted(
            (int(token_id), str(kind))
            for token_id, kind in tokenizer.id_to_kind.items()
        ),
        "special_ids": {
            name: int(getattr(tokenizer, name))
            for name in ("pad_id", "bos_id", "eos_id", "mask_id", "unk_id")
        },
    }


def grammar_terminal_authority(terminals: Any) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "name": str(terminal.name),
                "pattern_type": str(terminal.pattern.type),
                "pattern_value": str(terminal.pattern.value),
                "priority": int(terminal.priority),
            }
            for terminal in terminals
        ),
        key=lambda item: item["name"],
    )


def _flatten_direct_map(
    tokenizer: Any, direct_map: dict[str, Any], terminal_names: list[str]
) -> tuple[list[int], list[int]]:
    terminal_index = {name: index for index, name in enumerate(terminal_names)}
    roles = [_ROLE_NONE] * int(tokenizer.vocab_size)
    terminals = [-1] * int(tokenizer.vocab_size)

    def assign(token_id: int, role: int, terminal: str | None = None) -> None:
        token_id = int(token_id)
        if roles[token_id] != _ROLE_NONE:
            raise ValueError(f"overlapping static completion role for token {token_id}")
        roles[token_id] = role
        terminals[token_id] = terminal_index[terminal] if terminal is not None else -1

    for token_id, terminal in direct_map["punct"].items():
        assign(token_id, _ROLE_PUNCT, terminal)
    for token_id, terminal in direct_map["bool"].items():
        assign(token_id, _ROLE_BOOL, terminal)
    for token_id, terminal in direct_map["null"].items():
        assign(token_id, _ROLE_NULL, terminal)
    for kind, terminal in direct_map["kind_terminals"].items():
        for token_id in tokenizer.kind_ids(kind):
            assign(token_id, _ROLE_KIND, terminal)
    for token_id, terminal in direct_map["str_lit_ids"].items():
        assign(token_id, _ROLE_STRING_LITERAL, terminal)
    if direct_map["lit_str"] is not None:
        assign(direct_map["lit_str"], _ROLE_LITERAL_STRING_OPEN)
    assign(direct_map["lit_num"], _ROLE_LITERAL_NUMBER_OPEN)
    assign(direct_map["lit_end"], _ROLE_LITERAL_END)
    for token_id in sorted(direct_map["skip_ids"]):
        assign(token_id, _ROLE_SKIP)
    return roles, terminals


def _reconstruct_direct_map(
    tokenizer: Any,
    terminal_names: list[str],
    roles: list[int],
    terminals: list[int],
) -> dict[str, Any]:
    from slm_training.models.dsl_tokenizer import TokenKind

    result: dict[str, Any] = {
        "punct": {},
        "bool": {},
        "null": {},
        "kind_terminals": {},
        "str_lit_ids": {},
        "lit_str": None,
        "lit_num": None,
        "lit_end": None,
        "skip_ids": set(),
    }
    kind_by_terminal: dict[str, TokenKind] = {}
    for token_id, role in enumerate(roles):
        terminal_index = terminals[token_id]
        terminal = terminal_names[terminal_index] if terminal_index >= 0 else None
        if role == _ROLE_NONE:
            continue
        if role == _ROLE_PUNCT:
            result["punct"][token_id] = terminal
        elif role == _ROLE_BOOL:
            result["bool"][token_id] = terminal
        elif role == _ROLE_NULL:
            result["null"][token_id] = terminal
        elif role == _ROLE_KIND:
            kind = tokenizer.kind_of(token_id)
            prior = kind_by_terminal.setdefault(str(terminal), kind)
            if prior is not kind:
                raise ValueError("one terminal maps to multiple tokenizer kinds")
            result["kind_terminals"][kind] = terminal
        elif role == _ROLE_STRING_LITERAL:
            result["str_lit_ids"][token_id] = terminal
        elif role == _ROLE_LITERAL_STRING_OPEN:
            result["lit_str"] = token_id
        elif role == _ROLE_LITERAL_NUMBER_OPEN:
            result["lit_num"] = token_id
        elif role == _ROLE_LITERAL_END:
            result["lit_end"] = token_id
        elif role == _ROLE_SKIP:
            result["skip_ids"].add(token_id)
        else:
            raise ValueError(f"unknown static completion role {role}")
    return result


def _parse_table_arrays(parser: Any) -> tuple[dict[str, list[int]], dict[str, Any]]:
    table = parser.parser.parser._parse_table
    symbols = sorted({str(symbol) for row in table.states.values() for symbol in row})
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    rules = list(parser.rules)
    rule_key = {repr(rule): index for index, rule in enumerate(rules)}

    # Lark's numeric state ids depend on Python set iteration and can change
    # between processes. Canonicalize the graph by stable partition refinement:
    # shift targets are represented by their recursively refined state color,
    # while reductions use canonical rule ids. Equal colors are behaviorally
    # bisimilar for this labeled deterministic control graph.
    def normalized_colors(signatures: dict[Any, tuple]) -> dict[Any, int]:
        palette = {
            signature: index
            for index, signature in enumerate(sorted(set(signatures.values())))
        }
        return {state: palette[signature] for state, signature in signatures.items()}

    signatures = {
        state: tuple(
            sorted(
                (
                    str(symbol),
                    str(action[0]),
                    rule_key[repr(action[1])] if str(action[0]) == "Reduce" else -1,
                )
                for symbol, action in row.items()
            )
        )
        for state, row in table.states.items()
    }
    colors = normalized_colors(signatures)
    for _ in range(len(table.states) + 1):
        refined = {
            state: tuple(
                sorted(
                    (
                        str(symbol),
                        str(action[0]),
                        (
                            colors[action[1]]
                            if str(action[0]) == "Shift"
                            else rule_key[repr(action[1])]
                        ),
                    )
                    for symbol, action in row.items()
                )
            )
            for state, row in table.states.items()
        }
        next_colors = normalized_colors(refined)
        if next_colors == colors:
            break
        colors = next_colors
    else:  # pragma: no cover - finite refinement stabilizes in <= state count
        raise RuntimeError("LALR control graph canonicalization did not converge")

    canonical_rows = sorted(
        (
            colors[state],
            tuple(
                sorted(
                    (
                        str(symbol),
                        str(action[0]),
                        (
                            colors[action[1]]
                            if str(action[0]) == "Shift"
                            else rule_key[repr(action[1])]
                        ),
                    )
                    for symbol, action in row.items()
                )
            ),
        )
        for state, row in table.states.items()
    )
    offsets = [0]
    edge_symbols: list[int] = []
    action_kinds: list[int] = []
    action_targets: list[int] = []
    state_colors: list[int] = []
    for color, row in canonical_rows:
        state_colors.append(color)
        for symbol, action_name, target in row:
            edge_symbols.append(symbol_index[symbol])
            if action_name == "Shift":
                action_kinds.append(0)
                action_targets.append(target)
            elif action_name == "Reduce":
                action_kinds.append(1)
                action_targets.append(target)
            else:
                raise ValueError(f"unsupported LALR action {action_name!r}")
        offsets.append(len(edge_symbols))
    metadata = {
        "symbols": symbols,
        "rules": [repr(rule) for rule in rules],
        "start_states": {
            str(name): colors[state] for name, state in table.start_states.items()
        },
        "end_states": {
            str(name): colors[state] for name, state in table.end_states.items()
        },
    }
    return {
        "lalr_state_ids": list(range(len(canonical_rows))),
        "lalr_state_colors": state_colors,
        "lalr_state_offsets": offsets,
        "lalr_edge_symbols": edge_symbols,
        "lalr_action_kinds": action_kinds,
        "lalr_action_targets": action_targets,
    }, metadata


def _encode_safetensors(
    tensors: dict[str, list[int]], metadata: dict[str, str]
) -> bytes:
    header: dict[str, Any] = {"__metadata__": metadata}
    data = bytearray()
    for name in sorted(tensors):
        values = array("q", (int(value) for value in tensors[name]))
        if values.itemsize != 8:
            raise RuntimeError("platform does not provide 64-bit signed arrays")
        if struct.pack("=q", 1) != struct.pack("<q", 1):
            values.byteswap()
        raw = values.tobytes()
        start = len(data)
        data.extend(raw)
        header[name] = {
            "dtype": "I64",
            "shape": [len(values)],
            "data_offsets": [start, len(data)],
        }
    encoded_header = _canonical_json(header)
    encoded_header += b" " * ((8 - len(encoded_header) % 8) % 8)
    return struct.pack("<Q", len(encoded_header)) + encoded_header + bytes(data)


def _decode_safetensors(raw: bytes) -> tuple[dict[str, list[int]], dict[str, str]]:
    if len(raw) < 8:
        raise ValueError("truncated safetensors artifact")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length > len(raw) - 8:
        raise ValueError("invalid safetensors header length")
    header = json.loads(raw[8 : 8 + header_length])
    data = raw[8 + header_length :]
    tensors: dict[str, list[int]] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec.get("dtype") != "I64" or len(spec.get("shape") or []) != 1:
            raise ValueError(f"unsupported tensor declaration for {name}")
        start, end = (int(value) for value in spec["data_offsets"])
        if not 0 <= start <= end <= len(data) or (end - start) % 8:
            raise ValueError(f"invalid tensor offsets for {name}")
        values = array("q")
        values.frombytes(data[start:end])
        if struct.pack("=q", 1) != struct.pack("<q", 1):
            values.byteswap()
        if len(values) != int(spec["shape"][0]):
            raise ValueError(f"shape mismatch for {name}")
        tensors[name] = list(values)
    return tensors, dict(header.get("__metadata__") or {})


def build_completion_artifact(
    tokenizer: Any,
    parser: Any,
    *,
    grammar_path: Path,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Compile and write the static artifact plus its proof manifest."""

    direct_map = dsl_direct_terminal_map(tokenizer, parser.terminals)
    if direct_map is None:
        raise ValueError("live tokenizer/grammar projection is not certifiable")
    terminal_authority = grammar_terminal_authority(parser.terminals)
    terminal_names = sorted(item["name"] for item in terminal_authority)
    roles, terminals = _flatten_direct_map(tokenizer, direct_map, terminal_names)
    control_tensors, control_metadata = _parse_table_arrays(parser)
    tensors = {
        **control_tensors,
        "token_roles": roles,
        "token_terminals": terminals,
    }
    tokenizer_payload = tokenizer_authority(tokenizer)
    grammar_bytes = Path(grammar_path).read_bytes()
    artifact_bytes = _encode_safetensors(
        tensors,
        {
            "schema": ARTIFACT_SCHEMA,
            "terminal_names": json.dumps(terminal_names, separators=(",", ":")),
        },
    )
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "checker_schema": CHECKER_SCHEMA,
        "artifact": artifact_path.name,
        "artifact_sha256": _sha256_bytes(artifact_bytes),
        "grammar_sha256": _sha256_bytes(grammar_bytes),
        "grammar_terminal_sha256": _sha256_bytes(_canonical_json(terminal_authority)),
        "tokenizer_authority_sha256": _sha256_bytes(_canonical_json(tokenizer_payload)),
        "vocab_size": int(tokenizer.vocab_size),
        "lalr_state_count": len(tensors["lalr_state_ids"]),
        "lalr_edge_count": len(tensors["lalr_edge_symbols"]),
        "direct_token_count": sum(role != _ROLE_NONE for role in roles),
        "control": control_metadata,
        "proof_obligations": {
            "soundness": "artifact projection equals live grammar/tokenizer projection",
            "completeness": "all live direct-feed rows occur exactly once",
            "serialization": "decoded tensors reproduce all certified arrays",
            "request_overlay": "scope and semantic authority are excluded",
        },
        "request_dependent_authority_excluded": [
            "runtime symbols",
            "scope visibility",
            "literal bodies",
            "macro expansions",
            "semantic state",
        ],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(artifact_bytes)
    manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
    return manifest


def load_checked_completion_artifact(
    tokenizer: Any,
    parser: Any,
    *,
    grammar_path: Path,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> CompletionArtifact:
    """Load only when every certificate input matches live authority."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported completion artifact schema")
    raw = artifact_path.read_bytes()
    checks = {
        "artifact_sha256": _sha256_bytes(raw),
        "grammar_sha256": _sha256_bytes(Path(grammar_path).read_bytes()),
        "grammar_terminal_sha256": _sha256_bytes(
            _canonical_json(grammar_terminal_authority(parser.terminals))
        ),
        "tokenizer_authority_sha256": _sha256_bytes(
            _canonical_json(tokenizer_authority(tokenizer))
        ),
    }
    for key, actual in checks.items():
        if manifest.get(key) != actual:
            raise ValueError(f"completion artifact {key} mismatch")

    tensors, metadata = _decode_safetensors(raw)
    if metadata.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("artifact metadata schema mismatch")
    terminal_names = json.loads(metadata["terminal_names"])
    roles = tensors["token_roles"]
    terminal_ids = tensors["token_terminals"]
    if len(roles) != tokenizer.vocab_size or len(terminal_ids) != tokenizer.vocab_size:
        raise ValueError("artifact tokenizer row count mismatch")
    reconstructed = _reconstruct_direct_map(
        tokenizer, terminal_names, roles, terminal_ids
    )
    live = dsl_direct_terminal_map(tokenizer, parser.terminals)
    if live is None or reconstructed != live:
        raise ValueError("artifact direct map fails certified equivalence check")

    control_tensors, control_metadata = _parse_table_arrays(parser)
    for name, expected in control_tensors.items():
        if tensors.get(name) != expected:
            raise ValueError(f"artifact control table mismatch for {name}")
    if manifest.get("control") != control_metadata:
        raise ValueError("artifact control metadata mismatch")
    return CompletionArtifact(manifest=manifest, direct_map=reconstructed)


__all__ = [
    "ARTIFACT_SCHEMA",
    "CHECKER_SCHEMA",
    "CompletionArtifact",
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "build_completion_artifact",
    "completion_artifact_checkpoint_identity",
    "load_checked_completion_artifact",
    "require_checkpoint_completion_artifact",
    "tokenizer_authority",
]
