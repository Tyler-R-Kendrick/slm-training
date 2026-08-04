"""Certified, reusable static authority for OpenUI completion.

The artifact contains only request-independent facts:

* the compiled LALR control table;
* the rule tensors (origin symbol + expansion length per canonical rule id)
  that make the reduce actions executable;
* a control-only *lower bound* on how many terminals must still be consumed
  before the parse can accept, consumed by nothing;
* the frozen tokenizer-id to grammar-terminal projection.

Scope, symbol visibility, literal bodies, macro expansions, and semantic state
remain request-local.  The loader accepts an artifact only after hashes and an
independent reconstruction from the live grammar/tokenizer agree exactly.

``StaticLalrAdapter`` executes the control arrays, but Lark remains the parser
authority: the adapter is import-only until
``static_control_domain.require_certified_static_lalr`` proves accepts-set
equality against the live ``InteractiveParser`` in lockstep.
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
    lalr: "StaticLalrAdapter | None" = None

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
    rules = list(parser.rules)
    symbols = sorted(
        {str(symbol) for row in table.states.values() for symbol in row}
        | {str(rule.origin.name) for rule in rules}
    )
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
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
    rule_origins = [symbol_index[str(rule.origin.name)] for rule in rules]
    rule_lengths = [len(rule.expansion) for rule in rules]
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
    tensors = {
        "lalr_state_ids": list(range(len(canonical_rows))),
        "lalr_state_colors": state_colors,
        "lalr_state_offsets": offsets,
        "lalr_edge_symbols": edge_symbols,
        "lalr_action_kinds": action_kinds,
        "lalr_action_targets": action_targets,
        "rule_origins": rule_origins,
        "rule_lengths": rule_lengths,
    }
    tensors["state_min_terminals"] = _state_min_terminals(tensors, metadata)
    return tensors, metadata


# ---------------------------------------------------------------------------
# Acceptance-length lower bound (control-only; consumed by nothing)
# ---------------------------------------------------------------------------

_UNREACHABLE_MIN_TERMINALS = -1


def _control_rows(
    tensors: dict[str, list[int]],
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """Map canonical state color -> ``(symbol_index, action_kind, target)`` rows."""

    offsets = tensors["lalr_state_offsets"]
    rows: dict[int, tuple[tuple[int, int, int], ...]] = {}
    for index, color in enumerate(tensors["lalr_state_colors"]):
        row = tuple(
            (
                tensors["lalr_edge_symbols"][slot],
                tensors["lalr_action_kinds"][slot],
                tensors["lalr_action_targets"][slot],
            )
            for slot in range(offsets[index], offsets[index + 1])
        )
        previous = rows.setdefault(int(color), row)
        if previous != row:
            raise ValueError("canonical color collision with differing control rows")
    return rows


def _state_min_terminals(
    tensors: dict[str, list[int]], metadata: dict[str, Any]
) -> list[int]:
    """Sound LOWER bound on terminals still needed before the parse can accept.

    Acceptance is a property of the whole stack, not of the top state, so an
    exact per-state table does not exist.  This computes a *relaxation*: a
    reduction may expose any state that carries a goto on the reduced rule's
    origin, not only the state the real stack would expose.  The relaxed
    transition relation is a superset of every real one and reductions consume
    no terminals, therefore the fixpoint is ``<=`` the true stack-dependent
    minimum for every stack topped by that state — a lower bound, usable only
    for negative-direction reasoning ("fewer than this many tokens cannot
    finish").  Nothing in production consumes it yet.
    """

    rows = _control_rows(tensors)
    symbols = list(metadata["symbols"])
    rule_origins = tensors["rule_origins"]
    end_state = int(metadata["end_states"]["start"])

    goto_targets: dict[int, set[int]] = {}
    for row in rows.values():
        for symbol, kind, target in row:
            if kind == 0 and not symbols[symbol].isupper():
                goto_targets.setdefault(symbol, set()).add(int(target))

    infinity = len(rows) + 1
    cost = {color: infinity for color in rows}
    cost[end_state] = 0
    for _ in range(len(rows) + 2):
        changed = False
        for color, row in rows.items():
            best = cost[color]
            for symbol, kind, target in row:
                name = symbols[symbol]
                if kind == 0:
                    if not name.isupper() or name == "$END":
                        continue  # goto edges are traversed through reductions
                    candidate = cost[int(target)] + 1
                else:
                    origin = rule_origins[int(target)]
                    candidate = min(
                        (cost[state] for state in goto_targets.get(origin, ())),
                        default=infinity,
                    )
                best = min(best, candidate)
            if best < cost[color]:
                cost[color] = best
                changed = True
        if not changed:
            break
    else:  # pragma: no cover - monotone fixpoint converges in <= state count
        raise RuntimeError("acceptance-length fixpoint did not converge")

    return [
        (
            _UNREACHABLE_MIN_TERMINALS
            if cost[int(color)] >= infinity
            else int(cost[int(color)])
        )
        for color in tensors["lalr_state_colors"]
    ]


@dataclass(frozen=True)
class StaticLalrAdapter:
    """Executable shift/reduce over the certified control arrays.

    This is a *checked* adapter, never an authority: it is import-only until a
    lockstep certification against the live Lark ``InteractiveParser`` passes
    (see ``static_control_domain.require_certified_static_lalr``).  Nothing in
    the decode path consumes it.  ``$END`` follows Lark's end handling: the
    reduction chain runs until the end state is exposed.
    """

    symbols: tuple[str, ...]
    rows: dict[int, tuple[tuple[int, int, int], ...]]
    rule_origins: tuple[int, ...]
    rule_lengths: tuple[int, ...]
    start_state: int
    end_state: int
    state_min_terminals: tuple[int, ...]

    END_SYMBOL = "$END"

    @classmethod
    def from_arrays(
        cls,
        tensors: dict[str, list[int]],
        metadata: dict[str, Any],
        *,
        start: str = "start",
    ) -> StaticLalrAdapter:
        rows = _control_rows(tensors)
        colors = tensors["lalr_state_colors"]
        minimums = tensors["state_min_terminals"]
        by_color: dict[int, int] = {}
        for color, value in zip(colors, minimums, strict=True):
            if by_color.setdefault(int(color), int(value)) != int(value):
                raise ValueError("state_min_terminals disagrees within a color")
        if sorted(by_color) != list(range(len(by_color))):
            raise ValueError("canonical state colors are not densely numbered")
        return cls(
            symbols=tuple(str(name) for name in metadata["symbols"]),
            rows=rows,
            rule_origins=tuple(int(value) for value in tensors["rule_origins"]),
            rule_lengths=tuple(int(value) for value in tensors["rule_lengths"]),
            start_state=int(metadata["start_states"][start]),
            end_state=int(metadata["end_states"][start]),
            state_min_terminals=tuple(by_color[color] for color in sorted(by_color)),
        )

    def start_stack(self) -> tuple[int, ...]:
        return (int(self.start_state),)

    def _action(self, state: int, terminal: str) -> tuple[int, int] | None:
        for symbol, kind, target in self.rows[int(state)]:
            if self.symbols[symbol] == terminal:
                return int(kind), int(target)
        return None

    def step(
        self, stack: tuple[int, ...], terminal_name: str
    ) -> tuple[int, ...] | None:
        """Advance the stack over one terminal, or ``None`` when rejected."""

        working = [int(state) for state in stack]
        if not working:
            return None
        is_end = terminal_name == self.END_SYMBOL
        for _ in range(len(self.rows) * len(self.rule_lengths) + 8):
            action = self._action(working[-1], terminal_name)
            if action is None:
                return None
            kind, target = action
            if kind == 0:  # Shift
                if is_end:
                    # Lark never shifts $END; treat it as a rejection instead of
                    # inventing an acceptance the live parser would not grant.
                    return None
                working.append(target)
                return tuple(working)
            if not 0 <= target < len(self.rule_lengths):
                raise ValueError(f"reduce action targets unknown rule id {target}")
            size = self.rule_lengths[target]
            if size:
                if len(working) <= size:
                    return None
                del working[-size:]
            goto = self._action(working[-1], self.symbols[self.rule_origins[target]])
            if goto is None or goto[0] != 0:
                return None
            working.append(goto[1])
            if is_end and working[-1] == self.end_state:
                return tuple(working)
        raise RuntimeError("static LALR adapter reduction chain did not terminate")

    def accepts(self, stack: tuple[int, ...]) -> frozenset[str]:
        """Terminals the control table actually advances on from ``stack``.

        Mirrors ``InteractiveParser.accepts``: a terminal counts only when its
        whole reduction chain succeeds, not merely when the top state's row
        carries an entry for it.
        """

        names = {
            self.symbols[symbol] for symbol, _kind, _target in self.rows[stack[-1]]
        }
        return frozenset(
            name
            for name in names
            if name.isupper() and self.step(stack, name) is not None
        )

    def min_terminals(self, state: int) -> int:
        """Lower bound on remaining terminals; ``-1`` when no bound is known."""

        return int(self.state_min_terminals[int(state)])


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
        "lalr_rule_count": len(tensors["rule_lengths"]),
        "direct_token_count": sum(role != _ROLE_NONE for role in roles),
        "control": control_metadata,
        "state_min_terminals": {
            "kind": "control_only_lower_bound",
            "direction": "negative_only",
            "trivial": all(value <= 0 for value in tensors["state_min_terminals"]),
            "unknown_sentinel": _UNREACHABLE_MIN_TERMINALS,
            "max": max(tensors["state_min_terminals"]),
            "consumed_by": [],
        },
        "proof_obligations": {
            "soundness": "artifact projection equals live grammar/tokenizer projection",
            "completeness": "all live direct-feed rows occur exactly once",
            "serialization": "decoded tensors reproduce all certified arrays",
            "request_overlay": "scope and semantic authority are excluded",
            "rule_tensors": (
                "rule_origins/rule_lengths are ordered by the same canonical rule "
                "ids the reduce actions target"
            ),
            "state_min_terminals": (
                "control-only LOWER bound on terminals remaining before the parse "
                "can accept, computed by a monotone fixpoint that relaxes every "
                "reduction to any state carrying the origin goto; acceptance is "
                "stack-dependent, so this bound is safe for negative-direction "
                "pruning only and is consumed by nothing"
            ),
            "static_lalr_adapter": (
                "StaticLalrAdapter is import-only and is not an authority until "
                "lockstep certification against the live Lark InteractiveParser "
                "passes; Lark remains the parser authority"
            ),
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
    return CompletionArtifact(
        manifest=manifest,
        direct_map=reconstructed,
        lalr=StaticLalrAdapter.from_arrays(tensors, control_metadata),
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "CHECKER_SCHEMA",
    "CompletionArtifact",
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "StaticLalrAdapter",
    "build_completion_artifact",
    "completion_artifact_checkpoint_identity",
    "load_checked_completion_artifact",
    "require_checkpoint_completion_artifact",
    "tokenizer_authority",
]
