"""Arithmetic-sketch DSL pack (G4 / SLM-36): checkable-answer reasoning traces.

The validity oracle IS the deterministic straight-line evaluator: a program
is valid iff it parses and its `root` binding evaluates to a finite number
after ref resolution (the transformer's resolve step is the "deterministic
expansion for bound spans"). The same evaluator scores answers, so validity
and correctness share one code path — no separate judge to drift.
"""

from __future__ import annotations

import random
import operator
from decimal import Decimal

from slm_training.dsl.schema import ExampleRecord


def evaluate_answer(source: str) -> float:
    """Deterministically evaluate a sketch program's `root` answer.

    Raises ValueError on anything invalid: parse failure, missing root,
    undefined refs, cycles, division by zero, non-numeric results.
    """
    from slm_training.dsl.grammar.backends import get_backend

    try:
        data = get_backend("arith-sketch").resolved_data(source)
    except RecursionError as exc:  # cyclic refs blow the resolver
        raise ValueError("cyclic reference in sketch program") from exc
    except Exception as exc:  # noqa: BLE001 - normalize to one oracle error
        raise ValueError(f"invalid sketch program: {exc}") from exc
    if "root" not in (data.get("bindings") or {}):
        raise ValueError("sketch program must bind `root` to the answer")
    import math

    value = _eval_node(data.get("root"))
    if not math.isfinite(value):
        raise ValueError("non-finite arithmetic result")
    return value


def _eval_node(node: object) -> float:
    if isinstance(node, bool):
        raise ValueError("boolean has no numeric value")
    if isinstance(node, (int, float)):
        value = float(node)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite value")
        return value
    if isinstance(node, dict):
        if node.get("k") == "BinOp":
            left = _eval_node(node.get("left"))
            right = _eval_node(node.get("right"))
            op = node.get("op")
            operations = {"+": operator.add, "-": operator.sub,
                          "*": operator.mul, "/": operator.truediv}
            try:
                return operations[op](left, right)
            except KeyError as exc:
                raise ValueError(f"unsupported operator {op!r}") from exc
            except ZeroDivisionError as exc:
                raise ValueError("division by zero") from exc
        if node.get("k") == "UnaryOp" and node.get("op") == "-":
            return -_eval_node(node.get("operand"))
        if node.get("type") == "ref":
            # An unresolved ref after the transformer's resolve pass means
            # the name was never bound.
            raise ValueError(f"undefined reference {node.get('name')!r}")
    raise ValueError(f"non-numeric node {type(node).__name__}")


_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "Ada has {a} apples and buys {b} bags with {c} apples each. "
        "How many apples does Ada have now?",
        "x = {a}\ny = {b} * {c}\nroot = x + y",
    ),
    (
        "A crate holds {a} bottles. {b} crates arrive and {c} bottles break. "
        "How many intact bottles arrived?",
        "x = {a} * {b}\nroot = x - {c}",
    ),
    (
        "Ben reads {a} pages a day for {b} days, then {c} pages a day for "
        "{d} days. How many pages did Ben read?",
        "x = {a} * {b}\ny = {c} * {d}\nroot = x + y",
    ),
    (
        "A tank holds {a} liters. A pump adds {b} liters a minute for {c} "
        "minutes. How many liters are in the tank?",
        "x = {b} * {c}\nroot = {a} + x",
    ),
    (
        "Mia had {a} coins, spent {b}, then her {c} friends each gave her "
        "{d} coins. How many coins does Mia have?",
        "x = {a} - {b}\ny = {c} * {d}\nroot = x + y",
    ),
    (
        "Each of {a} shelves holds {b} books. {c} books are checked out. "
        "How many books remain?",
        "x = {a} * {b}\nroot = x - {c}",
    ),
)


def _corpus_generator(count: int, seed: int) -> list[ExampleRecord]:
    rng = random.Random(seed)
    records: list[ExampleRecord] = []
    for index in range(count):
        prompt_tpl, program_tpl = _TEMPLATES[index % len(_TEMPLATES)]
        values = {key: rng.randint(2, 12) for key in ("a", "b", "c", "d")}
        # Keep subtraction results non-negative so the word problems stay
        # coherent (the DSL itself allows negatives).
        if "-" in program_tpl:
            values["a"] = values.get("a", 2) + 12
        prompt = prompt_tpl.format(**values)
        source = canonical_training_source(program_tpl.format(**values))
        gold = evaluate_answer(source)
        records.append(
            ExampleRecord(
                id=f"arith_{seed}_{index}",
                prompt=prompt,
                openui=source,
                placeholders=[],
                meta={"gold_answer": gold, "task": "generation", "pack_id": "arith-sketch"},
            )
        )
    return records


def _validity_oracle(source: str, output_kind: str = "document") -> object:
    if output_kind != "document":
        raise ValueError("arith-sketch pack validates documents only")
    evaluate_answer(source)
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend("arith-sketch").parse(source)


def _canonicalize(source: str) -> str:
    # Identity normal form (stated in notes): evaluated-valid, stripped.
    evaluate_answer(source)
    return source.strip()


def _canonical_equal(a: str, b: str) -> bool:
    return _canonicalize(a) == _canonicalize(b)


def _scope_check(source: str):
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend("arith-sketch").stream_check(source)


def _training_validator(record: ExampleRecord) -> None:
    if record.target_kind != "document" or record.placeholders:
        raise ValueError("arithmetic training requires a placeholder-free document")
    # Parser alphabet contains only binders, numbers and arithmetic operators.
    # Oracle additionally rejects missing/undefined/cyclic roots and nonnumbers.
    evaluate_answer(record.openui)
    for target in record.accepted_outputs:
        if target.kind != "document":
            raise ValueError("arithmetic alternatives must be documents")
        evaluate_answer(target.text)


def build_canonical_pack(pack_type, policy_type, completion_domain):
    """Adapt existing arithmetic authority to canonical training/decode slots."""
    import re
    from functools import partial
    from slm_training.dsl.grammar.backends import get_backend
    from slm_training.dsl.grammar_capabilities import GrammarCapabilityAuthorityV1

    return pack_type(
        pack_id="arith-sketch",
        backend=get_backend("arith-sketch"),
        placeholder_policy=policy_type(
            placeholder_re=re.compile(r"(?!)"), content_props=frozenset(),
            slot_contract=lambda source, declared=(): tuple(declared or ()),
            is_placeholder=lambda value: False, extract=lambda source: [],
        ),
        reward_label="arithmetic_evaluation",
        canonicalize=canonical_training_source,
        oracle=_validity_oracle,
        training_validator=_training_validator,
        corpus_generator=_corpus_generator,
        grammar_capability_authority=GrammarCapabilityAuthorityV1(
            completion_domain=partial(completion_domain, evaluate_answer=evaluate_answer),
        ),
    )


def canonical_training_source(source: str) -> str:
    """Expand bound spans through the existing arithmetic AST, without solving."""
    from slm_training.dsl.grammar.backends import get_backend

    evaluate_answer(source)
    root = get_backend("arith-sketch").resolved_data(source)["root"]
    return "root = " + _expression_source(root)


def _expression_source(node) -> str:
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        literal = format(Decimal(str(abs(node))), "f")
        return f"(0 - {literal})" if node < 0 else literal
    if isinstance(node, dict) and node.get("k") == "BinOp":
        return f"({_expression_source(node['left'])} {node['op']} {_expression_source(node['right'])})"
    raise ValueError("arithmetic training AST contains an unsupported symbol")
