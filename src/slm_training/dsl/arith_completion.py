"""Canonical arithmetic completion authority over the shared lexer alphabet."""

from functools import lru_cache
import re

from slm_training.dsl.grammar_capabilities import (
    CompletionDomainCandidateV1, CompletionDomainV1,
)


_MAX_WITNESS_NODES = 256

def _zero_divisor(text, evaluate_answer):
    """A closed divisor atom cannot be repaired by any later continuation."""
    for match in re.finditer(r"/\s*", text):
        suffix = text[match.end():]
        if suffix.startswith("("):
            depth = 0
            for index, char in enumerate(suffix):
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    suffix = suffix[:index + 1]
                    break
            else:
                continue
        else:
            number = re.match(r"\d+(?:\.\d+)?", suffix)
            if number is None:
                continue
            suffix = number.group()
        try:
            if evaluate_answer("root = " + suffix) == 0:
                return True
        except ValueError:
            continue
    return False


class _Search:
    def __init__(self, request, evaluate_answer):
        from slm_training.dsl.grammar.fastpath.engine import engine_for_dsl

        self.evaluate_answer = evaluate_answer
        self.frontier = lru_cache(maxsize=4096)(self._frontier)
        self.tok = request.tokenizer
        self.engine = engine_for_dsl("arith-sketch")
        self.opener = self.tok.token_to_id["LIT_NUM"]
        self.closer = self.tok.token_to_id["LIT_END"]
        self.order = [self.tok.eos_id, self.closer, self.tok.token_to_id["="],
                      self.tok.token_to_id[")"], self.opener,
                      self.tok.token_to_id["B:31"], self.tok.token_to_id["+"]]

    def frame(self, prefix):
        start = -1
        for index, token in enumerate(prefix):
            if token == self.opener:
                start = index
            elif token == self.closer:
                start = -1
        if start < 0:
            return None
        return "".join(chr(int(self.tok.id_to_token[t][2:], 16)) for t in prefix[start + 1:])

    def _frontier(self, prefix):
        from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set, decode_prefix

        body = self.frame(prefix)
        if body is not None:
            choices = {self.tok.token_to_id[f"B:{ord(c):02x}"] for c in "0123456789"}
            if body and "." not in body:
                choices.add(self.tok.token_to_id["B:2e"])
            if body and not body.endswith("."):
                choices.add(self.closer)
            return tuple(choices)
        text = decode_prefix(self.tok, list(prefix))
        if _zero_divisor(text, self.evaluate_answer) or not self.engine.set_prefix(text):
            return ()
        choices = set(allowed_id_set(self.tok, self.engine.next_terminals()) or ())
        # Canonical representation is a single resolved root expression.
        choices = {t for t in choices if not self.tok.id_to_token[t].startswith("<BIND_")}
        choices.discard(self.tok.token_to_id["NL"])
        if not text.strip():
            choices.add(self.tok.bind_id(0))
        try:
            self.evaluate_answer(text)
            choices.add(self.tok.eos_id)
        except ValueError:
            choices.discard(self.tok.eos_id)
        return tuple(choices)

    def witness(self, prefix, room):
        self.nodes = _MAX_WITNESS_NODES
        self.unknown = False
        result = self._visit(prefix, room)
        return result, self.unknown

    def _visit(self, prefix, room):
        if room < 1:
            return None
        if self.nodes < 1:
            self.unknown = True
            return None
        self.nodes -= 1
        choices = sorted(self.frontier(prefix), key=self._priority)
        for token in choices:
            if token == self.tok.eos_id:
                return (token,)
            tail = self._visit(prefix + (token,), room - 1)
            if tail is not None:
                return (token,) + tail
        return None

    def _priority(self, token):
        return self.order.index(token) if token in self.order else len(self.order)


def completion_domain(request, evaluate_answer):
    if request.remaining_tokens is None or request.remaining_tokens < 1:
        return CompletionDomainV1("unsupported", reason="arithmetic_requires_finite_budget")
    if request.runtime_symbols or request.slot_contract:
        return CompletionDomainV1("unsupported", reason="arithmetic_has_no_external_symbols")
    search = _Search(request, evaluate_answer)
    prefix = tuple(request.prefix_ids)
    candidates = []
    for token in search.frontier(prefix):
        if token == search.tok.eos_id:
            tail, unknown = (), False
        else:
            tail, unknown = search.witness(prefix + (token,), request.remaining_tokens - 1)
        if tail is None:
            if unknown:
                return CompletionDomainV1("incomplete", reason="arithmetic_terminal_witness_unknown")
            continue
        candidates.append(CompletionDomainCandidateV1((token,), "arithmetic", (token,) + tail))
    return CompletionDomainV1("complete", tuple(candidates))
