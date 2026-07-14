// Sketch only — not compiled in this repo.
// Maps OpenUI LALR accepts() singleton structural terminals to forced tokens.
// Mirror of slm_training.grammar_fastpath.force_emit / engine.is_deterministic_next.
#pragma once
#include <cstdint>
#include <optional>
#include <string_view>

namespace slm::cactus_sketch {

enum class Term : uint8_t {
  Equal,
  LPar,
  RPar,
  LSqb,
  RSqb,
  Comma,
  Broad,  // NAME / COMPONENT / STRING / …
};

inline std::optional<char> force_emit_char(Term narrow_singleton) {
  switch (narrow_singleton) {
    case Term::Equal: return '=';
    case Term::LPar: return '(';
    case Term::RPar: return ')';
    case Term::LSqb: return '[';
    case Term::RSqb: return ']';
    case Term::Comma: return ',';
    default: return std::nullopt;
  }
}

// Pseudocode:
//   accepts = interactive_parser.accepts(prefix);
//   narrow = accepts - {NAME,COMPONENT,STRING,NUMBER,BOOL,$END,...};
//   if |narrow|==1 → emit force_emit_char(narrow); skip transformer step.

}  // namespace slm::cactus_sketch
