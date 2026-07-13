// Sketch only — not compiled in this repo.
// Hole-admissibility for MaskGIT fills (constrained-diffusion-inspired).
// Mirror of slm_training.grammar_fastpath.maskgit_constrain.admit_fill.
#pragma once
#include <string>
#include <string_view>

namespace slm::cactus_sketch {

// Replace mask holes with a benign nonterminal stand-in ("hole"), then ask
// whether the OpenUI CFG still has a completion (parse succeeds or UnexpectedEOF /
// unmatched open delimiters only).
inline bool admit_fill_probe(std::string_view canvas_with_holes) {
  std::string probe(canvas_with_holes);
  // replace "<mask>" → "hole"; collapse "hole hole"; ensure trailing newline;
  // return parser.can_complete(probe);
  (void)probe;
  return true;  // stub — real engine lives in Cactus transpile of Lark DFA
}

}  // namespace slm::cactus_sketch
