/-
  Explicit interpretation package. `#print axioms` on the certificates is
  insufficient for Big-Five labeling — callers must read this module's
  coding/base/status fields and the named axiom list.
-/
import RevMathResearch.Base
import RevMathResearch.Forward
import RevMathResearch.Reverse

namespace RevMathResearch.Interpretation

open RevMathResearch.Coding RevMathResearch.Principles
open RevMathResearch.Forward RevMathResearch.Reverse

structure InterpretationPackage where
  baseTheoryId : String := Coding.baseTheoryId
  codingId : String := Coding.codingId
  interpretationStatus : String := "explicit_reversal"
  targetSubsystem : String := Coding.targetSubsystem
  principleStatement : String := wklStatement
  theoremStatement : String := sepStatement
  namedAxioms : List String := [
    "RevMathResearch.Forward.wkl_implies_sigma01_sep",
    "RevMathResearch.Reverse.sigma01_sep_implies_wkl",
    "RevMathResearch.Reverse.not_both_dead_bridge"
  ]
  printAxiomsAloneSufficient : Bool := false
  productionAuthority : Bool := false

def package : InterpretationPackage := {}

/-- Bidirectional classification: both directions from the explicit certificates. -/
structure BidirectionalClassification where
  forward : WKL → Sigma01Sep := forwardCert
  reverse : Sigma01Sep → WKL := reverseCert
  package : InterpretationPackage := package

def classification : BidirectionalClassification := {}

theorem forward_from_wkl (h : WKL) : Sigma01Sep := classification.forward h
theorem reverse_from_sep (h : Sigma01Sep) : WKL := classification.reverse h

#guard package.interpretationStatus = "explicit_reversal"
#guard package.printAxiomsAloneSufficient = false
#guard package.productionAuthority = false
#guard package.codingId = "coding.wkl_sigma01_sep.v1"
#guard package.namedAxioms.length = 3

end RevMathResearch.Interpretation
