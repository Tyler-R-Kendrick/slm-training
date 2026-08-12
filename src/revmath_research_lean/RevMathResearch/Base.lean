/-
  Research RCA₀ fragment markers. Not a full second-order arithmetic kernel —
  only the named base-theory identity required by InterpretationPackageV1.
-/
import RevMathResearch.Coding

namespace RevMathResearch.Base

open RevMathResearch.Coding

/-- Research-only base-theory tag (matches `baseTheoryId`). -/
structure Rca0Research where
  id : String := baseTheoryId
  deriving Repr

def rca0 : Rca0Research := {}

/-- Marker that production finite-kernel paths must not import. -/
def researchOnlyAuthority : Bool := true

#guard rca0.id = "RCA0"
#guard researchOnlyAuthority = true

end RevMathResearch.Base
