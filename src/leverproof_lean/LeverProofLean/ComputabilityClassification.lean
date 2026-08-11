/-
  KERN-12 / SLM-532: practical computability classification vocabulary.

  Inductive mirrors Python PRACTICAL_CLASSES in
  slm_training.formal.computability_classification. String labels are the
  production-facing ids used by KERN-10 parity fixtures and HARN-08
  practical_class. `#print axioms` is never a Big-Five classifier here —
  research labels stay outside this inductive (rm_research:* is Python-only
  with an interpretation package).
-/
namespace LeverProofLean.ComputabilityClassification

inductive PracticalClass where
  | finiteDecidable
  | primitiveRecursive
  | boundedSearch
  | totalRecursive
  | semidecidable
  | coSemidecidable
  | oracleRelative
  | classicalPropositional
  | noncomputableExistence
  | unclassified
  deriving DecidableEq, Repr

/-- Canonical production label string (Python / JSON parity). -/
def PracticalClass.label : PracticalClass → String
  | .finiteDecidable => "finite_decidable"
  | .primitiveRecursive => "primitive_recursive"
  | .boundedSearch => "bounded_search"
  | .totalRecursive => "total_recursive"
  | .semidecidable => "semidecidable"
  | .coSemidecidable => "co_semidecidable"
  | .oracleRelative => "oracle_relative"
  | .classicalPropositional => "classical_propositional"
  | .noncomputableExistence => "noncomputable_existence"
  | .unclassified => "unclassified"

/-- KERN-10 legacy alias target. -/
def legacyClassicalNoncomputableExistenceLabel : String := "classical_noncomputable_existence"

def normalizeLabel (s : String) : String :=
  if s = legacyClassicalNoncomputableExistenceLabel then
    PracticalClass.noncomputableExistence.label
  else
    s

/-- Parse only production practical labels (fail closed via Option). -/
def parsePractical? (s : String) : Option PracticalClass :=
  let t := normalizeLabel s
  if t = "finite_decidable" then some .finiteDecidable
  else if t = "primitive_recursive" then some .primitiveRecursive
  else if t = "bounded_search" then some .boundedSearch
  else if t = "total_recursive" then some .totalRecursive
  else if t = "semidecidable" then some .semidecidable
  else if t = "co_semidecidable" then some .coSemidecidable
  else if t = "oracle_relative" then some .oracleRelative
  else if t = "classical_propositional" then some .classicalPropositional
  else if t = "noncomputable_existence" then some .noncomputableExistence
  else if t = "unclassified" then some .unclassified
  else none

/-- `rm_research:*` is never admitted by the practical parser. -/
def rejectsRmResearchPrefix (s : String) : Bool :=
  s.startsWith "rm_research:"

-- Fixture label parity (matches computability_classification_fixtures.v1.json).
#guard PracticalClass.finiteDecidable.label = "finite_decidable"
#guard PracticalClass.primitiveRecursive.label = "primitive_recursive"
#guard PracticalClass.boundedSearch.label = "bounded_search"
#guard PracticalClass.totalRecursive.label = "total_recursive"
#guard PracticalClass.semidecidable.label = "semidecidable"
#guard PracticalClass.coSemidecidable.label = "co_semidecidable"
#guard PracticalClass.oracleRelative.label = "oracle_relative"
#guard PracticalClass.classicalPropositional.label = "classical_propositional"
#guard PracticalClass.noncomputableExistence.label = "noncomputable_existence"
#guard PracticalClass.unclassified.label = "unclassified"
#guard normalizeLabel "classical_noncomputable_existence" = "noncomputable_existence"
#guard (parsePractical? "finite_decidable") = some .finiteDecidable
#guard (parsePractical? "classical_noncomputable_existence") = some .noncomputableExistence
#guard (parsePractical? "quantum_hyperarithmetical") = none
#guard rejectsRmResearchPrefix "rm_research:RCA0" = true
#guard rejectsRmResearchPrefix "finite_decidable" = false

/-- KERN-12 schema id (Python SCHEMA constant). -/
def classificationSchema : String := "computability_classification/v1"
#guard classificationSchema = "computability_classification/v1"

/-- Evidence-contract ids mirrored as strings for documentation parity. -/
def evidenceFiniteVss : String := "finite_vss_certificate"
def evidenceFiniteSearch : String := "finite_search_bound"
def evidenceBoundedClosure : String := "bounded_closure_certificate"
def evidencePrimitiveRecursive : String := "primitive_recursive_bound"
def evidenceTotalRecursive : String := "total_recursive_certificate"
def evidenceSemidecision : String := "semidecision_procedure"
def evidenceCoSemidecision : String := "co_semidecision_procedure"
def evidenceOracle : String := "oracle_declaration"
def evidenceClassicalProp : String := "classical_propositional_marker"
def evidenceNoncomputableEx : String := "noncomputable_existence_claim"
def evidenceUnclassified : String := "explicit_unclassified"
def evidenceInterpretationPackage : String := "interpretation_package"
def evidencePrintAxiomsOnly : String := "print_axioms_only"

#guard evidencePrintAxiomsOnly = "print_axioms_only"
#guard evidenceInterpretationPackage = "interpretation_package"

/--
  Non-implication law (documentation-level): finite decidability does not
  license an RCA0 label. Enforced in Python evidence contracts; Lean records
  the string constant so parity tests can bind both sides.
-/
def nonImplicationFiniteDecidableNotRca0 : String := "finite_decidable_does_not_imply_RCA0"
#guard nonImplicationFiniteDecidableNotRca0 = "finite_decidable_does_not_imply_RCA0"

end LeverProofLean.ComputabilityClassification
