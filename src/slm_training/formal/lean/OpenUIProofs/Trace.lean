import OpenUIProofs.Forest

namespace OpenUIProofs.Trace

open OpenUIProofs.Forest

structure Step where
  beforeRemoved : Finset Candidate
  afterRemoved : Finset Candidate
  certified : Finset Candidate
  beforeHistory : List Event
  afterHistory : List Event

def validStep (step : Step) : Bool :=
  step.afterRemoved = step.beforeRemoved ∪ step.certified &&
    step.beforeHistory.isPrefixOf step.afterHistory

def validTrace : List Step → Bool
  | [] => true
  | [step] => validStep step
  | step₁ :: step₂ :: rest =>
      validStep step₁ &&
      step₁.afterRemoved = step₂.beforeRemoved &&
      step₁.afterHistory = step₂.beforeHistory &&
      validTrace (step₂ :: rest)

theorem valid_step_sound (step : Step) (h : validStep step = true) :
    step.afterRemoved = step.beforeRemoved ∪ step.certified ∧
      step.beforeHistory <+: step.afterHistory := by
  simpa [validStep] using h

theorem valid_trace_all_steps (steps : List Step) (h : validTrace steps = true) :
    ∀ step ∈ steps, step.afterRemoved = step.beforeRemoved ∪ step.certified ∧
      step.beforeHistory <+: step.afterHistory := by
  induction steps with
  | nil =>
      simp
  | cons head tail ih =>
      cases tail with
      | nil =>
          simpa [validTrace] using valid_step_sound head h
      | cons next rest =>
          have hparts :
              ((validStep head = true ∧ head.afterRemoved = next.beforeRemoved) ∧
                head.afterHistory = next.beforeHistory) ∧
                validTrace (next :: rest) = true := by
            simpa [validTrace] using h
          have hhead : validStep head = true := by
            exact hparts.1.1.1
          have htail : validTrace (next :: rest) = true := by
            exact hparts.2
          intro step hmem
          obtain hstep | hstep := List.mem_cons.mp hmem
          · subst step
            exact valid_step_sound head hhead
          · exact ih htail step hstep

end OpenUIProofs.Trace
