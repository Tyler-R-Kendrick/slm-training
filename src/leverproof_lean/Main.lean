import LeverProofLean.Protocol

open Lean
open LeverProofLean

def loadEvidence (path : String) : IO Json := do
  let source ← IO.FS.readFile path
  IO.ofExcept (Json.parse source)

def loadJson (path : String) : IO Json := do
  let source ← IO.FS.readFile path
  IO.ofExcept (Json.parse source)

def commandCheck (evidencePath : String) : IO UInt32 := do
  let evidence ← loadEvidence evidencePath
  let certificate ← IO.ofExcept (computeCertificateJson evidence)
  IO.println certificate.pretty
  pure 0

def commandVerify
    (evidencePath certificatePath : String) : IO UInt32 := do
  let evidence ← loadEvidence evidencePath
  let expected ← IO.ofExcept (computeCertificateJson evidence)
  let actual ← loadJson certificatePath
  if actual == expected then
    IO.println "verified"
    pure 0
  else
    IO.eprintln "leverproof-lean: certificate does not replay from evidence"
    pure 1

def usage : IO UInt32 := do
  IO.eprintln "usage: leverproof-lean check EVIDENCE.json | \
    leverproof-lean verify EVIDENCE.json CERTIFICATE.json"
  pure 2

def main (args : List String) : IO UInt32 :=
  try
    match args with
    | ["check", evidencePath] => commandCheck evidencePath
    | ["verify", evidencePath, certificatePath] =>
        commandVerify evidencePath certificatePath
    | _ => usage
  catch error =>
    IO.eprintln s!"leverproof-lean: {error}"
    pure 1
