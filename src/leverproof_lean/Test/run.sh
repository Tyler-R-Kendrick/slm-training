#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
certificate=$(mktemp "${TMPDIR:-/tmp}/leverproof-lean.XXXXXX.json")
trap 'rm -f "$certificate"' EXIT HUP INT TERM

"$project_dir/.lake/build/bin/leverproof-lean" \
  check "$project_dir/Test/resource.json" >"$certificate"
"$project_dir/.lake/build/bin/leverproof-lean" \
  verify "$project_dir/Test/resource.json" "$certificate"
grep -q '"selected_candidate": "candidate"' "$certificate"

"$project_dir/.lake/build/bin/leverproof-lean" \
  check "$project_dir/Test/success-before-params.json" >"$certificate"
"$project_dir/.lake/build/bin/leverproof-lean" \
  verify "$project_dir/Test/success-before-params.json" "$certificate"
grep -q '"selected_candidate": "success-winner"' "$certificate"

"$project_dir/.lake/build/bin/leverproof-lean" \
  check "$project_dir/Test/bands.json" >"$certificate"
"$project_dir/.lake/build/bin/leverproof-lean" \
  verify "$project_dir/Test/bands.json" "$certificate"
grep -q '"relation": "mixed"' "$certificate"
grep -q '"below": 1' "$certificate"
grep -q '"within": 1' "$certificate"
grep -q '"above": 1' "$certificate"
