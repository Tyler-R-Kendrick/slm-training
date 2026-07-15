# Cross-seed example-loss check (2026-07-15)

A second two-step probe on the identical data manifest produced a different
set of highest-loss examples, with zero overlap against the first seed's top
examples. The observed outliers are therefore not evidence of one corrupt
record. No records were removed or reweighted; future work should test the
broader recipe/model behavior with longer, multi-seed runs.
