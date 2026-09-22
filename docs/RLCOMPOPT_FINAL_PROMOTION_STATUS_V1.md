# Final Governor Promotion Status

## Status: blocked by benchmark coverage, not by a governor result

The CompilerGym cBench registry has exactly 18 programs that support the
reference-output semantic certificate required by this project. They were all
already allocated to the development, selection, or previously touched
fixed-controller held-out cohorts in `compilergym_program_split_v2.json`.

The remaining cBench programs (`adpcm`, `ghostscript`, `ispell`, `lame`, and
`rijndael`) do not expose `benchmark_semantics_validated=true` through the
CompilerGym validator. They cannot supply a fail-closed final promotion test.

The partial `final_challenge_incumbent_v2.json` run is invalid evidence and
must not be reported or used for model selection.

## Required next step

Before any final challenger promotion claim, register a separate benchmark
family with a deterministic semantic-equivalence certificate, then freeze a
program-blocked final cohort before running either governor. The existing
paired-repair and promotion-audit scripts can be reused once that contract is
in place.
