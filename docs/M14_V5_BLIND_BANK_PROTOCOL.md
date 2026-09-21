# M14 V5 Blind Controller Bank

## Purpose

Parents 135--138 are the final blind bank for the V5 branch-governor study.
They are independent of V5 development parents 131--134.  No outcome from an
edit applied to this bank may be used to fit, tune, or select the governor.

## Construction

For every parent seed, run the fixed MT2 protocol: behavior cloning followed
by three DAgger rounds. Preserve BC, round-1, and round-3 checkpoints. The
three stages of a parent are correlated and remain in the same blind partition.

## Evaluation rule

Freeze the governor and its decision thresholds before reading any blind-bank
edit outcome. For each stage, permit the governor to propose one typed edit or
HOLD, then apply the same paired branch--certify rule used in development.
HOLD, a fixed grammar choice, and an exhaustive safe oracle are analysis
baselines. The oracle is never available to the governor at decision time.
