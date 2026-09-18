# M14 Controller Bank Protocol (v1)

## Goal

Create independent, capable two-task MT2 controllers from which to collect
held-out self-evolution transitions. Independence means separate stochastic
training runs, not differently updated copies of one checkpoint.

## Construction

Each seed follows the already validated two-stage procedure:

1. Behavior clone 100 expert episodes per task for 5,000 updates.
2. Perform three DAgger rounds, each with ten rollouts per task and 2,000
   correction updates.

The bank launcher fixes Python, NumPy, and Torch seeds and records results in
`/home/vmargapu/experiments/m14_controller_bank_v1/seed<seed>/`.

```bash
bash scripts/build_m14_controller_bank_wsl.sh 103 104 105 106 107 108
```

It can be rerun safely: completed behavior-cloning and DAgger stages are
skipped. `controller_manifest.json` is refreshed after every seed.

## Qualification rule

A controller qualifies only when its final DAgger evaluation has at least 0.60
success on both reach and pick-place. This is deliberately a screen, not a
claim of equal skill. Controllers below threshold are preserved in the manifest
but excluded from the main transition-learning set.

## Split rule

After enough controllers qualify, allocate whole controller identities—not
individual transition rows—to development and held-out partitions. All updates
starting from a held-out controller belong exclusively to held-out evaluation.
