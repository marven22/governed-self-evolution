# M14 V5 Restart Handoff

## Repository

The current branch is pushed through commit `4adda97`, which fixes the paired
confirmation bug in branch--certify control. The subsequent working tree was
clean at restart preparation.

## Completed evidence

- V5 one-edit calibration: 36 attempts, 3 certified commits.
- Transactional archive: 36 records at
  `/home/vmargapu/experiments/m14_transactional_v5_calibration_v1/archive.json`.
- Fresh parents 131--134: BC, R1, and R3 checkpoints complete.
- Corrected branch--certify development run: 8 of 12 trajectories complete
  before intentional shutdown. Those trajectory JSON files are durable.

## Important validity note

The earlier `m14_branch_certify_development_v1_invalid_confirmation` directory
contains invalid results: confirmation accidentally compared the edited branch
to itself. Preserve it for provenance; do not analyze it. The corrected runner
is `scripts/run_m14_branch_certify_v5.py`.

## Resume command

After WSL/MuJoCo is available again, resume safely; completed trajectory files
are skipped and only unfinished controller states are rerun:

```bash
cd /mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution
bash scripts/run_m14_branch_certify_development_wsl.sh
```

## Next scientific step

After all 12 corrected development trajectories complete, audit the archive of
branch outcomes, quantify certified sequential gains and retention, then fit
the archive-conditioned proposal/dynamics model before any new blind bank is
touched.
