# RLCompOpt Blame-Gated Development Ledger Audit

## Scope

This is an initial foundation collection under
`rlcompopt-blame-gated-transition-ledger-v1`. It used only the eight-program
development cohort in `compilergym_program_split_v2.json`. It did not use the
selection cohort or the previously touched fixed-controller held-out cohort.

The run artifact is outside Git because it contains detailed CompilerGym traces:

```text
/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1/
  blame_gated_development_transactions.json
sha256: 50ba0201ec67e939fa0abab1e33e203372825d4f4a3f12713fc7374cacfd2039
```

## Structural audit

| Check | Result |
|---|---:|
| Development programs | 8 |
| Matched transactions | 96 |
| Children per program | 12 distinct recipes |
| Trusted parent transactions | 96 / 96 |
| Semantically certified children | 96 / 96 |
| Insert operations | 28 |
| Delete operations | 24 |
| Replace operations | 20 |
| Splice operations | 24 |

Each parent was evaluated twice before attribution. A parent was trusted only
when both runs passed cBench output-equivalence validation, agreed in reward,
and had non-negative reward.

## Outcome distribution

| Outcome | Count |
|---|---:|
| Certified improvement | 6 |
| Certified neutral | 62 |
| Certified regression with blame | 28 |
| Unsafe rejection | 0 |

The dataset therefore contains meaningful negative attribution rather than
only positive examples. For example, several late deletions, donor-rank-four
insertions/replacements, and larger splices accumulated substantial blame.

## Interpretation and limitation

This audit establishes a functioning, certificate-backed transaction collector
and demonstrates that the grammar exposes both beneficial and harmful edits.
It does **not** establish a trained governor or a final self-evolution result.

The absence of unsafe-rejection examples is expected for this conservative
grammar and is a limitation for a future safety-probability head. The next
collection expansion should add bounded boundary-stress candidates, retain the
same hard certificate rule, and use the selection cohort only for calibration.
The final governor evaluation requires a newly reserved cohort rather than the
already touched fixed-controller held-out programs.
